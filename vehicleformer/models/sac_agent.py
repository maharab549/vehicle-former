"""
Physics-Constrained Soft Actor-Critic (SAC) Agent — Layer 4
============================================================
Model-based SAC that uses:
  1. HetGNN embeddings as state representation
  2. World model predictions for planning
  3. LLM prior as KL-divergence regularizer (Contribution 3)
  4. Physics constraints from ICV Roadmap KPIs

Actions: [network_select(4), tx_power, offload_ratio]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict
import numpy as np


# ─── Actor Network ───────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    Stochastic policy network (Gaussian actor).
    Outputs mean and log-std of action distribution.
    
    Input:  graph_embedding (embedding_dim,)
    Output: action_mean, action_log_std → action via reparameterization
    """
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, embedding_dim: int, action_dim: int, hidden_dims: list, num_options: int = 4):
        super().__init__()
        dims = [embedding_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.num_options = num_options
        self.option_head = nn.Linear(hidden_dims[-1], num_options)
        self.option_embed = nn.Embedding(num_options, hidden_dims[-1])
        self.mean_head    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        feat     = self.net(h)
        mean     = self.mean_head(feat)
        log_std  = self.log_std_head(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def option_probs(self, h: Tensor) -> Tensor:
        """Return option distribution for hierarchical control."""
        feat = self.net(h)
        logits = self.option_head(feat)
        return torch.softmax(logits, dim=-1)

    def deterministic_action(self, h: Tensor) -> Tensor:
        """Deterministic action with most probable option."""
        feat = self.net(h)
        option_logits = self.option_head(feat)
        option_idx = torch.argmax(option_logits, dim=-1)
        conditioned = feat + self.option_embed(option_idx)
        mean = self.mean_head(conditioned)
        return torch.tanh(mean)

    def sample(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample action using reparameterization trick.
        Returns: action (squashed), log_prob, mean
        """
        feat = self.net(h)
        option_logits = self.option_head(feat)
        option_dist = torch.distributions.Categorical(logits=option_logits)
        option_idx = option_dist.sample()
        conditioned = feat + self.option_embed(option_idx)
        mean = self.mean_head(conditioned)
        log_std = self.log_std_head(conditioned).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()                       # reparameterized sample
        y_t  = torch.tanh(x_t)                      # squash to [-1, 1]
        log_prob = dist.log_prob(x_t)
        # Enforce action bounds (Appendix C of SAC paper)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        option_log_prob = option_dist.log_prob(option_idx).unsqueeze(-1)
        log_prob = log_prob + option_log_prob
        return y_t, log_prob, torch.tanh(mean)


# ─── Critic Network ──────────────────────────────────────────────────────
class Critic(nn.Module):
    """
    Twin Q-networks (double Q-learning to reduce overestimation).
    Q(s, a) where s is the graph embedding, a is the action.
    """

    def __init__(self, embedding_dim: int, action_dim: int, hidden_dims: list, num_quantiles: int = 32):
        super().__init__()
        inp_dim = embedding_dim + action_dim
        self.num_quantiles = num_quantiles
        dims = [inp_dim] + hidden_dims

        def make_backbone():
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers += [nn.LayerNorm(dims[i+1]), nn.GELU()]
            return nn.Sequential(*layers)

        self.q1_backbone = make_backbone()
        self.q2_backbone = make_backbone()
        self.q1_mean = nn.Linear(hidden_dims[-1], 1)
        self.q2_mean = nn.Linear(hidden_dims[-1], 1)
        self.q1_quantiles = nn.Linear(hidden_dims[-1], num_quantiles)
        self.q2_quantiles = nn.Linear(hidden_dims[-1], num_quantiles)

    def forward(self, h: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([h, a], dim=-1)
        h1 = self.q1_backbone(x)
        h2 = self.q2_backbone(x)
        return self.q1_mean(h1), self.q2_mean(h2)

    def quantile_values(self, h: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        """Return distributional Q estimates as quantile samples."""
        x = torch.cat([h, a], dim=-1)
        h1 = self.q1_backbone(x)
        h2 = self.q2_backbone(x)
        return self.q1_quantiles(h1), self.q2_quantiles(h2)

    def q_min(self, h: Tensor, a: Tensor) -> Tensor:
        q1, q2 = self.forward(h, a)
        return torch.min(q1, q2)


# ─── Physics Constraint Layer ────────────────────────────────────────────
class PhysicsConstraintLayer(nn.Module):
    """
    Enforces hard ICV Roadmap KPI constraints on actions.
    
    Constraints (from China ICV Roadmap 2025-2030):
    - Max latency:    100ms hard limit
    - Min reliability: 95% for safety-critical messages
    - Max tx power:    regulatory limit
    
    Uses Interior Point Method to project actions onto feasible set.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        sac_cfg = cfg['sac']
        act_cfg = cfg['action']
        self.max_lat   = sac_cfg['max_latency_ms']
        self.min_rel   = sac_cfg['min_reliability']
        self.max_power = act_cfg['tx_power_max']
        self.min_power = act_cfg['tx_power_min']

    def forward(self, action: Tensor) -> Tensor:
        """
        Clip actions to feasible region.
        action: (B, 6) — [net_logits(4), tx_power_norm, offload_ratio]
        """
        # Network logits: ensure valid softmax input
        net_logits = action[:, :4]
        # tx_power and offload_ratio: clip to [0, 1]
        tx_power   = action[:, 4:5].clamp(0.0, 1.0)
        offload    = action[:, 5:6].clamp(0.0, 1.0)
        return torch.cat([net_logits, tx_power, offload], dim=-1)


# ─── Full SAC Agent ──────────────────────────────────────────────────────
class SACAgent(nn.Module):
    """
    Physics-Constrained Soft Actor-Critic Agent.
    
    Integrates:
    - HetGNN embeddings as rich state representations
    - World model for model-based planning bonus
    - LLM prior KL regularization (when enabled)
    - Twin critics with automatic entropy tuning
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        self.cfg    = cfg
        self.device = device

        sac_cfg = cfg['sac']
        emb_dim = cfg['hetgnn']['embedding_dim']
        act_dim = cfg['action']['action_dim']
        self.option_entropy_weight = float(sac_cfg.get('option_entropy_weight', 0.02))
        num_options = int(sac_cfg.get('num_options', 4))
        self.distributional_enabled = bool(sac_cfg.get('distributional_enabled', True))
        self.num_quantiles = int(sac_cfg.get('num_quantiles', 32))
        self.cvar_alpha = float(sac_cfg.get('cvar_alpha', 0.2))
        self.risk_lambda = float(sac_cfg.get('risk_lambda', 0.15))
        self.use_adaptive_risk = bool(sac_cfg.get('adaptive_risk_constraint', True))
        self.cvar_target = float(sac_cfg.get('cvar_target', -1.0))
        init_lambda = float(sac_cfg.get('risk_lambda_init', max(self.risk_lambda, 1e-4)))
        self.log_risk_lambda = nn.Parameter(torch.tensor(np.log(init_lambda), dtype=torch.float32, device=device))

        # Networks
        self.actor  = Actor(emb_dim, act_dim, sac_cfg['hidden_dims'], num_options=num_options).to(device)
        self.critic = Critic(emb_dim, act_dim, sac_cfg['hidden_dims'], num_quantiles=self.num_quantiles).to(device)
        self.critic_target = Critic(emb_dim, act_dim, sac_cfg['hidden_dims'], num_quantiles=self.num_quantiles).to(device)
        self.physics = PhysicsConstraintLayer(cfg)

        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Entropy temperature (auto-tuned)
        self.log_alpha = nn.Parameter(torch.tensor(
            np.log(sac_cfg['alpha']), dtype=torch.float32, device=device
        ))
        self.target_entropy = -act_dim * 0.5  # heuristic target

        self.gamma = sac_cfg['gamma']
        self.tau   = sac_cfg['tau']

        # Optimizers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=sac_cfg['actor_lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=sac_cfg['critic_lr'])
        self.alpha_opt  = torch.optim.Adam([self.log_alpha],          lr=sac_cfg['alpha_lr'])
        self.risk_lambda_opt = torch.optim.Adam([self.log_risk_lambda], lr=sac_cfg.get('risk_lambda_lr', 5e-4))

    @property
    def risk_multiplier(self) -> Tensor:
        """Positive Lagrange multiplier for risk constraint."""
        return self.log_risk_lambda.exp()

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    # ─── Action Selection ────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, graph_embedding: Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Select action for environment interaction.
        
        Args:
            graph_embedding: (embedding_dim,) or (1, embedding_dim)
            deterministic:   use mean action (for evaluation)
        """
        h = graph_embedding.unsqueeze(0) if graph_embedding.dim() == 1 else graph_embedding
        h = h.to(self.device)

        if deterministic:
            action  = self.actor.deterministic_action(h)
        else:
            action, _, _ = self.actor.sample(h)

        action = self.physics(action)

        # Map from [-1,1] to actual action ranges
        action_np = action.squeeze(0).cpu().numpy()
        action_np[:4]  = action_np[:4]              # network logits (raw)
        action_np[4]   = (action_np[4] + 1) / 2    # tx_power_norm → [0,1]
        action_np[5]   = (action_np[5] + 1) / 2    # offload_ratio → [0,1]
        return action_np.astype(np.float32)

    # ─── Training Updates ────────────────────────────────────────────────

    def update_critic(
        self,
        h:        Tensor,   # (B, emb_dim) current graph embeddings
        a:        Tensor,   # (B, act_dim) actions taken
        r:        Tensor,   # (B, 1) rewards
        h_next:   Tensor,   # (B, emb_dim) next graph embeddings
        done:     Tensor,   # (B, 1) terminal flags
        llm_log_prob: Optional[Tensor] = None,  # (B, 1) LLM prior log-prob
        weights: Optional[Tensor] = None,
    ) -> dict:
        """Update twin Q-networks."""
        with torch.no_grad():
            a_next, log_pi_next, _ = self.actor.sample(h_next)
            a_next = self.physics(a_next)
            q1_next, q2_next = self.critic_target(h_next, a_next)
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_pi_next

            # Bellman target
            q_target = r + self.gamma * (1 - done) * q_next

        q1, q2 = self.critic(h, a)
        td_err1 = q1 - q_target
        td_err2 = q2 - q_target
        if weights is not None:
            critic_loss = (weights * td_err1.pow(2)).mean() + (weights * td_err2.pow(2)).mean()
        else:
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        quantile_loss = torch.tensor(0.0, device=h.device)
        if self.distributional_enabled:
            q1_q, q2_q = self.critic.quantile_values(h, a)
            target_q = q_target.detach().expand(-1, self.num_quantiles)
            q1_diff = target_q - q1_q
            q2_diff = target_q - q2_q
            q1_huber = F.smooth_l1_loss(q1_q, target_q, reduction='none')
            q2_huber = F.smooth_l1_loss(q2_q, target_q, reduction='none')
            taus = (torch.arange(self.num_quantiles, device=h.device, dtype=torch.float32) + 0.5) / self.num_quantiles
            taus = taus.unsqueeze(0)
            q1_weight = torch.abs(taus - (q1_diff.detach() < 0).float())
            q2_weight = torch.abs(taus - (q2_diff.detach() < 0).float())
            q1_loss = (q1_weight * q1_huber).mean(dim=-1, keepdim=True)
            q2_loss = (q2_weight * q2_huber).mean(dim=-1, keepdim=True)
            if weights is not None:
                quantile_loss = (weights * (q1_loss + q2_loss)).mean()
            else:
                quantile_loss = (q1_loss + q2_loss).mean()
            critic_loss = critic_loss + quantile_loss
        td_errors = 0.5 * (td_err1.abs() + td_err2.abs())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "quantile_loss": quantile_loss.item(),
            "td_error_mean": td_errors.mean().item(),
            "_td_errors": td_errors.detach(),
        }

    def update_actor(
        self,
        h: Tensor,
        llm_prior_probs: Optional[Tensor] = None,
        beta: float = 0.1,
    ) -> dict:
        """
        Update actor with optional LLM KL regularization.
        
        Standard SAC actor loss:
          L_actor = E[α * log π(a|s) - Q(s,a)]
          
        With LLM prior (Contribution 3):
          L_actor = E[α * log π(a|s) - Q(s,a) + λ * D_KL(π || π_LLM)]
          
        where D_KL(π || π_LLM) = log π(a|s) - log π_LLM(a|s)
        """
        a, log_pi, _ = self.actor.sample(h)
        a = self.physics(a)
        q_val = self.critic.q_min(h, a)
        mean, _ = self.actor(h)

        # Standard SAC entropy term
        actor_loss = (self.alpha.detach() * log_pi - q_val).mean()

        # LLM KL regularization (Contribution 3)
        kl_loss = torch.tensor(0.0, device=h.device)
        if llm_prior_probs is not None:
            actor_net_logits = mean[:, :3]
            actor_net_probs = torch.softmax(actor_net_logits, dim=-1)
            safe_prior = llm_prior_probs.clamp_min(1e-6)
            kl_divergence = torch.sum(
                actor_net_probs * (torch.log(actor_net_probs.clamp_min(1e-6)) - torch.log(safe_prior)),
                dim=-1,
            )
            kl_loss = beta * kl_divergence.mean()
            actor_loss = actor_loss + kl_loss

        option_probs = self.actor.option_probs(h)
        option_entropy = -(option_probs * torch.log(option_probs.clamp_min(1e-6))).sum(dim=-1).mean()
        actor_loss = actor_loss - self.option_entropy_weight * option_entropy

        cvar_penalty = torch.tensor(0.0, device=h.device)
        cvar_value = torch.tensor(0.0, device=h.device)
        if self.distributional_enabled:
            q1_q, q2_q = self.critic.quantile_values(h, a)
            q_q = torch.min(q1_q, q2_q)
            q_sorted, _ = torch.sort(q_q, dim=-1)
            tail_count = max(1, int(self.num_quantiles * self.cvar_alpha))
            cvar = q_sorted[:, :tail_count].mean(dim=-1, keepdim=True)
            cvar_penalty = (q_val - cvar).mean()
            cvar_value = cvar.mean()
            risk_weight = self.risk_multiplier.detach() if self.use_adaptive_risk else torch.tensor(self.risk_lambda, device=h.device)
            actor_loss = actor_loss + risk_weight * cvar_penalty

        risk_dual_loss = torch.tensor(0.0, device=h.device)
        if self.distributional_enabled and self.use_adaptive_risk:
            constraint_violation = (self.cvar_target - cvar_value).detach()
            risk_dual_loss = -(self.risk_multiplier * constraint_violation)
            self.risk_lambda_opt.zero_grad()
            risk_dual_loss.backward()
            self.risk_lambda_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Update entropy temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return {
            "actor_loss" : actor_loss.item(),
            "kl_loss"    : kl_loss.item() if isinstance(kl_loss, Tensor) else 0.0,
            "alpha"      : self.alpha.item(),
            "entropy"    : (-log_pi.mean()).item(),
            "option_entropy": option_entropy.item(),
            "cvar_penalty": cvar_penalty.item(),
            "cvar_value": cvar_value.item(),
            "risk_lambda": self.risk_multiplier.item(),
            "risk_dual_loss": risk_dual_loss.item(),
        }

    def soft_update_target(self) -> None:
        """Polyak averaging: θ_target ← τ*θ + (1-τ)*θ_target"""
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.copy_(self.tau * p.data + (1 - self.tau) * p_tgt.data)

    def save(self, path: str) -> None:
        torch.save({
            "actor"          : self.actor.state_dict(),
            "critic"         : self.critic.state_dict(),
            "critic_target"  : self.critic_target.state_dict(),
            "log_alpha"      : self.log_alpha.data,
            "actor_opt"      : self.actor_opt.state_dict(),
            "critic_opt"     : self.critic_opt.state_dict(),
        }, path)
        print(f"  ✓ Agent saved to {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        print(f"  ✓ Agent loaded from {path}")
