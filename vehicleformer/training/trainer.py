"""
VehicleFormer Main Training Loop
=================================
Integrates: ICVEnvironment + HetGNN + WorldModel + SAC + ReplayBuffer
"""
import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from vehicleformer.env.icv_env import ICVEnvironment
from vehicleformer.models.hetgnn import HetGNNEncoder, GraphBuilder
from vehicleformer.models.world_model import WorldModel
from vehicleformer.models.sac_agent import SACAgent
from vehicleformer.models.llm_prior import LLMPolicyPrior
from vehicleformer.training.replay_buffer import ReplayBuffer
from vehicleformer.utils.logger import Logger
from vehicleformer.utils.metrics import MetricsTracker

console = Console()


class VehicleFormerTrainer:
    """
    Full training pipeline for VehicleFormer.
    
    Training phases:
      Phase 1 (M1-3):  Warm-up + baseline SAC (no world model, no LLM)
      Phase 2 (M4-6):  Add HetGNN encoder (Contribution 1)
      Phase 3 (M7-10): Add World Model (Contribution 2)
      Phase 4 (M11+):  Add LLM Prior (Contribution 3)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        proj = cfg['project']
        self.device = torch.device(proj['device'] if torch.cuda.is_available() else 'cpu')
        console.print(f"[cyan]Device:[/cyan] {self.device}")
        if self.device.type == 'cuda':
            console.print(f"[cyan]GPU:[/cyan] {torch.cuda.get_device_name(0)}")
            console.print(f"[cyan]VRAM:[/cyan] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Set seeds
        torch.manual_seed(proj['seed'])
        np.random.seed(proj['seed'])

        # ─── GPU Performance Tuning ───────────────────────────────────
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
            console.print("[green]✓ GPU optimizations: cudnn.benchmark + TF32 + medium matmul[/green]")

        # ─── Environment ──────────────────────────────────────────────
        self.env  = ICVEnvironment(cfg)
        self.eval_env = ICVEnvironment(cfg)
        obs_dim   = self.env.observation_space.shape[0]
        act_dim   = cfg['action']['action_dim']
        emb_dim   = cfg['hetgnn']['embedding_dim']

        console.print(f"[cyan]Obs dim:[/cyan]  {obs_dim}")
        console.print(f"[cyan]Act dim:[/cyan]  {act_dim}")
        console.print(f"[cyan]Emb dim:[/cyan]  {emb_dim}")

        # ─── Models ───────────────────────────────────────────────────
        # Ablation flags
        self.use_hetgnn = cfg['hetgnn'].get('enabled', True)
        self.use_world_model = cfg['world_model'].get('enabled', True)

        if self.use_hetgnn:
            self.graph_builder = GraphBuilder(cfg, self.device)
            self.hetgnn        = HetGNNEncoder(cfg).to(self.device)
        else:
            # Ablation: replace HetGNN with simple linear projection
            self.graph_builder = None
            self.hetgnn = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, emb_dim),
            ).to(self.device)
            console.print("[yellow]⚠ HetGNN DISABLED — using flat MLP projection (ablation)[/yellow]")

        self.world_model   = WorldModel(cfg).to(self.device)
        if not self.use_world_model:
            console.print("[yellow]⚠ World Model DISABLED (ablation)[/yellow]")

        self.agent         = SACAgent(cfg, self.device)

        # ─── LLM Prior (Contribution 3) ──────────────────────────────
        self.llm_enabled = cfg['llm_prior'].get('enabled', False)
        self.llm_prior = None
        self.llm_opt = None
        if self.llm_enabled:
            console.print("[yellow]Loading LLM Policy Prior...[/yellow]")
            self.llm_prior = LLMPolicyPrior(cfg, self.device)
            self.llm_prior.load_model()
            self.llm_opt = torch.optim.Adam(
                self.llm_prior.get_trainable_parameters(),
                lr=1e-4, weight_decay=1e-5
            )
            self.llm_kl_weight = cfg['llm_prior']['kl_weight']
            self.llm_update_freq = cfg['llm_prior']['update_frequency']
            self.llm_infer_freq = cfg['llm_prior'].get('inference_frequency', 10)
            self._cached_llm_log_prob = None
            console.print("[green]✓ LLM Prior ready[/green]")

        # ─── Optimizers for encoder + world model ─────────────────────
        self.gnn_opt = torch.optim.Adam(
            self.hetgnn.parameters(), lr=3e-4, weight_decay=1e-5
        )
        self.wm_opt = torch.optim.Adam(
            self.world_model.parameters(), lr=3e-4, weight_decay=1e-5
        )

        # ─── Replay buffer ────────────────────────────────────────────
        self.buffer = ReplayBuffer(cfg, obs_dim, act_dim, self.device)

        # ─── Logging ──────────────────────────────────────────────────
        self.logger  = Logger(cfg)
        self.metrics = MetricsTracker(cfg)

        # ─── Training state ───────────────────────────────────────────
        self.total_steps   = 0
        self.episodes_done = 0
        self.best_eval_reward = -np.inf

        self._print_model_sizes()

    def resume_from_checkpoint(self, tag_or_path: str):
        """
        Resume training from a saved checkpoint.
        
        Args:
            tag_or_path: Either a tag like 'best' or 'step_150000',
                         or a full path to the agent checkpoint file.
        """
        ckpt_dir = Path(self.cfg['project']['checkpoint_dir'])

        # Resolve tag to file paths
        if os.path.isfile(tag_or_path):
            agent_path = Path(tag_or_path)
            tag = agent_path.stem.replace("agent_", "")
        else:
            tag = tag_or_path
            agent_path = ckpt_dir / f"agent_{tag}.pt"

        hetgnn_path = ckpt_dir / f"hetgnn_{tag}.pt"
        wm_path     = ckpt_dir / f"worldmodel_{tag}.pt"
        llm_path    = ckpt_dir / f"llm_prior_{tag}.pt"

        # Load agent (actor, critic, critic_target, optimizers)
        if agent_path.exists():
            self.agent.load(str(agent_path))
            # Override optimizer LRs with current config (for fine-tuning)
            sac_cfg = self.cfg['sac']
            for pg in self.agent.actor_opt.param_groups:
                pg['lr'] = sac_cfg['actor_lr']
            for pg in self.agent.critic_opt.param_groups:
                pg['lr'] = sac_cfg['critic_lr']
            console.print(f"  [green]✓ Agent LRs overridden: actor={sac_cfg['actor_lr']}, critic={sac_cfg['critic_lr']}[/green]")
        else:
            console.print(f"  [red]✗ Agent checkpoint not found: {agent_path}[/red]")

        # Load HetGNN encoder
        if hetgnn_path.exists():
            self.hetgnn.load_state_dict(torch.load(hetgnn_path, map_location=self.device))
            console.print(f"  [green]✓ HetGNN loaded from {hetgnn_path}[/green]")

        # Load world model
        if wm_path.exists():
            self.world_model.load_state_dict(torch.load(wm_path, map_location=self.device))
            console.print(f"  [green]✓ World model loaded from {wm_path}[/green]")

        # Load LLM prior
        if self.llm_enabled and self.llm_prior is not None and llm_path.exists():
            self.llm_prior.load(str(llm_path))
            console.print(f"  [green]✓ LLM prior loaded from {llm_path}[/green]")

        console.print(f"[bold green]✓ Resumed from checkpoint '{tag}'[/bold green]")

    def _print_model_sizes(self):
        """Print parameter counts for each model."""
        def count(m): return sum(p.numel() for p in m.parameters()) / 1e6
        table = Table(title="Model Parameter Counts", style="cyan")
        table.add_column("Model",      style="white")
        table.add_column("Parameters", style="green")
        table.add_row("HetGNN Encoder",    f"{count(self.hetgnn):.2f}M")
        table.add_row("World Model",       f"{count(self.world_model):.2f}M")
        table.add_row("SAC Actor",         f"{count(self.agent.actor):.2f}M")
        table.add_row("SAC Critic",        f"{count(self.agent.critic):.2f}M")
        if self.llm_prior is not None:
            trainable_count = sum(p.numel() for p in self.llm_prior.get_trainable_parameters()) / 1e6
            table.add_row("LLM Prior (trainable)", f"{trainable_count:.2f}M")
        console.print(table)

    # ─── Embedding Helper ────────────────────────────────────────────────

    @torch.no_grad()
    def _get_embedding(self, obs: np.ndarray) -> np.ndarray:
        """Convert flat obs → embedding (via HetGNN or flat MLP)."""
        if self.use_hetgnn:
            graph = self.graph_builder.build(obs)
            emb, _ = self.hetgnn(graph["node_features"], graph["positions"])
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            emb = self.hetgnn(obs_t)
        return emb.cpu().numpy()

    def _get_embedding_batch(self, obs_batch: np.ndarray) -> torch.Tensor:
        """Batch version for training."""
        if self.use_hetgnn:
            graph = self.graph_builder.build_batch(obs_batch)
            emb, _ = self.hetgnn(graph["node_features"], graph["positions"])
            B = obs_batch.shape[0]
            max_v = self.cfg['observation']['max_vehicles']
            emb = emb.view(B, max_v, -1).mean(1)  # (B, emb_dim)
        else:
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            emb = self.hetgnn(obs_t)  # (B, emb_dim)
        return emb

    # ─── Main Training Loop ──────────────────────────────────────────────

    def train(self):
        cfg_t = self.cfg['training']
        total_steps = cfg_t['total_timesteps']
        warmup_steps = cfg_t['warm_up_steps']
        min_buffer   = self.cfg['replay_buffer']['min_size']

        console.rule("[bold cyan]VehicleFormer Training Started[/bold cyan]")
        console.print(f"Target: [green]{total_steps:,}[/green] total steps")
        console.print(f"Warm-up: [yellow]{warmup_steps:,}[/yellow] random steps\n")

        obs, _ = self.env.reset()
        emb = self._get_embedding(obs)

        episode_reward = 0
        episode_steps  = 0
        episode_start  = time.time()

        # AMP GradScaler for mixed-precision training
        self._amp_scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Steps: {task.completed:>7,}/{task.total:,}"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Training VehicleFormer", total=total_steps)

            while self.total_steps < total_steps:

                # ─── Action selection ──────────────────────────────────
                if self.total_steps < warmup_steps:
                    action = self.env.action_space.sample()     # random warm-up
                else:
                    emb_t = torch.tensor(emb, device=self.device)
                    action = self.agent.select_action(emb_t, deterministic=False)

                # ─── Environment step ──────────────────────────────────
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_emb = self._get_embedding(next_obs)

                # Store transition
                self.buffer.add(obs, action, reward, next_obs, done, emb, next_emb)

                obs = next_obs
                emb = next_emb
                episode_reward += reward
                episode_steps  += 1
                self.total_steps += 1
                progress.advance(task)

                # ─── Update networks ───────────────────────────────────
                if (self.buffer.is_ready(min_buffer)
                        and self.total_steps >= warmup_steps):
                    grad_steps = self.cfg['sac'].get('gradient_steps', 1)
                    for _ in range(grad_steps):
                        losses = self._update_step()
                    if self.total_steps % cfg_t['log_frequency'] == 0:
                        self.logger.log_losses(losses, self.total_steps)

                # ─── Episode done ──────────────────────────────────────
                if done:
                    self.episodes_done += 1
                    ep_time = time.time() - episode_start
                    ep_metrics = info.get("episode_metrics", {})

                    self.logger.log_episode(
                        episode       = self.episodes_done,
                        reward        = episode_reward,
                        steps         = episode_steps,
                        duration      = ep_time,
                        extra_metrics = ep_metrics,
                    )

                    if self.episodes_done % 10 == 0:
                        console.print(
                            f"  Ep {self.episodes_done:4d} | "
                            f"Steps {self.total_steps:7,} | "
                            f"Reward: [{'green' if episode_reward > 0 else 'red'}]{episode_reward:+.2f}[/] | "
                            f"P99 latency: {ep_metrics.get('p99_latency_ms', 0):.1f}ms | "
                            f"SLA met: {ep_metrics.get('sla_50ms_met_pct', 0):.1f}%"
                        )

                    obs, _ = self.env.reset()
                    emb = self._get_embedding(obs)
                    episode_reward = 0
                    episode_steps  = 0
                    episode_start  = time.time()

                # ─── Periodic evaluation ───────────────────────────────
                if self.total_steps % cfg_t['eval_frequency'] == 0:
                    eval_reward = self.evaluate(num_episodes=cfg_t['num_eval_episodes'])
                    console.print(
                        f"\n[bold yellow]EVAL @ step {self.total_steps:,}[/bold yellow] | "
                        f"Mean reward: [bold green]{eval_reward:.3f}[/bold green]"
                        + (" [NEW BEST!]" if eval_reward > self.best_eval_reward else "")
                    )
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        self._save_checkpoint("best")

                # ─── Periodic checkpoint ───────────────────────────────
                if self.total_steps % cfg_t['save_frequency'] == 0:
                    self._save_checkpoint(f"step_{self.total_steps}")

        console.rule("[bold green]Training Complete![/bold green]")
        self._save_checkpoint("final")
        self.env.close()
        self.eval_env.close()

    # ─── Network Update Step ─────────────────────────────────────────────

    def _update_step(self) -> dict:
        """One gradient update step for all networks."""
        batch_size = self.cfg['sac']['batch_size']
        batch = self.buffer.sample(batch_size)
        losses = {}

        # ─── Get embeddings (use cached if available) ─────────────────
        if "embeddings" in batch:
            h      = batch["embeddings"]
            h_next = batch["next_embeddings"]
        else:
            h      = self._get_embedding_batch(batch["obs"].cpu().numpy()).to(self.device)
            h_next = self._get_embedding_batch(batch["next_obs"].cpu().numpy()).to(self.device)

        r    = batch["rewards"]
        a    = batch["actions"]
        done = batch["dones"]

        # ─── Update World Model (AMP) ─────────────────────────────────
        if self.use_world_model:
            # Build sequences from batch (simplified: treat as T=1 sequences)
            h_seq = h.unsqueeze(1)          # (B, 1, emb_dim)
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                wm_loss, wm_info = self.world_model.compute_loss(
                    h_seq, h_next.unsqueeze(1), r.squeeze(-1), done.squeeze(-1)
                )
            self.wm_opt.zero_grad()
            self._amp_scaler.scale(wm_loss).backward()
            self._amp_scaler.unscale_(self.wm_opt)
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self._amp_scaler.step(self.wm_opt)
            self._amp_scaler.update()
            losses.update(wm_info)

        # ─── Update SAC Critic (AMP) ──────────────────────────────────
        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            critic_info = self.agent.update_critic(h.detach(), a, r, h_next.detach(), done)
        losses.update(critic_info)

        # ─── Update LLM Prior & get log-probs for KL ─────────────────
        llm_log_prob = None
        if self.llm_enabled and self.llm_prior is not None:
            # Compute LLM log-prob only every N steps (expensive), cache between
            if self.total_steps % self.llm_infer_freq == 0 or self._cached_llm_log_prob is None:
                with torch.no_grad():
                    a_curr, _, _ = self.agent.actor.sample(h.detach())
                    a_curr = self.agent.physics(a_curr)
                    self._cached_llm_log_prob = self.llm_prior.get_log_prob(h.detach(), a_curr)
            llm_log_prob = self._cached_llm_log_prob.detach()

            # Update LLM prior periodically (fine-tune projection + LoRA)
            if self.total_steps % self.llm_update_freq == 0:
                llm_pred_mean, llm_pred_logstd = self.llm_prior(h.detach())
                llm_std = llm_pred_logstd.exp()
                llm_dist = torch.distributions.Normal(llm_pred_mean, llm_std)
                with torch.no_grad():
                    target_actions, _, _ = self.agent.actor.sample(h.detach())
                    target_actions = self.agent.physics(target_actions)
                    target_pre_tanh = torch.atanh(target_actions.clamp(-0.999, 0.999))
                llm_loss = -llm_dist.log_prob(target_pre_tanh).mean()
                self.llm_opt.zero_grad()
                llm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.llm_prior.get_trainable_parameters(), 1.0)
                self.llm_opt.step()
                losses["llm_prior_loss"] = llm_loss.item()

        # ─── Update SAC Actor (AMP) ───────────────────────────────────
        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            actor_info = self.agent.update_actor(
                h.detach(),
                llm_log_prob=llm_log_prob,
                kl_weight=self.llm_kl_weight if self.llm_enabled else 0.0,
            )
        losses.update(actor_info)

        # ─── Soft update target networks ──────────────────────────────
        self.agent.soft_update_target()

        return losses

    # ─── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self, num_episodes: int = 10) -> float:
        """Run evaluation episodes and return mean reward."""
        rewards = []
        kpis    = []
        self.hetgnn.eval()

        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            ep_reward = 0
            done = False
            while not done:
                emb = self._get_embedding(obs)
                emb_t = torch.tensor(emb, device=self.device)
                action = self.agent.select_action(emb_t, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
            rewards.append(ep_reward)
            if "episode_metrics" in info:
                kpis.append(info["episode_metrics"])

        self.hetgnn.train()
        self.logger.log_eval(np.mean(rewards), kpis, self.total_steps)
        return float(np.mean(rewards))

    # ─── Checkpoint ──────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        ckpt_dir = Path(self.cfg['project']['checkpoint_dir'])
        ckpt_dir.mkdir(exist_ok=True)
        self.agent.save(str(ckpt_dir / f"agent_{tag}.pt"))
        torch.save(self.hetgnn.state_dict(),     ckpt_dir / f"hetgnn_{tag}.pt")
        torch.save(self.world_model.state_dict(), ckpt_dir / f"worldmodel_{tag}.pt")
        if self.llm_prior is not None:
            self.llm_prior.save(str(ckpt_dir / f"llm_prior_{tag}.pt"))
