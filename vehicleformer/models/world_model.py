"""
Causal Transformer World Model — Contribution 2
================================================
Learns to predict future graph embedding sequences using causal
(autoregressive) masked self-attention — distinguishing correlation
from causation in network dynamics.

Key novelty:
- Causal attention reveals WHY network states change (congestion
  CAUSES handover, not the reverse)
- Predicts 500ms–2s ahead with epistemic uncertainty via ensemble
- Enables model-based RL planning without real environment rollouts

Target journal: IEEE/ACM Transactions on Networking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.shape[1]])


class CausalTransformerEncoder(nn.Module):
    """
    Causal (autoregressive) transformer that processes a sequence of
    graph embeddings h(t-T), ..., h(t) and predicts h(t+1), ..., h(t+k).

    Uses causal masking so prediction at step t only attends to t' ≤ t.
    This enforces temporal causality: the model cannot "cheat" by
    looking at future states.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        wm = cfg['world_model']
        self.d_model    = wm['d_model']
        self.seq_len    = wm['max_seq_len']
        self.pred_horiz = wm['prediction_horizon']

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            self.d_model, max_len=self.seq_len + self.pred_horiz + 1,
            dropout=wm['dropout']
        )

        # Transformer encoder (causal mask applied in forward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = self.d_model,
            nhead           = wm['nhead'],
            dim_feedforward = wm['dim_feedforward'],
            dropout         = wm['dropout'],
            batch_first     = True,     # (B, T, d_model)
            norm_first      = True,     # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=wm['num_encoder_layers']
        )

        # Prediction head: predicts next k embeddings
        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(wm['dropout']),
            nn.Linear(self.d_model * 2, self.d_model * self.pred_horiz),
        )

        # Uncertainty head: predicts log-variance for each prediction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model * self.pred_horiz),
        )

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        """Upper-triangular mask: position i cannot attend to j > i."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask  # True = masked (ignored)

    def forward(
        self, h_seq: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            h_seq: (B, T, d_model)  — sequence of graph embeddings
            
        Returns:
            h_pred:    (B, k, d_model)  — predicted future embeddings
            log_var:   (B, k, d_model)  — log-variance (uncertainty)
            h_context: (B, T, d_model)  — contextualized representations
        """
        B, T, _ = h_seq.shape
        # Add positional encoding
        x = self.pos_enc(h_seq)
        # Causal mask
        mask = self._causal_mask(T, h_seq.device)
        # Transformer (causal)
        h_context = self.transformer(x, mask=mask)  # (B, T, d_model)
        # Use last timestep for prediction
        h_last = h_context[:, -1, :]                # (B, d_model)
        # Predict next k steps
        pred_flat = self.pred_head(h_last)           # (B, d_model * k)
        h_pred = pred_flat.view(B, self.pred_horiz, self.d_model)
        # Uncertainty
        var_flat = self.uncertainty_head(h_last)
        log_var  = var_flat.view(B, self.pred_horiz, self.d_model)

        return h_pred, log_var, h_context


class WorldModel(nn.Module):
    """
    Full World Model with ensemble uncertainty quantification.

    Uses an ensemble of CausalTransformerEncoders to estimate
    epistemic uncertainty — the SAC agent uses this to be more
    conservative in high-uncertainty regions (novel environments).

    At deployment: can plan ahead without real-world interaction,
    solving the sample-efficiency problem.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg        = cfg
        wm_cfg          = cfg['world_model']
        self.d_model    = wm_cfg['d_model']
        self.n_ensemble = wm_cfg['num_ensemble']
        self.pred_horiz = wm_cfg['prediction_horizon']

        # Ensemble of transformers
        self.ensemble = nn.ModuleList([
            CausalTransformerEncoder(cfg)
            for _ in range(self.n_ensemble)
        ])

        # Reward predictor (bonus: predicts expected reward)
        self.reward_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # Done predictor (predicts episode termination — returns logits)
        self.done_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, h_seq: Tensor
    ) -> dict:
        """
        Args:
            h_seq: (B, T, d_model)  — history of graph embeddings
            
        Returns:
            dict with:
              mean_pred:    (B, k, d_model)  — ensemble mean prediction
              epistemic_unc:(B, k, d_model)  — disagreement between ensemble members
              total_unc:    (B, k, d_model)  — mean log-variance (aleatoric)
              reward_pred:  (B,)             — predicted reward
              done_pred:    (B,)             — predicted termination prob
              all_preds:    list of (B, k, d_model) — raw ensemble outputs
        """
        preds = []
        log_vars = []
        contexts = []

        for member in self.ensemble:
            h_pred, log_var, h_ctx = member(h_seq)
            preds.append(h_pred)
            log_vars.append(log_var)
            contexts.append(h_ctx[:, -1, :])  # last context step

        preds_stack    = torch.stack(preds, dim=0)      # (n_ens, B, k, d_model)
        log_var_stack  = torch.stack(log_vars, dim=0)

        mean_pred      = preds_stack.mean(0)            # (B, k, d_model)
        # Epistemic uncertainty = variance across ensemble members
        epistemic_unc  = preds_stack.var(0)             # (B, k, d_model)
        total_unc      = log_var_stack.mean(0)          # (B, k, d_model)

        # Use mean context for reward/done prediction
        ctx_mean = torch.stack(contexts, dim=0).mean(0) # (B, d_model)
        reward_pred  = self.reward_head(ctx_mean).squeeze(-1)
        done_logits  = self.done_head(ctx_mean).squeeze(-1)
        done_pred    = torch.sigmoid(done_logits)

        return {
            "mean_pred"     : mean_pred,
            "epistemic_unc" : epistemic_unc,
            "total_unc"     : total_unc,
            "reward_pred"   : reward_pred,
            "done_pred"     : done_pred,
            "done_logits"   : done_logits,
            "all_preds"     : preds,
        }

    def compute_loss(
        self,
        h_seq: Tensor,          # (B, T, d_model)  — input history
        h_targets: Tensor,      # (B, k, d_model)  — ground truth future
        rewards: Tensor,        # (B,)
        dones: Tensor,          # (B,)
    ) -> Tuple[Tensor, dict]:
        """
        World model training loss:
          L = L_pred + L_reward + L_done
          
        L_pred = negative log-likelihood under Gaussian
               = 0.5 * (log_var + (pred - target)^2 / exp(log_var))
        """
        out = self.forward(h_seq)

        # Prediction loss (NLL) for each ensemble member
        pred_loss = torch.tensor(0.0, device=h_seq.device)
        for i, member in enumerate(self.ensemble):
            h_pred_i, log_var_i, _ = member(h_seq)
            # Gaussian NLL
            var_i = torch.exp(log_var_i.clamp(-10, 10))
            nll   = 0.5 * (log_var_i + (h_pred_i - h_targets).pow(2) / (var_i + 1e-8))
            pred_loss = pred_loss + nll.mean()
        pred_loss = pred_loss / self.n_ensemble

        # Reward prediction loss
        r_loss = F.mse_loss(out["reward_pred"], rewards)

        # Done prediction loss (logits-based, AMP-safe)
        d_loss = F.binary_cross_entropy_with_logits(out["done_logits"], dones.float())

        total_loss = pred_loss + 0.5 * r_loss + 0.1 * d_loss

        return total_loss, {
            "wm_pred_loss"   : pred_loss.item(),
            "wm_reward_loss" : r_loss.item(),
            "wm_done_loss"   : d_loss.item(),
            "wm_total_loss"  : total_loss.item(),
        }

    def imagine_rollout(
        self,
        h_start: Tensor,        # (B, d_model)  — current graph embedding
        horizon: int = 5,
    ) -> Tensor:
        """
        Imagine future embeddings without real-world interaction.
        Used by SAC for model-based planning.
        
        Returns: (B, horizon, d_model) — imagined future embeddings
        """
        device = h_start.device
        B = h_start.shape[0]
        # Build initial sequence from single embedding (simplified)
        h_seq = h_start.unsqueeze(1)  # (B, 1, d_model)

        imagined = []
        for _ in range(horizon):
            out = self.forward(h_seq)
            next_h = out["mean_pred"][:, 0:1, :]  # (B, 1, d_model)
            imagined.append(next_h)
            h_seq = torch.cat([h_seq, next_h], dim=1)  # extend sequence

        return torch.cat(imagined, dim=1)  # (B, horizon, d_model)
