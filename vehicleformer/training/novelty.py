"""Novelty-oriented training utilities for robustness and generalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DomainProfile:
    """Sampled domain shift parameters for one episode."""

    outage_probability: float
    congestion_probability: float
    congestion_magnitude_low: float
    congestion_magnitude_high: float
    sat_latency_bias_ms: float

    def as_dict(self) -> Dict[str, float]:
        """Convert to primitive dictionary."""
        return {
            "outage_probability": float(self.outage_probability),
            "congestion_probability": float(self.congestion_probability),
            "congestion_magnitude_low": float(self.congestion_magnitude_low),
            "congestion_magnitude_high": float(self.congestion_magnitude_high),
            "sat_latency_bias_ms": float(self.sat_latency_bias_ms),
        }


class DomainRandomizer:
    """Episode-level domain randomization for traffic and channel conditions."""

    def __init__(self, cfg: dict, seed: int):
        novelty_cfg = cfg.get("novelty", {}).get("domain_randomization", {})
        self.enabled = novelty_cfg.get("enabled", False)
        self.curriculum_enabled = novelty_cfg.get("curriculum_enabled", True)
        self.difficulty = float(novelty_cfg.get("initial_difficulty", 0.2))
        self.difficulty_step = float(novelty_cfg.get("difficulty_step", 0.05))
        self.max_difficulty = float(novelty_cfg.get("max_difficulty", 1.0))
        self.min_difficulty = float(novelty_cfg.get("min_difficulty", 0.0))
        self.outage_prob_range = novelty_cfg.get("outage_prob_range", [0.001, 0.02])
        self.congestion_prob_range = novelty_cfg.get("congestion_prob_range", [0.002, 0.03])
        self.congestion_mag_low_range = novelty_cfg.get("congestion_mag_low_range", [0.1, 0.35])
        self.congestion_mag_high_range = novelty_cfg.get("congestion_mag_high_range", [0.2, 0.6])
        self.sat_latency_bias_range = novelty_cfg.get("sat_latency_bias_range_ms", [0.0, 12.0])
        self.rng = np.random.default_rng(seed)

    def set_difficulty(self, value: float) -> None:
        """Set curriculum difficulty in [min_difficulty, max_difficulty]."""
        self.difficulty = float(np.clip(value, self.min_difficulty, self.max_difficulty))

    def update_from_metrics(self, episode_metrics: Dict[str, float]) -> float:
        """Update difficulty based on episode KPI success indicators."""
        if not self.enabled or not self.curriculum_enabled:
            return self.difficulty
        sla = float(episode_metrics.get("sla_50ms_met_pct", 0.0))
        pdr = float(episode_metrics.get("mean_pdr", episode_metrics.get("mean_reliability", 0.0)))
        if sla > 88.0 and pdr > 0.93:
            self.difficulty = min(self.max_difficulty, self.difficulty + self.difficulty_step)
        elif sla < 70.0 or pdr < 0.85:
            self.difficulty = max(self.min_difficulty, self.difficulty - self.difficulty_step)
        return self.difficulty

    def _sample_with_difficulty(self, value_range: list) -> float:
        """Sample inside a range scaled by curriculum difficulty."""
        low, high = float(value_range[0]), float(value_range[1])
        span = (high - low) * self.difficulty
        return float(self.rng.uniform(low, low + span if span > 1e-8 else low + 1e-8))

    def sample(self) -> DomainProfile:
        """Sample one domain profile for an episode."""
        if not self.enabled:
            return DomainProfile(0.002, 0.003, 0.15, 0.35, 0.0)
        low = self._sample_with_difficulty(self.congestion_mag_low_range)
        high = max(low + 0.05, self._sample_with_difficulty(self.congestion_mag_high_range))
        return DomainProfile(
            outage_probability=self._sample_with_difficulty(self.outage_prob_range),
            congestion_probability=self._sample_with_difficulty(self.congestion_prob_range),
            congestion_magnitude_low=low,
            congestion_magnitude_high=high,
            sat_latency_bias_ms=self._sample_with_difficulty(self.sat_latency_bias_range),
        )


class InvarianceRegularizer:
    """Encourages embeddings to remain stable under nuisance perturbations."""

    def __init__(self, cfg: dict, device: torch.device):
        inv_cfg = cfg.get("novelty", {}).get("invariance", {})
        self.enabled = inv_cfg.get("enabled", False)
        self.weight = float(inv_cfg.get("weight", 0.05))
        self.noise_std = float(inv_cfg.get("noise_std", 0.015))
        self.mask_probability = float(inv_cfg.get("feature_mask_probability", 0.08))
        self.device = device

    def augment(self, obs_batch: np.ndarray) -> np.ndarray:
        """Apply small feature noise and random masking."""
        if not self.enabled:
            return obs_batch
        noisy = obs_batch + np.random.normal(0.0, self.noise_std, size=obs_batch.shape).astype(np.float32)
        mask = np.random.binomial(1, 1.0 - self.mask_probability, size=obs_batch.shape).astype(np.float32)
        return noisy * mask

    def loss(self, base_embeddings: torch.Tensor, aug_embeddings: torch.Tensor) -> torch.Tensor:
        """Return cosine-invariant consistency loss."""
        if not self.enabled:
            return torch.zeros(1, device=self.device).squeeze(0)
        base = F.normalize(base_embeddings, dim=-1)
        aug = F.normalize(aug_embeddings, dim=-1)
        return self.weight * F.mse_loss(base, aug)


class RNDExplorer:
    """Random Network Distillation for intrinsic motivation."""

    def __init__(self, cfg: dict, embedding_dim: int, device: torch.device):
        rnd_cfg = cfg.get("novelty", {}).get("exploration", {})
        self.enabled = rnd_cfg.get("enabled", False)
        self.device = device
        self.intrinsic_scale = float(rnd_cfg.get("intrinsic_reward_scale", 0.05))
        hidden = int(rnd_cfg.get("hidden_dim", 256))
        self.target = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
        ).to(device)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        lr = float(rnd_cfg.get("lr", 1e-4))
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def intrinsic_reward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward from prediction error."""
        if not self.enabled:
            return torch.zeros((embeddings.shape[0], 1), device=self.device)
        with torch.no_grad():
            target = self.target(embeddings)
        pred = self.predictor(embeddings)
        err = torch.mean((pred - target) ** 2, dim=-1, keepdim=True)
        return self.intrinsic_scale * err

    def update(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Update predictor network on current embeddings."""
        if not self.enabled:
            return torch.zeros(1, device=self.device).squeeze(0)
        with torch.no_grad():
            target = self.target(embeddings)
        pred = self.predictor(embeddings)
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        return loss.detach()


class PolicySmoothnessRegularizer:
    """Encourage local smoothness of the actor under embedding perturbations."""

    def __init__(self, cfg: dict, device: torch.device):
        smooth_cfg = cfg.get("novelty", {}).get("policy_smoothness", {})
        self.enabled = smooth_cfg.get("enabled", True)
        self.weight = float(smooth_cfg.get("weight", 0.03))
        self.noise_std = float(smooth_cfg.get("noise_std", 0.02))
        self.device = device

    def loss(self, actor, embeddings: torch.Tensor) -> torch.Tensor:
        """MSE between action means at clean and perturbed embeddings."""
        if not self.enabled:
            return torch.zeros(1, device=self.device).squeeze(0)
        noise = torch.randn_like(embeddings) * self.noise_std
        mean_clean, _ = actor(embeddings)
        mean_pert, _ = actor(embeddings + noise)
        return self.weight * F.mse_loss(torch.tanh(mean_clean), torch.tanh(mean_pert))