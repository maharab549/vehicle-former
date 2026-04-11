"""
Replay Buffer for VehicleFormer SAC Training.
Stores (obs, action, reward, next_obs, done) transitions.
"""
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Dict


class ReplayBuffer:
    """Efficient circular replay buffer with graph embedding support."""

    def __init__(self, cfg: dict, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity   = cfg['replay_buffer']['capacity']
        self.device     = device
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.emb_dim    = cfg['hetgnn']['embedding_dim']
        self.ptr = 0
        self.size = 0

        # Pre-allocate tensors (much faster than list append)
        self.obs      = np.zeros((self.capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((self.capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((self.capacity, 1),          dtype=np.float32)
        # Cached graph embeddings (optional - speeds up training)
        self.embeddings      = np.zeros((self.capacity, self.emb_dim), dtype=np.float32)
        self.next_embeddings = np.zeros((self.capacity, self.emb_dim), dtype=np.float32)
        self._has_embeddings = False
        rb_cfg = cfg.get('replay_buffer', {})
        self.prioritized = rb_cfg.get('prioritized', True)
        self.priority_alpha = float(rb_cfg.get('priority_alpha', 0.6))
        self.priority_beta_start = float(rb_cfg.get('priority_beta_start', 0.4))
        self.priority_beta_frames = int(rb_cfg.get('priority_beta_frames', 200000))
        self.priority_eps = float(rb_cfg.get('priority_eps', 1e-6))
        self.priorities = np.ones((self.capacity,), dtype=np.float32)
        self._sample_count = 0

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        embedding:      np.ndarray = None,
        next_embedding: np.ndarray = None,
    ) -> None:
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        if embedding is not None:
            self.embeddings[self.ptr]      = embedding
            self.next_embeddings[self.ptr] = next_embedding
            self._has_embeddings = True
        self.priorities[self.ptr] = self.priorities.max() if self.size > 0 else 1.0
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _beta(self) -> float:
        """Anneal importance-sampling correction from beta_start to 1."""
        frac = min(1.0, self._sample_count / max(self.priority_beta_frames, 1))
        return self.priority_beta_start + frac * (1.0 - self.priority_beta_start)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        if self.prioritized:
            scaled = self.priorities[:self.size] ** self.priority_alpha
            probs = scaled / (scaled.sum() + 1e-8)
            idx = np.random.choice(self.size, size=batch_size, p=probs)
            self._sample_count += 1
            beta = self._beta()
            weights = (self.size * probs[idx]) ** (-beta)
            weights = weights / (weights.max() + 1e-8)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones((batch_size,), dtype=np.float32)
        def t(arr): return torch.tensor(arr[idx], dtype=torch.float32, device=self.device)
        batch = {
            "obs"      : t(self.obs),
            "actions"  : t(self.actions),
            "rewards"  : t(self.rewards),
            "next_obs" : t(self.next_obs),
            "dones"    : t(self.dones),
            "weights"  : torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "indices"  : idx,
        }
        if self._has_embeddings:
            batch["embeddings"]      = t(self.embeddings)
            batch["next_embeddings"] = t(self.next_embeddings)
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update PER priorities from fresh TD errors."""
        if not self.prioritized:
            return
        priorities = np.abs(td_errors).reshape(-1) + self.priority_eps
        self.priorities[indices] = priorities.astype(np.float32)

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
