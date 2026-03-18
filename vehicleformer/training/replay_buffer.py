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
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        def t(arr): return torch.tensor(arr[idx], dtype=torch.float32, device=self.device)
        batch = {
            "obs"      : t(self.obs),
            "actions"  : t(self.actions),
            "rewards"  : t(self.rewards),
            "next_obs" : t(self.next_obs),
            "dones"    : t(self.dones),
        }
        if self._has_embeddings:
            batch["embeddings"]      = t(self.embeddings)
            batch["next_embeddings"] = t(self.next_embeddings)
        return batch

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
