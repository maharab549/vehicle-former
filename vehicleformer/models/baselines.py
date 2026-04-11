"""Baseline models and evaluation utilities for VehicleFormer experiments."""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from vehicleformer.env.icv_env import ICVEnvironment


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible baseline runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discrete_to_env_action(network_id: int, action_dim: int) -> np.ndarray:
    """Map a discrete network choice to the environment action vector."""
    action = np.zeros(action_dim, dtype=np.float32)
    action[min(network_id, 2)] = 1.0
    if action_dim > 4:
        action[4] = 0.5
    if action_dim > 5:
        action[5] = 0.5
    return action


def flatten_state(obs: np.ndarray, cfg: dict) -> np.ndarray:
    """Extract a compact baseline state from the full observation tensor."""
    observation_cfg = cfg["observation"]
    vehicle_dim = observation_cfg["vehicle_dim"]
    max_vehicles = observation_cfg["max_vehicles"]
    max_rsus = observation_cfg["max_rsus"]
    max_bs = observation_cfg["max_base_stations"]
    rsu_dim = observation_cfg["rsu_dim"]
    bs_dim = observation_cfg["bs_dim"]
    vehicle_block = obs[: vehicle_dim * max_vehicles].reshape(max_vehicles, vehicle_dim)
    ego = vehicle_block[0, :10]
    offset = vehicle_dim * max_vehicles + rsu_dim * max_rsus + bs_dim * max_bs
    network_obs = obs[offset : offset + 18]
    return np.concatenate([ego, network_obs]).astype(np.float32)


def observation_dim(cfg: dict) -> int:
    """Return the environment observation dimension from config."""
    observation_cfg = cfg["observation"]
    return (
        observation_cfg["max_vehicles"] * observation_cfg["vehicle_dim"]
        + observation_cfg["max_rsus"] * observation_cfg["rsu_dim"]
        + observation_cfg["max_base_stations"] * observation_cfg["bs_dim"]
        + 3 * 6 + 4
    )


class ReplayMemory:
    """Simple replay buffer for DQN-style baselines."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: Deque[Tuple[np.ndarray, object, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        """Store a transition."""
        self.memory.append(tuple(transition))

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return the current buffer size."""
        return len(self.memory)


class MLPQNetwork(nn.Module):
    """Small MLP used by the DQN baselines."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all discrete actions."""
        return self.net(state)


class FlatActor(nn.Module):
    """Continuous actor for the vanilla SAC baseline."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log standard deviation."""
        features = self.backbone(state)
        return self.mean(features), self.log_std(features).clamp(-5, 2)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a squashed action and its log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class FlatCritic(nn.Module):
    """Twin Q-network for the vanilla SAC baseline."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return twin Q estimates."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class HomogeneousGraphEncoder(nn.Module):
    """Lightweight GraphSAGE-like encoder over the flattened observation graph."""

    def __init__(self, cfg: dict, hidden_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        max_feat = max(cfg["observation"]["vehicle_dim"], cfg["observation"]["rsu_dim"], cfg["observation"]["bs_dim"])
        self.proj = nn.Linear(max_feat, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Encode the observation as a homogeneous graph embedding."""
        embeddings = []
        for obs in obs_batch:
            nodes, positions = self._nodes_from_obs(obs.detach().cpu().numpy())
            node_tensor = torch.tensor(nodes, dtype=torch.float32, device=obs_batch.device)
            hidden = self.proj(node_tensor)
            adjacency = self._adjacency(positions, obs_batch.device)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            neighbor_mean = adjacency @ hidden / degree
            updated = torch.relu(self.update(torch.cat([hidden, neighbor_mean], dim=-1)))
            embeddings.append(updated.mean(dim=0))
        return torch.stack(embeddings, dim=0)

    def _nodes_from_obs(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recover padded node features and positions from an observation vector."""
        cfg = self.cfg["observation"]
        vehicle_dim = cfg["vehicle_dim"]
        rsu_dim = cfg["rsu_dim"]
        bs_dim = cfg["bs_dim"]
        max_v = cfg["max_vehicles"]
        max_r = cfg["max_rsus"]
        max_b = cfg["max_base_stations"]
        max_feat = max(vehicle_dim, rsu_dim, bs_dim)
        v_end = vehicle_dim * max_v
        r_end = v_end + rsu_dim * max_r
        b_end = r_end + bs_dim * max_b
        vehicles = obs[:v_end].reshape(max_v, vehicle_dim)
        rsus = obs[v_end:r_end].reshape(max_r, rsu_dim)
        bss = obs[r_end:b_end].reshape(max_b, bs_dim)
        nodes = []
        positions = []
        for block in (vehicles, rsus, bss):
            for row in block:
                padded = np.zeros(max_feat, dtype=np.float32)
                padded[: row.shape[0]] = row
                nodes.append(padded)
                positions.append(row[:2])
        return np.asarray(nodes, dtype=np.float32), np.asarray(positions, dtype=np.float32)

    @staticmethod
    def _adjacency(positions: np.ndarray, device: torch.device) -> torch.Tensor:
        """Build a dense proximity adjacency matrix."""
        pos = torch.tensor(positions, dtype=torch.float32, device=device)
        distance = torch.cdist(pos, pos)
        adjacency = (distance < 0.35).float()
        adjacency.fill_diagonal_(1.0)
        return adjacency


@dataclass
class EvaluationResult:
    """Container for baseline evaluation results."""

    rewards: List[float]
    episode_metrics: List[Dict[str, float]]


class BaselinePolicy:
    """Base class for all baselines in the paper suite."""

    name = "baseline"

    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or cfg["project"].get("device", "cpu"))
        self.action_dim = int(cfg["action"]["action_dim"])
        self.state_dim = len(flatten_state(np.zeros(observation_dim(cfg), dtype=np.float32), cfg))

    def train(self, total_episodes: int = 10, seed: int = 42, checkpoint_dir: Optional[Path] = None) -> Dict[str, float]:
        """Train the baseline policy."""
        raise NotImplementedError

    def evaluate(self, num_episodes: int = 10, seed: int = 42) -> EvaluationResult:
        """Evaluate the baseline policy."""
        env = ICVEnvironment(self.cfg)
        set_seed(seed)
        rewards: List[float] = []
        metrics: List[Dict[str, float]] = []
        try:
            for episode_idx in range(num_episodes):
                obs, _ = env.reset(seed=seed + episode_idx)
                done = False
                reward_sum = 0.0
                info = {}
                while not done:
                    action = self.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_sum += reward
                    done = terminated or truncated
                rewards.append(float(reward_sum))
                metrics.append(info.get("episode_metrics", {}))
        finally:
            env.close()
        return EvaluationResult(rewards=rewards, episode_metrics=metrics)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return an action for the given observation."""
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    """Uniform random network selector baseline."""

    name = "random"

    def train(self, total_episodes: int = 0, seed: int = 42, checkpoint_dir: Optional[Path] = None) -> Dict[str, float]:
        """Random policy has no trainable parameters."""
        return {"episodes": 0, "seed": seed}

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample a uniformly random network."""
        return discrete_to_env_action(np.random.randint(0, 3), self.action_dim)


class GreedyHeuristic(BaselinePolicy):
    """Select the network with the strongest current channel reading."""

    name = "greedy"

    def train(self, total_episodes: int = 0, seed: int = 42, checkpoint_dir: Optional[Path] = None) -> Dict[str, float]:
        """Greedy heuristic is non-parametric."""
        return {"episodes": 0, "seed": seed}

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Pick the network with the highest normalized RSSI/SINR score."""
        state = flatten_state(obs, self.cfg)
        network_obs = state[10:28].reshape(3, 6)
        score = network_obs[:, 0] + 0.5 * network_obs[:, 1] - 0.2 * network_obs[:, 2]
        return discrete_to_env_action(int(np.argmax(score)), self.action_dim)


class VanillaDQN(BaselinePolicy):
    """DQN baseline over flat state features."""

    name = "dqn"

    def __init__(self, cfg: dict, device: Optional[str] = None):
        super().__init__(cfg, device)
        self.q = MLPQNetwork(self.state_dim).to(self.device)
        self.target_q = MLPQNetwork(self.state_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        self.memory = ReplayMemory(50000)
        self.gamma = 0.99
        self.epsilon = 1.0

    def train(self, total_episodes: int = 25, seed: int = 42, checkpoint_dir: Optional[Path] = None) -> Dict[str, float]:
        """Train the DQN baseline with epsilon-greedy exploration."""
        env = ICVEnvironment(self.cfg)
        set_seed(seed)
        episode_rewards: List[float] = []
        try:
            for episode in range(total_episodes):
                obs, _ = env.reset(seed=seed + episode)
                state = flatten_state(obs, self.cfg)
                done = False
                episode_reward = 0.0
                while not done:
                    if np.random.rand() < self.epsilon:
                        action_id = np.random.randint(0, 3)
                    else:
                        with torch.no_grad():
                            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                            action_id = int(torch.argmax(self.q(state_t), dim=-1).item())
                    next_obs, reward, terminated, truncated, _ = env.step(discrete_to_env_action(action_id, self.action_dim))
                    next_state = flatten_state(next_obs, self.cfg)
                    done = terminated or truncated
                    self.memory.push(state, action_id, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    self._update_if_ready()
                self.epsilon = max(0.05, self.epsilon * 0.95)
                episode_rewards.append(float(episode_reward))
                if episode % 5 == 0:
                    self.target_q.load_state_dict(self.q.state_dict())
        finally:
            env.close()
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.q.state_dict(), checkpoint_dir / "model.pt")
            with open(checkpoint_dir / "training_curve.csv", "w", encoding="utf-8") as handle:
                handle.write("episode,reward\n")
                for idx, reward in enumerate(episode_rewards, start=1):
                    handle.write(f"{idx},{reward:.6f}\n")
        return {"episodes": total_episodes, "seed": seed}

    def _update_if_ready(self, batch_size: int = 64) -> None:
        """Run a single DQN update when enough data is available."""
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        q_values = self.q(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_q(next_states_t).max(dim=-1, keepdim=True).values
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q
        loss = F.mse_loss(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return the greedy discrete action."""
        state = flatten_state(obs, self.cfg)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_id = int(torch.argmax(self.q(state_t), dim=-1).item())
        return discrete_to_env_action(action_id, self.action_dim)


class VanillaSACBaseline(BaselinePolicy):
    """Vanilla SAC baseline with a flat MLP encoder only."""

    name = "sac_mlp"

    def __init__(self, cfg: dict, device: Optional[str] = None):
        super().__init__(cfg, device)
        self.actor = FlatActor(self.state_dim, self.action_dim).to(self.device)
        self.critic = FlatCritic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = FlatCritic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = ReplayMemory(50000)
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def train(self, total_episodes: int = 25, seed: int = 42, checkpoint_dir: Optional[Path] = None) -> Dict[str, float]:
        """Train the flat SAC baseline."""
        env = ICVEnvironment(self.cfg)
        set_seed(seed)
        episode_rewards: List[float] = []
        try:
            for episode in range(total_episodes):
                obs, _ = env.reset(seed=seed + episode)
                state = flatten_state(obs, self.cfg)
                done = False
                episode_reward = 0.0
                while not done:
                    action = self.act(obs, deterministic=False)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = flatten_state(next_obs, self.cfg)
                    done = terminated or truncated
                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    obs = next_obs
                    episode_reward += reward
                    self._update_if_ready()
                episode_rewards.append(float(episode_reward))
        finally:
            env.close()
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, checkpoint_dir / "model.pt")
            with open(checkpoint_dir / "training_curve.csv", "w", encoding="utf-8") as handle:
                handle.write("episode,reward\n")
                for idx, reward in enumerate(episode_rewards, start=1):
                    handle.write(f"{idx},{reward:.6f}\n")
        return {"episodes": total_episodes, "seed": seed}

    def _update_if_ready(self, batch_size: int = 64) -> None:
        """Run a flat SAC update."""
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states_t)
            q1_next, q2_next = self.target_critic(next_states_t, next_action)
            target = rewards_t + self.gamma * (1.0 - dones_t) * (torch.min(q1_next, q2_next) - self.alpha * next_log_prob)

        q1, q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action_sample, log_prob = self.actor.sample(states_t)
        q1_pi, q2_pi = self.critic(states_t, action_sample)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return a continuous action from the flat actor."""
        state = flatten_state(obs, self.cfg)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if self.action_dim > 4:
            action_np[4] = (action_np[4] + 1.0) / 2.0
        if self.action_dim > 5:
            action_np[5] = (action_np[5] + 1.0) / 2.0
        return action_np


class GraphSAGEDQN(VanillaDQN):
    """GraphSAGE plus DQN baseline approximating prior graph RL work."""

    name = "graphsage_dqn"

    def __init__(self, cfg: dict, device: Optional[str] = None):
        BaselinePolicy.__init__(self, cfg, device)
        self.encoder = HomogeneousGraphEncoder(cfg).to(self.device)
        self.q = MLPQNetwork(128).to(self.device)
        self.target_q = MLPQNetwork(128).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.q.parameters()), lr=1e-3)
        self.memory = ReplayMemory(50000)
        self.gamma = 0.99
        self.epsilon = 1.0

    def _encode(self, states: np.ndarray) -> torch.Tensor:
        """Encode raw observations into homogeneous graph embeddings."""
        state_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        return self.encoder(state_t)

    def _update_if_ready(self, batch_size: int = 64) -> None:
        """Run a GraphSAGE-DQN update."""
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        embed = self._encode(states)
        next_embed = self._encode(next_states)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        q_values = self.q(embed).gather(1, actions_t)
        with torch.no_grad():
            target = rewards_t + self.gamma * (1.0 - dones_t) * self.target_q(next_embed).max(dim=-1, keepdim=True).values
        loss = F.mse_loss(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return the greedy graph-based discrete action."""
        with torch.no_grad():
            embedding = self._encode(np.expand_dims(obs, axis=0))
            action_id = int(torch.argmax(self.q(embedding), dim=-1).item())
        return discrete_to_env_action(action_id, self.action_dim)


BASELINE_REGISTRY: Dict[str, Type[BaselinePolicy]] = {
    RandomPolicy.name: RandomPolicy,
    GreedyHeuristic.name: GreedyHeuristic,
    VanillaDQN.name: VanillaDQN,
    VanillaSACBaseline.name: VanillaSACBaseline,
    GraphSAGEDQN.name: GraphSAGEDQN,
}


def train_and_evaluate(model_name: str, cfg: dict, seed: int, episodes: int, output_dir: Path) -> Dict[str, object]:
    """Train a baseline and return a compact result dictionary."""
    model = BASELINE_REGISTRY[model_name](cfg)
    model_dir = output_dir / model_name / f"seed_{seed}"
    model.train(total_episodes=episodes, seed=seed, checkpoint_dir=model_dir)
    evaluation = model.evaluate(num_episodes=max(5, episodes // 2), seed=seed)
    episode_metrics = evaluation.episode_metrics
    summary = {
        "model": model_name,
        "seed": seed,
        "mean_reward": float(np.mean(evaluation.rewards)) if evaluation.rewards else 0.0,
        "std_reward": float(np.std(evaluation.rewards)) if evaluation.rewards else 0.0,
    }
    if episode_metrics:
        keys = sorted({key for metric in episode_metrics for key in metric})
        for key in keys:
            values = [metric[key] for metric in episode_metrics if key in metric]
            summary[key] = float(np.mean(values)) if values else 0.0
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    """Command-line entry point for baseline training."""
    parser = argparse.ArgumentParser(description="Train and evaluate VehicleFormer baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=sorted(BASELINE_REGISTRY), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    result = train_and_evaluate(args.model, cfg, args.seed, args.episodes, Path(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()