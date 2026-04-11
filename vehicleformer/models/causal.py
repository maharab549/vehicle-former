"""Counterfactual analysis utilities for the causal VehicleFormer pipeline."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


class CounterfactualEvaluator:
    """Simulate short-horizon counterfactual decisions with the world model."""

    def __init__(self, world_model, horizon: int = 5):
        self.world_model = world_model
        self.horizon = horizon

    @staticmethod
    def _network_to_action(template_action: np.ndarray, network_id: int) -> np.ndarray:
        """Replace the network logits in an action while preserving continuous controls."""
        action = template_action.copy()
        action[:4] = 0.0
        action[network_id] = 1.0
        return action

    @staticmethod
    def _reward_proxy(embedding: torch.Tensor, action: np.ndarray) -> float:
        """Map predicted embeddings to a scalar reward surrogate for comparison."""
        network_bonus = np.array([0.25, 0.20, -0.05, 0.15], dtype=np.float32)
        stability_term = float(torch.tanh(embedding.mean()).item())
        usable = min(len(network_bonus), action.shape[0])
        return stability_term + float(np.dot(action[:usable], network_bonus[:usable]))

    def evaluate(self, trajectory: Sequence[Dict[str, np.ndarray]]) -> Dict[int, Dict[str, object]]:
        """Return a counterfactual reward delta for each decision point in a trajectory."""
        results: Dict[int, Dict[str, object]] = {}
        for step_idx, transition in enumerate(trajectory):
            embedding = torch.tensor(transition["embedding"], dtype=torch.float32).view(1, 1, -1)
            factual_action = np.asarray(transition["action"], dtype=np.float32)
            factual_reward = self._rollout_proxy(embedding, factual_action)
            alternatives = {}
            factual_network = int(np.argmax(factual_action[:4]))
            for network_id in range(min(3, factual_action.shape[0])):
                if network_id == factual_network:
                    continue
                cf_action = self._network_to_action(factual_action, network_id)
                cf_reward = self._rollout_proxy(embedding, cf_action)
                alternatives[network_id] = {
                    "counterfactual_reward": cf_reward,
                    "reward_delta": cf_reward - factual_reward,
                }
            results[step_idx] = {
                "factual_network": factual_network,
                "factual_reward_proxy": factual_reward,
                "counterfactuals": alternatives,
            }
        return results

    def _rollout_proxy(self, embedding: torch.Tensor, action: np.ndarray) -> float:
        """Roll the world model forward and aggregate reward proxies."""
        total = 0.0
        seq = embedding
        with torch.no_grad():
            for _ in range(self.horizon):
                out = self.world_model(seq)
                predicted = out["mean_pred"][:, :1, :]
                total += self._reward_proxy(predicted[:, 0, :], action)
                seq = torch.cat([seq[:, 1:, :], predicted], dim=1) if seq.shape[1] > 1 else predicted
        return float(total)