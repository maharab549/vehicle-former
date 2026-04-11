"""Utilities for publication-grade V2X KPI computation and aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from vehicleformer.env.network_sim import ChannelState, NetworkType


SINR_THRESHOLDS_DB = {
    NetworkType.G5: 3.0,
    NetworkType.V2X: 5.0,
    NetworkType.SAT: 0.0,
    NetworkType.G5V2X: 3.0,
}


def spectral_efficiency_from_sinr(sinr_db: float) -> float:
    """Return instantaneous spectral efficiency in bits/s/Hz."""
    return float(np.log2(1.0 + 10.0 ** (sinr_db / 10.0)))


def packet_delivery_ratio(channel: ChannelState) -> float:
    """Approximate packet delivery ratio from SINR and interference margin."""
    threshold = SINR_THRESHOLDS_DB.get(channel.network_type, 3.0)
    margin = channel.sinr_db - threshold
    logistic = 1.0 / (1.0 + np.exp(-0.9 * margin))
    interference_penalty = np.clip((channel.interference_dbm + 120.0) / 50.0, 0.6, 1.0)
    return float(np.clip(channel.reliability * logistic * interference_penalty, 0.0, 1.0))


@dataclass
class EpisodeKPIAccumulator:
    """Aggregates step-level KPIs into episode statistics."""

    latency_ms: List[float] = field(default_factory=list)
    pdr: List[float] = field(default_factory=list)
    throughput_mbps: List[float] = field(default_factory=list)
    spectral_efficiency: List[float] = field(default_factory=list)
    recovery_time_ms: List[float] = field(default_factory=list)
    handover_count: int = 0
    distance_m: float = 0.0

    def add(self, metrics: Dict[str, float], distance_delta_m: float) -> None:
        self.latency_ms.append(float(metrics["latency_ms"]))
        self.pdr.append(float(metrics["pdr"]))
        self.throughput_mbps.append(float(metrics["throughput_mbps"]))
        self.spectral_efficiency.append(float(metrics["spectral_efficiency"]))
        self.recovery_time_ms.append(float(metrics["recovery_time_ms"]))
        self.handover_count = int(metrics["handover_count"])
        self.distance_m += float(max(distance_delta_m, 0.0))

    def summary(self) -> Dict[str, float]:
        if not self.latency_ms:
            return {}
        latencies = np.asarray(self.latency_ms, dtype=np.float32)
        pdr = np.asarray(self.pdr, dtype=np.float32)
        throughput = np.asarray(self.throughput_mbps, dtype=np.float32)
        spectral_eff = np.asarray(self.spectral_efficiency, dtype=np.float32)
        recovery = np.asarray(self.recovery_time_ms, dtype=np.float32)
        km_travelled = max(self.distance_m / 1000.0, 1e-6)
        return {
            "mean_latency_ms": float(latencies.mean()),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "mean_pdr": float(pdr.mean()),
            "final_pdr": float(pdr[-1]),
            "mean_throughput_mbps": float(throughput.mean()),
            "mean_spectral_efficiency": float(spectral_eff.mean()),
            "mean_recovery_time_ms": float(recovery.mean()),
            "handover_count": int(self.handover_count),
            "handover_frequency_events_per_km": float(self.handover_count / km_travelled),
            "sla_50ms_met_pct": float((latencies <= 50.0).mean() * 100.0),
            "sla_30ms_met_pct": float((latencies <= 30.0).mean() * 100.0),
            "mean_reliability": float(pdr.mean()),
        }


class V2XKPITracker:
    """Stateful per-episode KPI tracker for the environment."""

    def __init__(self, cfg: dict):
        self.step_length_s = float(cfg["simulation"].get("step_length", 0.1))
        self.reset()

    def reset(self) -> None:
        self.previous_network: Optional[NetworkType] = None
        self.previous_position: Optional[np.ndarray] = None
        self.outage_start_step: Optional[int] = None
        self.handover_count: int = 0
        self.step_count: int = 0
        self.episode = EpisodeKPIAccumulator()

    def _distance_delta(self, position: Optional[np.ndarray]) -> float:
        if position is None:
            return 0.0
        if self.previous_position is None:
            self.previous_position = position.copy()
            return 0.0
        delta = float(np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        return delta

    def step(
        self,
        channel: Optional[ChannelState],
        selected_network: NetworkType,
        handover: bool,
        ego_position: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Update internal state and return publication KPIs for the current step."""
        self.step_count += 1
        distance_delta_m = self._distance_delta(ego_position)
        if handover:
            self.handover_count += 1
        effective_channel = channel
        if effective_channel is None:
            metrics = {
                "latency_ms": 999.0,
                "pdr": 0.0,
                "handover_count": int(self.handover_count),
                "throughput_mbps": 0.0,
                "recovery_time_ms": 0.0,
                "spectral_efficiency": 0.0,
                "network_selected": int(min(int(selected_network), 2)),
            }
            self.episode.add(metrics, distance_delta_m)
            return metrics

        if effective_channel.available:
            if self.outage_start_step is not None:
                outage_steps = self.step_count - self.outage_start_step
                recovery_time_ms = outage_steps * self.step_length_s * 1000.0
                self.outage_start_step = None
            else:
                recovery_time_ms = 0.0
        else:
            if self.outage_start_step is None:
                self.outage_start_step = self.step_count
            recovery_time_ms = 0.0

        latency_ms = (
            effective_channel.propagation_delay_ms
            + effective_channel.processing_delay_ms
            + effective_channel.queuing_delay_ms
            + effective_channel.mac_contention_delay_ms
        )
        spectral_efficiency = spectral_efficiency_from_sinr(effective_channel.sinr_db)
        metrics = {
            "latency_ms": float(latency_ms),
            "pdr": packet_delivery_ratio(effective_channel),
            "handover_count": int(self.handover_count),
            "throughput_mbps": float(effective_channel.throughput_mbps if effective_channel.available else 0.0),
            "recovery_time_ms": float(recovery_time_ms),
            "spectral_efficiency": float(spectral_efficiency if effective_channel.available else 0.0),
            "network_selected": int(min(int(selected_network), 2)),
        }
        self.previous_network = selected_network
        self.episode.add(metrics, distance_delta_m)
        return metrics

    def episode_summary(self) -> Dict[str, float]:
        """Return episode-level aggregate KPIs."""
        return self.episode.summary()