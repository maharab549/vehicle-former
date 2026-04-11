"""
Main VehicleFormer Gymnasium Environment.

Combines SUMO traffic simulation with multi-network channel physics
to create the observation/action/reward interface for the SAC agent.

Observation: heterogeneous graph (vehicles, RSUs, base stations)
Action:      [network_select, tx_power, offload_ratio]
Reward:      multi-objective (latency, reliability, energy, coverage)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from vehicleformer.env.sumo_env import SUMOBridge, VehicleState
from vehicleformer.env.network_sim import NetworkSimulator, NetworkType, ChannelState
from vehicleformer.env.kpi_metrics import V2XKPITracker
from vehicleformer.training.novelty import DomainRandomizer


class ICVEnvironment(gym.Env):
    """
    Intelligent Connected Vehicle Gymnasium Environment.
    
    This is the core training environment for VehicleFormer.
    It provides a graph-structured observation of the vehicle-road-cloud
    system and a multi-objective reward aligned with the ICV roadmap KPIs.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: dict, vehicle_id: str = "veh_0"):
        super().__init__()
        self.cfg = cfg
        self.vehicle_id = vehicle_id    # ego vehicle to control

        # Sub-components
        self.sumo = SUMOBridge(cfg, use_gui=cfg['simulation']['use_gui'])
        self.net_sim = NetworkSimulator(cfg)

        # Config shortcuts
        sim_cfg = cfg['simulation']
        obs_cfg = cfg['observation']
        act_cfg = cfg['action']
        self.max_steps = sim_cfg['max_episode_steps']
        self.max_vehicles = sim_cfg['num_vehicles']
        self.history_len = obs_cfg['history_len']

        # ─── Observation Space ────────────────────────────────────────
        # Graph node features (flattened for now, HetGNN processes them)
        # Shape: [max_vehicles * vehicle_dim + max_rsus * rsu_dim + max_bs * bs_dim + network_obs]
        self.vehicle_dim = obs_cfg['vehicle_dim']
        self.rsu_dim     = obs_cfg['rsu_dim']
        self.bs_dim      = obs_cfg['bs_dim']
        self.max_rsus    = obs_cfg['max_rsus']
        self.max_bs      = obs_cfg['max_base_stations']
        self.network_obs_dim = 3 * 6 + 4  # 3 networks × 6 features + 4 onehot

        # Total flat obs dim
        self.obs_dim = (
            self.max_vehicles * self.vehicle_dim
            + self.max_rsus * self.rsu_dim
            + self.max_bs * self.bs_dim
            + self.network_obs_dim
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # ─── Action Space ─────────────────────────────────────────────
        # [network_0, network_1, network_2, network_3, tx_power, offload_ratio]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ─── History buffer (for world model) ────────────────────────
        self.obs_history = deque(maxlen=self.history_len)

        # ─── Episode tracking ─────────────────────────────────────────
        self._step_count = 0
        self._current_network = NetworkType.G5
        self._episode_latencies = []
        self._episode_reliabilities = []
        self._last_obs = None
        self._kpi_tracker = V2XKPITracker(cfg)
        self._domain_randomizer = DomainRandomizer(cfg, seed=cfg['project']['seed'])
        self._domain_profile = {}

    def set_curriculum_level(self, level: float) -> None:
        """Externally control domain randomization curriculum difficulty."""
        self._domain_randomizer.set_difficulty(level)

    # ─── Core Gym API ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.sumo.reset()
        self.net_sim.step_count = 0
        profile = self._domain_randomizer.sample()
        self._domain_profile = profile.as_dict()
        self.net_sim.apply_domain_profile(self._domain_profile)
        self._step_count = 0
        self._current_network = NetworkType.G5
        self._episode_latencies = []
        self._episode_reliabilities = []
        self._kpi_tracker.reset()

        # Clear history
        self.obs_history.clear()

        # Warm up SUMO (spawn vehicles)
        for _ in range(20):
            self.sumo.step()
            self.net_sim.step()

        obs = self._get_observation()
        self._last_obs = obs

        # Fill history with initial obs
        for _ in range(self.history_len):
            self.obs_history.append(obs.copy())

        info = {
            "episode_step": 0,
            "network": self._current_network.name,
            "network_selected": 0,
            "domain_profile": self._domain_profile,
            "curriculum_difficulty": self._domain_randomizer.difficulty,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.
        
        Args:
            action: [net_logits(4), tx_power_norm, offload_ratio_norm]
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        # ─── Parse action ─────────────────────────────────────────────
        net_logits   = action[:4]
        tx_power_norm = float(np.clip(action[4], 0, 1))
        offload_ratio = float(np.clip(action[5], 0, 1))

        selected_net = NetworkType(int(np.argmax(net_logits)))
        tx_power_w   = (
            self.cfg['action']['tx_power_min']
            + tx_power_norm * (self.cfg['action']['tx_power_max']
                               - self.cfg['action']['tx_power_min'])
        )

        # Detect handover
        handover = (selected_net != self._current_network)
        self._current_network = selected_net

        # ─── Advance simulation ───────────────────────────────────────
        self.sumo.step()
        self._step_count += 1

        # Feed vehicle positions into network sim for congestion-aware loads
        vehicle_states = self.sumo.get_vehicle_states()
        if vehicle_states:
            vpos = np.array([vs.position for vs in vehicle_states])
            self.net_sim.set_vehicle_positions(vpos)
        self.net_sim.step()

        # ─── Get ego vehicle state ────────────────────────────────────
        ego = self._get_ego_vehicle(vehicle_states)

        # ─── Get channel state ────────────────────────────────────────
        if ego is not None:
            channel_states = self.net_sim.get_channel_states(
                ego.position, ego.speed
            )
            chosen_channel = channel_states[selected_net]
        else:
            chosen_channel = None

        # ─── Compute reward ───────────────────────────────────────────
        reward, reward_info = self._compute_reward(
            chosen_channel, handover, tx_power_w, offload_ratio
        )

        # ─── Track episode metrics ────────────────────────────────────
        if chosen_channel and chosen_channel.available:
            self._episode_latencies.append(chosen_channel.latency_ms)
            self._episode_reliabilities.append(chosen_channel.reliability)

        # ─── Get observation ──────────────────────────────────────────
        obs = self._get_observation()
        self.obs_history.append(obs.copy())
        self._last_obs = obs

        # ─── Termination ─────────────────────────────────────────────
        terminated = self.sumo.is_done()
        truncated  = self._step_count >= self.max_steps

        kpi_info = self._kpi_tracker.step(
            channel=chosen_channel,
            selected_network=selected_net,
            handover=handover,
            ego_position=ego.position if ego is not None else None,
        )

        info = {
            "episode_step"     : self._step_count,
            "network"          : selected_net.name,
            "handover"         : handover,
            "latency_ms"       : kpi_info["latency_ms"],
            "pdr"              : kpi_info["pdr"],
            "reliability"      : kpi_info["pdr"],
            "throughput_mbps"  : kpi_info["throughput_mbps"],
            "handover_count"   : kpi_info["handover_count"],
            "recovery_time_ms" : kpi_info["recovery_time_ms"],
            "spectral_efficiency": kpi_info["spectral_efficiency"],
            "network_selected" : kpi_info["network_selected"],
            "domain_profile"   : self._domain_profile,
            "curriculum_difficulty": self._domain_randomizer.difficulty,
            "tx_power_w"       : tx_power_w,
            "offload_ratio"    : offload_ratio,
            **reward_info,
        }

        if terminated or truncated:
            info["episode_metrics"] = self._compute_episode_metrics()

        return obs, reward, terminated, truncated, info

    def close(self):
        self.sumo.close()

    # ─── Observation Builder ─────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector from all components."""
        vehicle_states = self.sumo.get_vehicle_states()
        infra = self.net_sim.get_infrastructure_state()
        bs_positions = np.array(self.cfg['simulation']['base_stations_5g'])
        rsu_positions = np.array(self.cfg['simulation']['rsu_positions'])

        ego = self._get_ego_vehicle(vehicle_states)
        if ego is not None:
            channel_states = self.net_sim.get_channel_states(
                ego.position, ego.speed
            )
        else:
            channel_states = None

        # ─── Vehicle node features ────────────────────────────────────
        veh_obs = np.zeros((self.max_vehicles, self.vehicle_dim), dtype=np.float32)
        for i, vs in enumerate(vehicle_states[:self.max_vehicles]):
            cs = self.net_sim.get_channel_states(vs.position, vs.speed)
            net_oh = np.zeros(3)
            net_oh[int(self._current_network) % 3] = 1.0
            veh_obs[i] = np.array([
                vs.position[0] / 800.0,             # x normalized
                vs.position[1] / 800.0,             # y normalized
                vs.speed / 14.0,                    # speed (max ~50 km/h)
                vs.heading / 360.0,                 # heading
                (cs[NetworkType.G5].rssi_dbm + 120) / 90,
                (cs[NetworkType.V2X].rssi_dbm + 120) / 90,
                np.clip(cs[NetworkType.SAT].latency_ms / 200, 0, 1),
                infra.bs_loads.mean(),
                np.random.uniform(0.1, 1.0),        # task_size (placeholder)
                np.random.uniform(0.3, 1.0),        # deadline_remaining
                1.0,                                 # battery_level (EV)
                *net_oh,
            ], dtype=np.float32)

        # ─── RSU node features ────────────────────────────────────────
        rsu_obs = np.zeros((self.max_rsus, self.rsu_dim), dtype=np.float32)
        for i, rp in enumerate(rsu_positions[:self.max_rsus]):
            rsu_obs[i] = np.array([
                rp[0] / 800.0,
                rp[1] / 800.0,
                float(len(vehicle_states)) / self.max_vehicles,
                infra.rsu_loads[i] if i < len(infra.rsu_loads) else 0.5,
                float(self.net_sim.rsu_online[i]) if i < len(self.net_sim.rsu_online) else 0.0,
            ], dtype=np.float32)

        # ─── Base station node features ───────────────────────────────
        bs_obs = np.zeros((self.max_bs, self.bs_dim), dtype=np.float32)
        for i, bp in enumerate(bs_positions[:self.max_bs]):
            bs_obs[i] = np.array([
                bp[0] / 800.0,
                bp[1] / 800.0,
                3.5 / 6.0,          # frequency 3.5 GHz normalized
                infra.bs_loads[i] if i < len(infra.bs_loads) else 0.5,
                500.0 / 1000.0,     # coverage_radius normalized
                0.2,                # latency_mean placeholder
            ], dtype=np.float32)

        # ─── Network channel observation ──────────────────────────────
        if channel_states is not None:
            net_obs = self.net_sim.to_observation_vector(
                channel_states, self._current_network
            )
        else:
            net_obs = np.zeros(self.network_obs_dim, dtype=np.float32)

        # ─── Concatenate all ──────────────────────────────────────────
        obs = np.concatenate([
            veh_obs.flatten(),
            rsu_obs.flatten(),
            bs_obs.flatten(),
            net_obs,
        ])

        # Verify shape
        assert obs.shape[0] == self.obs_dim, \
            f"Obs shape mismatch: {obs.shape[0]} != {self.obs_dim}"

        return obs.astype(np.float32)

    # ─── Reward Function ─────────────────────────────────────────────────

    def _compute_reward(
        self,
        channel: Optional[ChannelState],
        handover: bool,
        tx_power_w: float,
        offload_ratio: float,
    ) -> Tuple[float, dict]:
        """
        Multi-objective reward aligned with ICV Roadmap KPIs.
        
        R = w_lat * r_lat + w_rel * r_rel + w_eng * r_eng + w_cov * r_cov
        
        KPI targets (from China ICV Roadmap 2025-2030):
        - Latency: 50ms@99% for 5G, 30ms average for V2X
        - Reliability: 99%
        - Coverage: >98% urban
        """
        cfg_r = self.cfg['reward']
        reward_mode = cfg_r.get('mode', 'legacy')

        if channel is None or not channel.available:
            disconnect_penalty = cfg_r.get('disconnect_penalty', -10.0)
            if reward_mode == 'robust_paper':
                return float(disconnect_penalty), {
                    "r_lat": -1.0,
                    "r_rel": -1.0,
                    "r_eng": 0.0,
                    "r_cov": -1.0,
                    "r_sla30": -1.0,
                    "r_thr": -1.0,
                    "r_stab": -1.0,
                }
            return float(disconnect_penalty), {"r_lat": -1, "r_rel": -1, "r_eng": 0, "r_cov": -1}

        if reward_mode == 'robust_paper':
            lat_target = cfg_r['latency_target_ms']
            sla30_target = cfg_r.get('v2x_latency_target_ms', 30.0)
            rel_target = cfg_r['reliability_target']
            max_power = self.cfg['action']['tx_power_max']

            lat_ratio = channel.latency_ms / max(lat_target, 1e-6)
            r_lat = float(np.clip(2.0 * np.exp(-1.8 * lat_ratio) - 1.0, -1.0, 1.0))

            if channel.latency_ms <= sla30_target:
                r_sla30 = 1.0
            else:
                r_sla30 = float(np.clip(1.0 - (channel.latency_ms - sla30_target) / 25.0, -1.0, 1.0))

            r_rel = float(np.clip((channel.reliability - (1 - rel_target)) / rel_target, -1.0, 1.0))
            r_eng = float(1.0 - tx_power_w / max_power)
            r_thr = float(np.clip(np.log1p(channel.throughput_mbps) / np.log1p(cfg_r.get('throughput_norm_mbps', 200.0)), 0.0, 1.0))

            if len(self._episode_latencies) >= 5:
                recent = np.array(self._episode_latencies[-5:])
                r_stab = float(np.clip(1.0 - recent.std() / cfg_r.get('stability_scale_ms', 12.0), -1.0, 1.0))
            else:
                r_stab = 0.0

            if handover:
                previous_latency = self._episode_latencies[-1] if self._episode_latencies else channel.latency_ms
                improved = previous_latency - channel.latency_ms
                handover_penalty = cfg_r.get('handover_penalty_bad', -0.9)
                if improved >= cfg_r.get('handover_improvement_ms', 5.0):
                    handover_penalty = cfg_r.get('handover_penalty_good', -0.2)
            else:
                handover_penalty = 0.0

            reward = (
                cfg_r.get('w_latency', 0.30) * r_lat
                + cfg_r.get('w_sla30', 0.25) * r_sla30
                + cfg_r.get('w_reliability', 0.15) * r_rel
                + cfg_r.get('w_throughput', 0.15) * r_thr
                + cfg_r.get('w_stability', 0.10) * r_stab
                + cfg_r.get('w_energy', 0.05) * r_eng
                + handover_penalty
            )

            return float(reward), {
                "r_lat": r_lat,
                "r_rel": r_rel,
                "r_eng": r_eng,
                "r_cov": 1.0,
                "r_sla30": r_sla30,
                "r_thr": r_thr,
                "r_stab": r_stab,
            }

        # Legacy reward for backwards-compatible experiments
        lat_target = cfg_r['latency_target_ms']
        r_lat = np.clip(1.0 - channel.latency_ms / (lat_target * 2), -1, 1)

        rel_target = cfg_r['reliability_target']
        r_rel = (channel.reliability - (1 - rel_target)) / rel_target
        r_rel = float(np.clip(r_rel, -1, 1))

        max_power = self.cfg['action']['tx_power_max']
        r_eng = 1.0 - tx_power_w / max_power

        r_cov = float(channel.available)
        handover_penalty = -0.5 if handover else 0.0

        reward = (
            cfg_r['w_latency']     * r_lat
            + cfg_r['w_reliability'] * r_rel
            + cfg_r['w_energy']      * r_eng
            + cfg_r['w_coverage']    * r_cov
            + handover_penalty
        )

        return float(reward), {
            "r_lat": r_lat, "r_rel": r_rel,
            "r_eng": r_eng, "r_cov": r_cov,
        }

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _get_ego_vehicle(
        self, vehicle_states: List[VehicleState]
    ) -> Optional[VehicleState]:
        for vs in vehicle_states:
            if vs.vehicle_id == self.vehicle_id:
                return vs
        return vehicle_states[0] if vehicle_states else None

    def _compute_episode_metrics(self) -> dict:
        """Compute final episode KPIs for evaluation."""
        return self._kpi_tracker.episode_summary()

    def get_obs_history(self) -> np.ndarray:
        """Return observation history for world model input. Shape: (T, obs_dim)"""
        hist = list(self.obs_history)
        if len(hist) < self.history_len:
            pad = [hist[0]] * (self.history_len - len(hist))
            hist = pad + hist
        return np.stack(hist, axis=0)  # (T, obs_dim)
