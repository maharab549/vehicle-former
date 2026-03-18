"""
Multi-Network Channel Simulator for VehicleFormer
Simulates realistic 5G, C-V2X, and Satellite channel physics.

Based on:
- 3GPP TR 37.885 (C-V2X channel model)
- 3GPP TR 38.901 (5G channel model)
- ITU-R S.1503 (satellite link budget)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


class NetworkType(IntEnum):
    G5    = 0   # 5G cellular
    V2X   = 1   # C-V2X direct connection
    SAT   = 2   # Satellite
    G5V2X = 3   # 5G + V2X combined


@dataclass
class ChannelState:
    """Represents the current channel state for one vehicle-network link."""
    network_type: NetworkType
    rssi_dbm: float          # Received Signal Strength Indicator
    sinr_db: float           # Signal-to-Interference-plus-Noise Ratio
    latency_ms: float        # End-to-end latency
    throughput_mbps: float   # Available throughput
    reliability: float       # Packet delivery ratio [0,1]
    handover_cost_ms: float  # Cost to switch to this network
    available: bool          # Whether this network is reachable


@dataclass
class InfrastructureState:
    """State of all network infrastructure."""
    bs_loads: np.ndarray        # 5G base station loads [0,1]
    rsu_loads: np.ndarray       # C-V2X RSU loads [0,1]
    sat_coverage: bool          # Satellite availability
    sat_elevation_deg: float    # Satellite elevation angle


class NetworkSimulator:
    """
    Physics-based multi-network channel simulator.
    Implements realistic channel models for 5G, C-V2X, and Satellite.
    
    Used as a drop-in for NS3 during development.
    Later: replace with actual NS3 via ns3-gym bridge.
    """

    def __init__(self, cfg: dict, rng_seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)

        # Infrastructure positions from config
        self.bs_positions = np.array(cfg['simulation']['base_stations_5g'])  # (N, 2)
        self.rsu_positions = np.array(cfg['simulation']['rsu_positions'])    # (M, 2)
        self.n_rsu = len(self.rsu_positions)
        self.n_bs  = len(self.bs_positions)

        # 5G parameters (3GPP TR 38.901 Urban Macro)
        self.f_5g_ghz     = 3.5          # Carrier frequency (GHz)
        self.p_tx_5g_dbm  = 46.0         # BS transmit power (dBm)
        self.g_tx_5g_db   = 8.0          # BS antenna gain (dBi)
        self.noise_fig_db = 7.0          # UE noise figure (dB)
        self.bw_5g_mhz    = 100.0        # Bandwidth (MHz)

        # C-V2X parameters (3GPP TR 37.885)
        self.f_v2x_ghz    = 5.9          # ITS band (GHz)
        self.p_tx_v2x_dbm = 23.0         # OBU/RSU transmit power (dBm)
        self.v2x_range_m  = 200.0        # Reduced from 300m — realistic urban V2X

        # Satellite parameters (LEO, ~550km orbit, similar to Starlink)
        self.sat_altitude_km = 550.0
        self.sat_freq_ghz    = 12.0      # Ku band
        self.sat_eirp_dbw    = 34.0      # Effective isotropic radiated power
        self.sat_latency_base_ms = 20.0  # Minimum propagation delay

        # State tracking
        self.time_step = 0
        self.bs_loads = self.rng.uniform(0.3, 0.7, self.n_bs)
        self.rsu_loads = self.rng.uniform(0.1, 0.4, self.n_rsu)
        self.sat_elevation = 45.0        # degrees

        # ─── RSU outage model ─────────────────────────────────────────
        # Each RSU can go offline (failure/maintenance).
        # State: True = online, False = offline
        self.rsu_online = np.ones(self.n_rsu, dtype=bool)
        self._rsu_outage_countdown = np.zeros(self.n_rsu, dtype=int)  # steps until back online
        self._rsu_outage_prob = 0.002    # per-step probability of new outage
        self._rsu_outage_duration = (50, 200)  # duration range in steps

        # ─── Congestion burst model ───────────────────────────────────
        # Periodic load spikes simulate rush-hour or event traffic
        self._congestion_active = False
        self._congestion_countdown = 0
        self._congestion_burst_prob = 0.003  # per-step probability
        self._congestion_duration = (100, 400)
        self._congestion_magnitude = 0.0  # extra load during burst

        # Vehicle positions cache (updated externally for congestion-aware loads)
        self._vehicle_positions: Optional[np.ndarray] = None

    def set_vehicle_positions(self, positions: np.ndarray) -> None:
        """Update vehicle positions for congestion-aware RSU/BS loads.
        positions: (N, 2) array of vehicle [x, y] coords."""
        self._vehicle_positions = positions

    def step(self) -> None:
        """Advance simulator time step - update loads, outages, congestion."""
        self.time_step += 1

        # ─── RSU outage dynamics ──────────────────────────────────────
        for i in range(self.n_rsu):
            if not self.rsu_online[i]:
                self._rsu_outage_countdown[i] -= 1
                if self._rsu_outage_countdown[i] <= 0:
                    self.rsu_online[i] = True   # back online
            else:
                if self.rng.random() < self._rsu_outage_prob:
                    self.rsu_online[i] = False
                    dur_lo, dur_hi = self._rsu_outage_duration
                    self._rsu_outage_countdown[i] = self.rng.integers(dur_lo, dur_hi)

        # ─── Congestion burst dynamics ────────────────────────────────
        if self._congestion_active:
            self._congestion_countdown -= 1
            if self._congestion_countdown <= 0:
                self._congestion_active = False
                self._congestion_magnitude = 0.0
        else:
            if self.rng.random() < self._congestion_burst_prob:
                self._congestion_active = True
                dur_lo, dur_hi = self._congestion_duration
                self._congestion_countdown = self.rng.integers(dur_lo, dur_hi)
                mag_lo, mag_hi = getattr(self, '_stress_congestion_mag_range', (0.15, 0.35))
                self._congestion_magnitude = self.rng.uniform(mag_lo, mag_hi)

        # ─── Base load evolution (AR(1) process) ─────────────────────
        noise = self.rng.normal(0, 0.03, self.bs_loads.shape)
        self.bs_loads = np.clip(
            self.bs_loads + noise + self._congestion_magnitude * 0.5,
            0.05, 0.98
        )
        # Mean-revert BS loads so they don't drift permanently high
        self.bs_loads += 0.01 * (0.5 - self.bs_loads)

        noise = self.rng.normal(0, 0.025, self.rsu_loads.shape)
        self.rsu_loads = np.clip(
            self.rsu_loads + noise + self._congestion_magnitude,
            0.01, 0.95
        )
        self.rsu_loads += 0.01 * (0.35 - self.rsu_loads)

        # ─── Proximity-based load: nearby vehicles drive up RSU/BS loads ──
        if self._vehicle_positions is not None and len(self._vehicle_positions) > 0:
            for i, rp in enumerate(self.rsu_positions):
                dists = np.linalg.norm(self._vehicle_positions - rp, axis=1)
                n_nearby = int((dists <= self.v2x_range_m).sum())
                # Each nearby vehicle adds ~0.03 load
                self.rsu_loads[i] = np.clip(
                    self.rsu_loads[i] + n_nearby * 0.03, 0.01, 0.95
                )
            for i, bp in enumerate(self.bs_positions):
                dists = np.linalg.norm(self._vehicle_positions - bp, axis=1)
                n_nearby = int((dists <= 400).sum())
                self.bs_loads[i] = np.clip(
                    self.bs_loads[i] + n_nearby * 0.015, 0.05, 0.98
                )

        # Satellite oscillates over time
        self.sat_elevation = 30 + 40 * np.sin(self.time_step * 0.01)

    def get_channel_states(
        self,
        vehicle_position: np.ndarray,  # (2,) [x, y] in meters
        vehicle_speed: float,           # m/s
    ) -> Dict[NetworkType, ChannelState]:
        """
        Compute channel state for all available networks.
        
        Args:
            vehicle_position: Vehicle [x, y] coordinates
            vehicle_speed: Vehicle speed in m/s (affects Doppler)
            
        Returns:
            Dict mapping NetworkType to ChannelState
        """
        states = {}

        # ─── 5G Channel ────────────────────────────────────────────────
        states[NetworkType.G5] = self._compute_5g_channel(
            vehicle_position, vehicle_speed
        )

        # ─── C-V2X Channel ─────────────────────────────────────────────
        states[NetworkType.V2X] = self._compute_v2x_channel(
            vehicle_position, vehicle_speed
        )

        # ─── Satellite Channel ──────────────────────────────────────────
        states[NetworkType.SAT] = self._compute_satellite_channel(
            vehicle_position
        )

        # ─── 5G + V2X Combined ─────────────────────────────────────────
        g5  = states[NetworkType.G5]
        v2x = states[NetworkType.V2X]
        states[NetworkType.G5V2X] = ChannelState(
            network_type    = NetworkType.G5V2X,
            rssi_dbm        = max(g5.rssi_dbm, v2x.rssi_dbm),
            sinr_db         = max(g5.sinr_db, v2x.sinr_db),
            latency_ms      = min(g5.latency_ms, v2x.latency_ms),
            throughput_mbps = g5.throughput_mbps + v2x.throughput_mbps * 0.5,
            reliability     = 1 - (1 - g5.reliability) * (1 - v2x.reliability),
            handover_cost_ms= max(g5.handover_cost_ms, v2x.handover_cost_ms),
            available       = g5.available or v2x.available,
        )

        return states

    # ─── 5G Channel Model (3GPP UMa) ────────────────────────────────────
    def _compute_5g_channel(
        self, pos: np.ndarray, speed: float
    ) -> ChannelState:
        # Find nearest base station
        distances = np.linalg.norm(self.bs_positions - pos, axis=1)
        nearest_idx = np.argmin(distances)
        d = max(distances[nearest_idx], 10.0)  # min 10m
        load = self.bs_loads[nearest_idx]

        # Path loss: 3GPP UMa model (dB)
        pl_db = (
            28.0 + 22 * np.log10(d) + 20 * np.log10(self.f_5g_ghz)
            + self.rng.normal(0, 4.0)  # shadow fading σ=4dB
        )

        # RSSI
        rssi = self.p_tx_5g_dbm + self.g_tx_5g_db - pl_db
        rssi = float(np.clip(rssi, -120, -30))

        # Interference from other BSs (simplified)
        interference_db = -90 + load * 10
        noise_dbm = -174 + 10 * np.log10(self.bw_5g_mhz * 1e6) + self.noise_fig_db
        sinr = rssi - 10 * np.log10(10**(interference_db/10) + 10**(noise_dbm/10))
        sinr = float(np.clip(sinr, -10, 30))

        # Shannon throughput (Mbps)
        throughput = self.bw_5g_mhz * np.log2(1 + 10**(sinr/10)) * (1 - load)
        throughput = float(max(throughput, 0.1))

        # Latency model (base + load + distance propagation)
        base_latency = 5.0 + load * 20 + d / 3e5 * 1000
        doppler_penalty = speed * 0.5  # higher speed → more handover risk
        latency = float(max(base_latency + doppler_penalty + self.rng.exponential(2), 3.0))

        # Reliability (Friis + load)
        reliability = float(np.clip(0.999 - load * 0.05 - (d / 5000) * 0.1, 0.5, 0.999))

        available = rssi > -110

        return ChannelState(
            network_type    = NetworkType.G5,
            rssi_dbm        = rssi,
            sinr_db         = sinr,
            latency_ms      = latency,
            throughput_mbps = throughput,
            reliability     = reliability,
            handover_cost_ms= 5.0 + speed * 0.3,
            available       = available,
        )

    # ─── C-V2X Channel Model (3GPP TR 37.885) ───────────────────────────
    def _compute_v2x_channel(
        self, pos: np.ndarray, speed: float
    ) -> ChannelState:
        # Find nearest RSU
        distances = np.linalg.norm(self.rsu_positions - pos, axis=1)
        nearest_idx = np.argmin(distances)
        d = max(distances[nearest_idx], 5.0)
        load = self.rsu_loads[nearest_idx]

        # Check RSU outage
        rsu_is_online = bool(self.rsu_online[nearest_idx])
        in_range = d <= self.v2x_range_m and rsu_is_online

        if not in_range:
            return ChannelState(
                NetworkType.V2X, -120, -30, 999.0, 0.0, 0.0, 2.0, False
            )

        # Winner+ B1 path loss model for V2X
        pl_db = (
            40 * np.log10(d) + 9.45
            - 17.3 * np.log10(1.5)   # tx height 1.5m
            - 17.3 * np.log10(1.5)   # rx height 1.5m
            + 2.7 * np.log10(self.f_v2x_ghz)
            + self.rng.normal(0, 4.0)  # increased shadow fading σ=4dB
        )

        rssi = self.p_tx_v2x_dbm - pl_db
        rssi = float(np.clip(rssi, -100, -30))

        # Interference scales with load (congestion makes SINR worse)
        sinr = rssi + 90 - load * 15  # was load*5 — tripled
        sinr = float(np.clip(sinr, -5, 25))

        # V2X latency: base + load penalty + distance + speed Doppler + jitter
        # Under high load, V2X can become slower than 5G!
        load_penalty = load * 30       # was load*10 — tripled
        distance_penalty = d * 0.04    # was d*0.01 — 4x
        speed_penalty = speed * 0.3    # Doppler effect on V2X sidelink
        jitter = self.rng.exponential(2 + load * 5)  # load-dependent jitter
        latency = float(max(
            3.0 + load_penalty + distance_penalty + speed_penalty + jitter,
            2.0
        ))

        throughput = 27 * np.log2(1 + 10**(sinr/10)) * (1 - load) / 100
        throughput = float(max(throughput, 0.1))

        reliability = float(np.clip(
            0.999 - load * 0.08 - (d / self.v2x_range_m) * 0.1 - speed * 0.002,
            0.7, 0.999
        ))

        return ChannelState(
            network_type    = NetworkType.V2X,
            rssi_dbm        = rssi,
            sinr_db         = sinr,
            latency_ms      = latency,
            throughput_mbps = throughput,
            reliability     = reliability,
            handover_cost_ms= 2.0 + speed * 0.1,  # speed-dependent
            available       = True,
        )

    # ─── Satellite Channel Model (LEO, ~Starlink) ────────────────────────
    def _compute_satellite_channel(self, pos: np.ndarray) -> ChannelState:
        elev = self.sat_elevation
        available = elev > 10.0  # minimum elevation angle

        if not available:
            return ChannelState(
                NetworkType.SAT, -120, -20, 999.0, 0.0, 0.0, 50.0, False
            )

        # Free-space path loss at satellite distance
        slant_range_km = self.sat_altitude_km / np.sin(np.radians(elev))
        fspl_db = (
            20 * np.log10(slant_range_km * 1000)
            + 20 * np.log10(self.sat_freq_ghz * 1e9)
            + 20 * np.log10(4 * np.pi / 3e8)
        )

        # Link budget
        rx_power_dbw = self.sat_eirp_dbw - fspl_db + 30  # 30dB antenna gain
        noise_dbw = -228.6 + 10 * np.log10(290) + 10 * np.log10(500e6)  # 500MHz BW
        sinr = float(np.clip(rx_power_dbw - noise_dbw, -5, 20))

        # Throughput ~100 Kbps for vehicle satellite (per roadmap)
        throughput = float(max(0.1 + sinr * 0.005, 0.05))

        # Latency: propagation (2-way) + processing
        prop_latency = 2 * slant_range_km / 300  # ms (light speed ~300 km/ms)
        latency = float(max(self.sat_latency_base_ms + prop_latency, 18.0))

        rssi = float(np.clip(rx_power_dbw + 30 - 50, -90, -50))
        reliability = float(0.99 * np.sin(np.radians(elev)) ** 0.3)

        return ChannelState(
            network_type    = NetworkType.SAT,
            rssi_dbm        = rssi,
            sinr_db         = sinr,
            latency_ms      = latency,
            throughput_mbps = throughput,
            reliability     = reliability,
            handover_cost_ms= 50.0,  # satellite handover is expensive
            available       = True,
        )

    def get_infrastructure_state(self) -> InfrastructureState:
        return InfrastructureState(
            bs_loads        = self.bs_loads.copy(),
            rsu_loads       = self.rsu_loads.copy(),
            sat_coverage    = self.sat_elevation > 10.0,
            sat_elevation_deg = self.sat_elevation,
        )

    def to_observation_vector(
        self,
        channel_states: Dict[NetworkType, ChannelState],
        current_network: NetworkType,
    ) -> np.ndarray:
        """Convert channel states to a flat observation vector."""
        obs = []
        for net_type in [NetworkType.G5, NetworkType.V2X, NetworkType.SAT]:
            cs = channel_states[net_type]
            obs.extend([
                (cs.rssi_dbm + 120) / 90,          # normalize to [0,1]
                (cs.sinr_db + 10) / 40,
                np.clip(cs.latency_ms / 1000, 0, 1),
                np.clip(cs.throughput_mbps / 1000, 0, 1),
                cs.reliability,
                float(cs.available),
            ])
        # One-hot current network
        net_onehot = np.zeros(4)
        net_onehot[int(current_network)] = 1.0
        obs.extend(net_onehot.tolist())
        return np.array(obs, dtype=np.float32)
