"""
SUMO Traffic Simulator Bridge for VehicleFormer.
Uses TraCI (Traffic Control Interface) to communicate with SUMO.
"""
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ─── SUMO / TraCI import ────────────────────────────────────────────────
def _setup_sumo():
    """Add SUMO to path from SUMO_HOME environment variable."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home is None:
        # Try common locations (Windows + Linux)
        candidates = [
            "D:\\sum",
            "C:\\Program Files (x86)\\Eclipse\\Sumo",
            "C:\\Program Files\\Eclipse\\Sumo",
            "/usr/share/sumo",
            "/usr/local/share/sumo",
            os.path.expanduser("~/sumo"),
        ]
        for c in candidates:
            if Path(c).exists():
                sumo_home = c
                os.environ["SUMO_HOME"] = c
                break
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    return sumo_home

SUMO_HOME = _setup_sumo()

try:
    import traci
    import sumolib
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("⚠ TraCI not found — using mock SUMO mode for development")


@dataclass
class VehicleState:
    """State of a single vehicle from SUMO."""
    vehicle_id: str
    position: np.ndarray      # (2,) [x, y] meters
    speed: float              # m/s
    heading: float            # degrees [0, 360]
    acceleration: float       # m/s^2
    edge_id: str              # current road edge
    lane_id: str              # current lane


class SUMOBridge:
    """
    Interface between VehicleFormer and SUMO traffic simulator.
    
    Manages TraCI connection, vehicle tracking, and state extraction.
    Falls back to mock mode if SUMO is not available.
    """

    _instance_counter = 0  # class-level counter for unique labels

    def __init__(self, cfg: dict, use_gui: bool = False):
        self.cfg = cfg
        self.use_gui = use_gui
        self.sumo_cfg = cfg['simulation']['sumo_cfg']
        self.step_length = cfg['simulation']['step_length']
        self.max_vehicles = cfg['simulation']['num_vehicles']
        self._connected = False
        self._step = 0
        self._use_mock = not TRACI_AVAILABLE

        # Unique label and port per instance (avoids TraCI "already active" error)
        SUMOBridge._instance_counter += 1
        self._label = f"vf_{SUMOBridge._instance_counter}"
        import random
        self._port = random.randint(8000, 9000) + SUMOBridge._instance_counter

    # ─── Connection Management ───────────────────────────────────────────

    def start(self) -> None:
        """Start SUMO simulation and connect via TraCI."""
        if self._use_mock:
            self._start_mock()
            return

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--collision.action", "warn",
        ]
        try:
            traci.start(sumo_cmd, port=self._port, label=self._label)
            self._conn = traci.getConnection(self._label)
            self._connected = True
            self._step = 0
            print(f"  ✓ SUMO started (port {self._port}, label={self._label})")
        except Exception as e:
            print(f"  ⚠ SUMO failed to start ({e}) — falling back to mock mode")
            self._use_mock = True
            self._start_mock()

    def close(self) -> None:
        """Close SUMO connection."""
        if self._connected and not self._use_mock:
            try:
                self._conn.close()
            except Exception:
                pass
        self._connected = False

    def reset(self) -> None:
        """Reset simulation to start."""
        self.close()
        self.start()

    # ─── Simulation Step ─────────────────────────────────────────────────

    def step(self) -> None:
        """Advance SUMO simulation by one step."""
        if self._use_mock:
            self._step_mock()
            return
        self._conn.simulationStep()
        self._step += 1

    # ─── Vehicle State Extraction ────────────────────────────────────────

    def get_vehicle_states(self) -> List[VehicleState]:
        """Get current state of all vehicles in simulation."""
        if self._use_mock:
            return self._get_mock_states()

        vehicle_ids = self._conn.vehicle.getIDList()
        states = []
        for vid in vehicle_ids[:self.max_vehicles]:
            try:
                pos = self._conn.vehicle.getPosition(vid)
                states.append(VehicleState(
                    vehicle_id   = vid,
                    position     = np.array(pos, dtype=np.float32),
                    speed        = self._conn.vehicle.getSpeed(vid),
                    heading      = self._conn.vehicle.getAngle(vid),
                    acceleration = self._conn.vehicle.getAcceleration(vid),
                    edge_id      = self._conn.vehicle.getRoadID(vid),
                    lane_id      = self._conn.vehicle.getLaneID(vid),
                ))
            except Exception:
                continue
        return states

    def get_num_vehicles(self) -> int:
        """Get number of active vehicles."""
        if self._use_mock:
            return len(self._mock_vehicles)
        return self._conn.vehicle.getIDCount()

    def get_sim_time(self) -> float:
        """Get current simulation time in seconds."""
        if self._use_mock:
            return self._step * self.step_length
        return self._conn.simulation.getTime()

    def is_done(self) -> bool:
        """Check if simulation has ended."""
        if self._use_mock:
            return self._step >= 1000
        return self._conn.simulation.getMinExpectedNumber() <= 0

    # ─── Mock Mode (when SUMO not available) ────────────────────────────
    # This lets you develop/test code without SUMO running

    def _start_mock(self) -> None:
        """Initialize mock vehicle simulation."""
        self._step = 0
        n = self.max_vehicles
        self._mock_vehicles = {
            f"veh_{i}": {
                "pos": np.random.uniform([0, 0], [800, 800]).astype(np.float32),
                "vel": np.random.uniform([5, 5], [10, 10]).astype(np.float32),
                "speed": float(np.random.uniform(5, 14)),
                "heading": float(np.random.uniform(0, 360)),
            }
            for i in range(n)
        }
        print("  ⚠ Using mock SUMO mode (install SUMO for real simulation)")

    def _step_mock(self) -> None:
        """Advance mock simulation."""
        self._step += 1
        for vid, v in self._mock_vehicles.items():
            # Simple random walk with boundary reflection
            v["pos"] += v["vel"] * self.step_length
            # Bounce off boundaries [0, 800]
            for dim in range(2):
                if v["pos"][dim] < 0 or v["pos"][dim] > 800:
                    v["vel"][dim] *= -1
                    v["pos"][dim] = np.clip(v["pos"][dim], 0, 800)
            v["speed"] = float(np.linalg.norm(v["vel"]))
            v["heading"] = float(np.degrees(np.arctan2(v["vel"][1], v["vel"][0])) % 360)

    def _get_mock_states(self) -> List[VehicleState]:
        return [
            VehicleState(
                vehicle_id   = vid,
                position     = v["pos"].copy(),
                speed        = v["speed"],
                heading      = v["heading"],
                acceleration = 0.0,
                edge_id      = "mock_edge",
                lane_id      = "mock_lane",
            )
            for vid, v in self._mock_vehicles.items()
        ]
