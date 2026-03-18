"""Logger with TensorBoard support."""
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    SummaryWriter = None


class Logger:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        run_name = f"vehicleformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir  = Path(cfg['project']['log_dir']) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Save config
        with open(log_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

        # TensorBoard
        if TB_AVAILABLE:
            self.writer = SummaryWriter(str(log_dir))
        else:
            self.writer = None

        self.csv_path = log_dir / "training_log.csv"
        with open(self.csv_path, "w") as f:
            f.write("step,episode,reward,mean_latency_ms,p99_latency_ms,sla_50ms_pct,sla_30ms_pct,mean_reliability\n")

        self.eval_csv_path = log_dir / "eval_log.csv"
        with open(self.eval_csv_path, "w") as f:
            f.write("step,mean_reward,mean_latency_ms,p99_latency_ms,sla_50ms_pct,sla_30ms_pct,mean_reliability\n")

    def log_losses(self, losses: dict, step: int):
        if self.writer:
            for k, v in losses.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"loss/{k}", v, step)

    def log_episode(self, episode: int, reward: float, steps: int,
                    duration: float, extra_metrics: Optional[dict] = None):
        if self.writer:
            self.writer.add_scalar("train/episode_reward", reward, episode)
            self.writer.add_scalar("train/episode_steps", steps, episode)
            if extra_metrics:
                for k, v in extra_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, episode)
        # CSV
        mean_latency = extra_metrics.get("mean_latency_ms", 0) if extra_metrics else 0
        p99 = extra_metrics.get("p99_latency_ms", 0) if extra_metrics else 0
        sla = extra_metrics.get("sla_50ms_met_pct", 0) if extra_metrics else 0
        sla30 = extra_metrics.get("sla_30ms_met_pct", 0) if extra_metrics else 0
        mean_reliability = extra_metrics.get("mean_reliability", 0) if extra_metrics else 0
        with open(self.csv_path, "a") as f:
            f.write(f"{episode},{episode},{reward:.4f},{mean_latency:.2f},{p99:.2f},{sla:.2f},{sla30:.2f},{mean_reliability:.4f}\n")

    def log_eval(self, mean_reward: float, kpis: list, step: int):
        if self.writer:
            self.writer.add_scalar("eval/mean_reward", mean_reward, step)
            if kpis:
                for k in kpis[0].keys():
                    vals = [ep[k] for ep in kpis if k in ep]
                    if vals:
                        self.writer.add_scalar(f"eval/{k}", np.mean(vals), step)
        if kpis:
            def mean_for(key: str) -> float:
                vals = [ep[key] for ep in kpis if key in ep]
                return float(np.mean(vals)) if vals else 0.0
            with open(self.eval_csv_path, "a") as f:
                f.write(
                    f"{step},{mean_reward:.4f},{mean_for('mean_latency_ms'):.2f},"
                    f"{mean_for('p99_latency_ms'):.2f},{mean_for('sla_50ms_met_pct'):.2f},"
                    f"{mean_for('sla_30ms_met_pct'):.2f},{mean_for('mean_reliability'):.4f}\n"
                )
