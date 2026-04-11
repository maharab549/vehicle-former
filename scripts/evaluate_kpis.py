"""Evaluate checkpoints and export KPI summaries in LaTeX format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from vehicleformer.training.trainer import VehicleFormerTrainer


SEEDS = [42, 123, 456, 789, 1337]


def aggregate(metrics: List[Dict[str, float]]) -> Dict[str, str]:
    """Aggregate KPI lists as mean plus/minus std strings."""
    keys = sorted({key for metric in metrics for key in metric})
    summary: Dict[str, str] = {}
    for key in keys:
        values = np.asarray([metric[key] for metric in metrics if key in metric], dtype=np.float32)
        if len(values):
            summary[key] = f"{values.mean():.3f}\\pm{values.std(ddof=0):.3f}"
    return summary


def latex_table(summary: Dict[str, str]) -> str:
    """Format a single-model KPI table for IEEE LaTeX."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{VehicleFormer KPI evaluation across 50 episodes and 5 seeds.}",
        "\\label{tab:kpi_eval}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Mean $\\pm$ Std \\\",
        "\\midrule",
    ]
    for key, value in summary.items():
        lines.append(f"{key.replace('_', ' ')} & ${value}$ \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main() -> None:
    """Evaluate a set of checkpoints and write KPI summaries."""
    parser = argparse.ArgumentParser(description="Evaluate VehicleFormer KPIs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint-root", default="checkpoints/vehicleformer")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-json", default="logs/kpi_eval.json")
    parser.add_argument("--output-latex", default="logs/kpi_eval.tex")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    all_metrics: List[Dict[str, float]] = []
    checkpoint_root = Path(args.checkpoint_root)
    for seed in SEEDS:
        cfg["project"]["seed"] = seed
        trainer = VehicleFormerTrainer(cfg)
        ckpt_dir = checkpoint_root / f"seed_{seed}"
        agent_path = ckpt_dir / "agent_best.pt"
        if agent_path.exists():
            trainer.resume_from_checkpoint(str(agent_path))
        _, kpis = [], []
        for _ in range(args.episodes):
            obs, _ = trainer.eval_env.reset(seed=seed)
            done = False
            info = {}
            while not done:
                emb = trainer._get_embedding(obs)
                action = trainer.agent.select_action(torch.tensor(emb, device=trainer.device), deterministic=True)
                obs, _, terminated, truncated, info = trainer.eval_env.step(action)
                done = terminated or truncated
            if "episode_metrics" in info:
                all_metrics.append(info["episode_metrics"])
        trainer.eval_env.close()
        trainer.env.close()

    summary = aggregate(all_metrics)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_latex = Path(args.output_latex)
    output_latex.parent.mkdir(parents=True, exist_ok=True)
    output_latex.write_text(latex_table(summary), encoding="utf-8")
    print(f"Saved KPI summary to {output_json} and {output_latex}")


if __name__ == "__main__":
    main()