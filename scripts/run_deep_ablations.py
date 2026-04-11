"""Run the four deep ablation groups for VehicleFormer."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


SEEDS = [42, 123, 456, 789, 1337]

ABLATION_GROUPS = {
    "encoder": {
        "full_hetgnn": {},
        "homogeneous_gnn": {"hetgnn": {"homogeneous_weights": True}},
        "graphsage": {"baseline_model": "graphsage_dqn"},
        "mlp_only": {"hetgnn": {"enabled": False}},
        "raw_features": {"hetgnn": {"enabled": False}, "sac": {"hidden_dims": [256, 128]}},
    },
    "world_model": {
        "h0": {"world_model": {"enabled": False, "prediction_horizon": 0}},
        "h1": {"world_model": {"prediction_horizon": 1}},
        "h3": {"world_model": {"prediction_horizon": 3}},
        "h5": {"world_model": {"prediction_horizon": 5}},
        "h10": {"world_model": {"prediction_horizon": 10}},
    },
    "llm_beta": {
        "beta0": {"llm_prior": {"enabled": False, "beta_schedule": "fixed", "beta_fixed": 0.0}},
        "beta001": {"llm_prior": {"beta_schedule": "fixed", "beta_fixed": 0.01}},
        "beta01": {"llm_prior": {"beta_schedule": "fixed", "beta_fixed": 0.1}},
        "beta1": {"llm_prior": {"beta_schedule": "fixed", "beta_fixed": 1.0}},
        "adaptive": {"llm_prior": {"enabled": True, "beta_schedule": "adaptive"}},
    },
    "stress": {
        "full": {},
        "no_world_model": {"world_model": {"enabled": False}},
        "no_llm": {"llm_prior": {"enabled": False}},
        "no_hetgnn": {"hetgnn": {"enabled": False}},
        "random_fallback": {"baseline_model": "random"},
    },
}


def deep_update(base: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    """Recursively merge dictionaries for ablation configs."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    """Generate per-ablation configs and launch runs."""
    parser = argparse.ArgumentParser(description="Run deep VehicleFormer ablations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--output-dir", default="configs/generated_ablations")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    with open(repo / args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []

    for group_name, conditions in ABLATION_GROUPS.items():
        for condition_name, updates in conditions.items():
            for seed in SEEDS:
                run_cfg = deep_update(cfg, updates)
                run_cfg["project"]["seed"] = seed
                config_path = output_dir / f"{group_name}_{condition_name}_seed_{seed}.yaml"
                with open(config_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(run_cfg, handle, sort_keys=False)
                tag = f"ablation_{group_name}_{condition_name}/seed_{seed}"
                cmd = [sys.executable, str(repo / "train.py"), "--config", str(config_path), "--steps", str(args.steps), "--tag", tag]
                subprocess.run(cmd, cwd=str(repo), check=True)
                manifest.append({"group": group_name, "condition": condition_name, "seed": seed, "config": str(config_path), "tag": tag})

    manifest_path = output_dir / "ablation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved ablation manifest to {manifest_path}")


if __name__ == "__main__":
    main()