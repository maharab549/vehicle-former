"""Run VehicleFormer and all baselines across multiple seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from vehicleformer.models.baselines import BASELINE_REGISTRY, train_and_evaluate


SEEDS = [42, 123, 456, 789, 1337]
FULL_MODEL_NAME = "vehicleformer"


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent


def train_full_model(config_path: Path, seed: int, steps: int, output_dir: Path) -> Dict[str, object]:
    """Launch the main training script for the full VehicleFormer model."""
    repo = project_root()
    tag = f"{FULL_MODEL_NAME}/seed_{seed}"
    cmd = [
        sys.executable,
        str(repo / "train.py"),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--tag",
        tag,
    ]
    subprocess.run(cmd, cwd=str(repo), check=True)
    return {"model": FULL_MODEL_NAME, "seed": seed, "checkpoint_dir": str(output_dir / FULL_MODEL_NAME / f"seed_{seed}")}


def main() -> None:
    """Run all models across the fixed seed set."""
    parser = argparse.ArgumentParser(description="Run VehicleFormer multi-seed experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--skip-full-model", action="store_true")
    args = parser.parse_args()

    repo = project_root()
    output_dir = repo / args.output_dir
    with open(repo / args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    records: List[Dict[str, object]] = []
    for seed in SEEDS:
        if not args.skip_full_model:
            records.append(train_full_model(repo / args.config, seed, args.steps, output_dir))
        for model_name in sorted(BASELINE_REGISTRY):
            summary = train_and_evaluate(model_name, cfg, seed, args.baseline_episodes, output_dir)
            records.append(summary)

    results_path = output_dir / "multi_seed_manifest.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Saved experiment manifest to {results_path}")


if __name__ == "__main__":
    main()