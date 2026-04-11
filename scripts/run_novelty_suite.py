"""Run novelty-focused ablations for domain randomization and invariance learning."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


SEEDS = [42, 123, 456, 789, 1337]


def deep_update(base: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    """Recursive dictionary merge helper."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    """Run novelty study conditions and write a manifest."""
    parser = argparse.ArgumentParser(description="Run novelty study for VehicleFormer")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--output-dir", default="configs/generated_novelty")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    with open(repo / args.config, "r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    conditions = {
        "full_novel": {},
        "no_domain_randomization": {"novelty": {"domain_randomization": {"enabled": False}}},
        "no_invariance": {"novelty": {"invariance": {"enabled": False}}},
        "no_novelty": {
            "novelty": {
                "domain_randomization": {"enabled": False},
                "invariance": {"enabled": False},
            }
        },
    }

    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []

    for condition_name, updates in conditions.items():
        for seed in SEEDS:
            run_cfg = deep_update(base_cfg, updates)
            run_cfg["project"]["seed"] = seed
            config_path = output_dir / f"{condition_name}_seed_{seed}.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(run_cfg, handle, sort_keys=False)
            tag = f"novelty_{condition_name}/seed_{seed}"
            cmd = [sys.executable, str(repo / "train.py"), "--config", str(config_path), "--steps", str(args.steps), "--tag", tag]
            subprocess.run(cmd, cwd=str(repo), check=True)
            manifest.append({"condition": condition_name, "seed": seed, "config": str(config_path), "tag": tag})

    manifest_path = output_dir / "novelty_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved novelty manifest to {manifest_path}")


if __name__ == "__main__":
    main()