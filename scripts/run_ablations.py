"""Run VehicleFormer ablations and external baselines under a shared protocol."""
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import yaml

from vehicleformer.models.baselines import BASELINE_REGISTRY, train_and_evaluate

# Use venv Python if available, otherwise sys.executable
_project_root = Path(__file__).resolve().parent.parent
_venv_python = _project_root / "venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable
STEPS = 100_000  # sufficient to show performance gap

ABLATIONS = [
    {
        "name": "ablation_no_hetgnn",
        "flags": ["--no-hetgnn"],
        "desc": "Ablation 1/3: No HetGNN (flat MLP baseline)",
    },
    {
        "name": "ablation_no_wm",
        "flags": ["--no-wm"],
        "desc": "Ablation 2/3: No World Model",
    },
    {
        "name": "ablation_no_llm",
        "flags": ["--no-llm"],
        "desc": "Ablation 3/3: No LLM Prior",
    },
]

BASELINES = sorted(BASELINE_REGISTRY)


def run_ablation(abl: dict):
    print(f"\n{'='*70}")
    print(f"  {abl['desc']}")
    print(f"  Tag: {abl['name']}  |  Steps: {STEPS:,}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    cmd = [
        PYTHON, "train.py",
        "--config", "configs/default.yaml",
        "--steps", str(STEPS),
        "--tag", abl["name"],
    ] + abl["flags"]

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_project_root)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_project_root), text=True, env=env)
    elapsed = time.time() - t0

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{status}] {abl['name']} finished in {elapsed/3600:.1f}h")
    return result.returncode == 0


def main():
    print("=" * 70)
    print("  VehicleFormer Ablation Studies")
    print(f"  {len(ABLATIONS)} runs × {STEPS:,} steps each")
    print("=" * 70)

    results = {}
    for abl in ABLATIONS:
        ok = run_ablation(abl)
        results[abl["name"]] = ok
        if not ok:
            print(f"\n[WARNING] {abl['name']} failed — continuing to next ablation")

    with open(_project_root / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    baseline_results = []
    for model_name in BASELINES:
        print(f"\n[BASELINE] Training {model_name}")
        summary = train_and_evaluate(model_name, cfg, seed=cfg["project"]["seed"], episodes=20, output_dir=_project_root / "checkpoints")
        baseline_results.append(summary)

    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}")
    for row in baseline_results:
        print(f"  ✓ baseline {row['model']}: reward={row['mean_reward']:.3f}")
    print("=" * 70)
    print(f"\nCheck logs/ and checkpoints/ for per-ablation results.")
    print("Compare against full model best eval: 799.053 @ 275K steps")
    with open(_project_root / "logs" / "ablation_baseline_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"ablations": results, "baselines": baseline_results}, handle, indent=2)


if __name__ == "__main__":
    main()
