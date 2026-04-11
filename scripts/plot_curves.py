"""Plot multi-seed training curves for VehicleFormer experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_curves(root: Path) -> Dict[str, List[pd.DataFrame]]:
    """Load training curves from checkpoint directories."""
    curves: Dict[str, List[pd.DataFrame]] = {}
    for csv_path in root.glob("**/training_curve.csv"):
        model_name = csv_path.parent.parent.name
        curves.setdefault(model_name, []).append(pd.read_csv(csv_path))
    for csv_path in root.glob("../logs/**/training_log.csv"):
        model_name = "vehicleformer"
        curves.setdefault(model_name, []).append(pd.read_csv(csv_path))
    return curves


def align_curves(curves: List[pd.DataFrame]) -> np.ndarray:
    """Align variable-length curves by truncating to the minimum length."""
    min_len = min(len(frame) for frame in curves)
    return np.stack([frame.iloc[:min_len, 1].to_numpy(dtype=np.float32) for frame in curves], axis=0)


def main() -> None:
    """Generate an IEEE-style PDF plot of multi-seed reward curves."""
    parser = argparse.ArgumentParser(description="Plot multi-seed training curves")
    parser.add_argument("--results-root", default="checkpoints")
    parser.add_argument("--output", default="logs/reward_curves.pdf")
    args = parser.parse_args()

    root = Path(args.results_root)
    curves = load_curves(root)
    plt.figure(figsize=(7.0, 3.2))
    color_cycle = plt.cm.tab10.colors
    for idx, (model_name, frames) in enumerate(sorted(curves.items())):
        if not frames:
            continue
        values = align_curves(frames)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        steps = np.arange(1, len(mean) + 1)
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(steps, mean, label=model_name.replace("_", " "), color=color, linewidth=1.8)
        plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved curve plot to {output_path}")


if __name__ == "__main__":
    main()