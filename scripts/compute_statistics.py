"""Compute KPI statistics and significance tests for VehicleFormer experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon


METRICS = [
    "mean_reward",
    "mean_latency_ms",
    "mean_pdr",
    "mean_throughput_mbps",
    "mean_recovery_time_ms",
    "mean_spectral_efficiency",
    "handover_count",
]


def significance_marker(p_value: float) -> str:
    """Map a p-value to the requested significance marker."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def collect_results(root: Path) -> Dict[str, List[Dict[str, float]]]:
    """Load all per-seed result files under the checkpoint root."""
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for result_path in root.glob("**/results.json"):
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        grouped.setdefault(payload["model"], []).append(payload)
    return grouped


def latex_table(grouped: Dict[str, List[Dict[str, float]]], reference_model: str = "vehicleformer") -> str:
    """Build an IEEE-style LaTeX table for the aggregated metrics."""
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Multi-seed statistical comparison across models.}",
        "\\label{tab:multi_seed_stats}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Model & Reward & Latency & PDR & Throughput & Recovery & Spectral Eff. & Handover \\\",
        "\\midrule",
    ]
    reference = grouped.get(reference_model, [])
    for model_name, rows in sorted(grouped.items()):
        metric_text = []
        for metric in METRICS:
            values = np.asarray([row.get(metric, 0.0) for row in rows], dtype=np.float32)
            mean = float(values.mean()) if len(values) else 0.0
            std = float(values.std(ddof=0)) if len(values) else 0.0
            marker = ""
            if model_name != reference_model and reference and len(reference) == len(rows):
                ref_values = np.asarray([row.get(metric, 0.0) for row in reference], dtype=np.float32)
                try:
                    p_value = wilcoxon(ref_values, values, zero_method="zsplit").pvalue
                except ValueError:
                    p_value = 1.0
                marker = significance_marker(float(p_value))
            metric_text.append(f"${mean:.3f}\\pm{std:.3f}{marker}$")
        lines.append(model_name.replace("_", " ") + " & " + " & ".join(metric_text) + " \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def main() -> None:
    """Load all results and write aggregated statistics."""
    parser = argparse.ArgumentParser(description="Compute multi-seed statistics")
    parser.add_argument("--results-root", default="checkpoints")
    parser.add_argument("--output", default="logs/statistics_summary.json")
    parser.add_argument("--latex-output", default="logs/statistics_table.tex")
    args = parser.parse_args()

    root = Path(args.results_root)
    grouped = collect_results(root)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, rows in grouped.items():
        summary[model_name] = {}
        for metric in METRICS:
            values = np.asarray([row.get(metric, 0.0) for row in rows], dtype=np.float32)
            summary[model_name][metric] = {
                "mean": float(values.mean()) if len(values) else 0.0,
                "std": float(values.std(ddof=0)) if len(values) else 0.0,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    latex = latex_table(grouped)
    latex_path = Path(args.latex_output)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex, encoding="utf-8")
    print(f"Saved statistics to {output_path} and {latex_path}")


if __name__ == "__main__":
    main()