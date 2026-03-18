"""KPI metrics aligned with China ICV Roadmap 2025-2030 targets."""
import numpy as np
from typing import List, Dict
from rich.table import Table
from rich.console import Console

console = Console()

# Roadmap KPI targets
TARGETS = {
    "p99_latency_ms"     : 50.0,    # 5G 50ms@99%
    "v2x_avg_latency_ms" : 30.0,    # V2X average latency
    "reliability"        : 0.99,    # 99% PDR
    "urban_coverage"     : 0.98,    # 98% coverage
}


class MetricsTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.latencies     = []
        self.reliabilities = []
        self.networks_used = []

    def update(self, info: dict):
        if "latency_ms" in info:
            self.latencies.append(info["latency_ms"])
        if "reliability" in info:
            self.reliabilities.append(info["reliability"])
        if "network" in info:
            self.networks_used.append(info["network"])

    def compute(self) -> Dict[str, float]:
        if not self.latencies:
            return {}
        lats = np.array(self.latencies)
        rels = np.array(self.reliabilities) if self.reliabilities else np.array([0])
        return {
            "mean_latency_ms"    : float(lats.mean()),
            "p50_latency_ms"     : float(np.percentile(lats, 50)),
            "p99_latency_ms"     : float(np.percentile(lats, 99)),
            "mean_reliability"   : float(rels.mean()),
            "sla_50ms_met_pct"   : float((lats <= 50).mean() * 100),
            "sla_30ms_met_pct"   : float((lats <= 30).mean() * 100),
        }

    def print_kpi_table(self, metrics: Dict[str, float]):
        table = Table(title="📊 ICV Roadmap KPI Report", style="cyan")
        table.add_column("Metric",         style="white",  width=28)
        table.add_column("Your Result",    style="bold",   width=16)
        table.add_column("Roadmap Target", style="yellow", width=18)
        table.add_column("Status",         style="bold",   width=10)

        rows = [
            ("P99 Latency",      "p99_latency_ms",    50,    "ms",  True),
            ("Mean Reliability", "mean_reliability",  0.99,  "",    False),
            ("SLA 50ms met",     "sla_50ms_met_pct",  99.0,  "%",   False),
            ("SLA 30ms met",     "sla_30ms_met_pct",  None,  "%",   False),
        ]
        for label, key, target, unit, lower_is_better in rows:
            val = metrics.get(key)
            if val is None:
                continue
            val_str = f"{val:.2f}{unit}"
            tgt_str = f"{target}{unit}" if target is not None else "N/A"
            if target is not None:
                met = (val <= target) if lower_is_better else (val >= target)
                status = "[green]✓ MET[/green]" if met else "[red]✗ MISS[/red]"
            else:
                status = "—"
            table.add_row(label, val_str, tgt_str, status)
        console.print(table)
