"""
VehicleFormer Stress-Test Evaluation
=====================================
Loads trained checkpoints and evaluates under increasingly harder scenarios
to reveal meaningful performance gaps between full model and ablations.

Creates the paper's Table II / Figure 5 showing robustness comparison.

Usage:
    python scripts/stress_eval.py
"""
import sys
import os
import copy
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table as RichTable

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from vehicleformer.env.icv_env import ICVEnvironment
from vehicleformer.models.hetgnn import HetGNNEncoder, GraphBuilder
from vehicleformer.models.sac_agent import SACAgent
from vehicleformer.models.llm_prior import LLMPolicyPrior

console = Console()

# ═══════════════════════════════════════════════════════════════════════
#  Stress Scenarios — progressively harder conditions
# ═══════════════════════════════════════════════════════════════════════
STRESS_SCENARIOS = {
    "standard": {
        "desc": "Standard (training conditions)",
        "overrides": {},  # no changes
    },
    "high_outage": {
        "desc": "High RSU Outages (10× failure rate)",
        "overrides": {
            "rsu_outage_prob": 0.02,           # 10× training value
            "rsu_outage_duration": (100, 500),  # longer outages
        },
    },
    "heavy_congestion": {
        "desc": "Heavy Congestion (persistent load spikes)",
        "overrides": {
            "congestion_burst_prob": 0.02,      # 6.7× training value
            "congestion_duration": (200, 800),   # longer bursts
            "congestion_mag_range": (0.3, 0.5),  # heavier load
        },
    },
    "sparse_v2x": {
        "desc": "Sparse V2X (range halved + high outage)",
        "overrides": {
            "v2x_range": 100.0,                 # 100m vs 200m training
            "rsu_outage_prob": 0.01,
            "rsu_outage_duration": (100, 400),
        },
    },
    "nightmare": {
        "desc": "Nightmare (all stressors combined)",
        "overrides": {
            "rsu_outage_prob": 0.02,
            "rsu_outage_duration": (100, 500),
            "congestion_burst_prob": 0.02,
            "congestion_duration": (200, 800),
            "congestion_mag_range": (0.3, 0.5),
            "v2x_range": 100.0,
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════
#  Models to evaluate
# ═══════════════════════════════════════════════════════════════════════
MODELS = [
    {
        "name": "Full VehicleFormer",
        "tag": None,                  # main checkpoints
        "flags": {},
    },
    {
        "name": "w/o HetGNN",
        "tag": "ablation_no_hetgnn",
        "flags": {"no_hetgnn": True},
    },
    {
        "name": "w/o World Model",
        "tag": "ablation_no_wm",
        "flags": {"no_wm": True},
    },
    {
        "name": "w/o LLM Prior",
        "tag": "ablation_no_llm",
        "flags": {"no_llm": True},
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="VehicleFormer stress-test evaluation")
    p.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per model per scenario (default: 20)",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional subset of scenario keys (e.g., standard nightmare)",
    )
    return p.parse_args()


def apply_stress_overrides(env: ICVEnvironment, overrides: dict):
    """Modify the environment's network simulator for stress testing."""
    ns = env.net_sim
    if "rsu_outage_prob" in overrides:
        ns._rsu_outage_prob = overrides["rsu_outage_prob"]
    if "rsu_outage_duration" in overrides:
        ns._rsu_outage_duration = overrides["rsu_outage_duration"]
    if "congestion_burst_prob" in overrides:
        ns._congestion_burst_prob = overrides["congestion_burst_prob"]
    if "congestion_duration" in overrides:
        ns._congestion_duration = overrides["congestion_duration"]
    if "congestion_mag_range" in overrides:
        lo, hi = overrides["congestion_mag_range"]
        # Override the magnitude sampling in step()
        ns._stress_congestion_mag_range = (lo, hi)
    if "v2x_range" in overrides:
        ns.v2x_range_m = overrides["v2x_range"]


def load_model(cfg, model_spec, device):
    """Load a trained model from checkpoints."""
    ckpt_dir = Path(cfg['project']['checkpoint_dir'])
    if model_spec["tag"]:
        ckpt_dir = ckpt_dir / model_spec["tag"]

    flags = model_spec["flags"]
    use_hetgnn = not flags.get("no_hetgnn", False)
    emb_dim = cfg['hetgnn']['embedding_dim']
    obs_dim = (
        cfg['observation']['max_vehicles'] * cfg['observation']['vehicle_dim']
        + cfg['observation']['max_rsus'] * cfg['observation']['rsu_dim']
        + cfg['observation']['max_base_stations'] * cfg['observation']['bs_dim']
        + 3 * 6 + 4
    )

    # HetGNN or MLP fallback
    if use_hetgnn:
        hetgnn = HetGNNEncoder(cfg).to(device)
        graph_builder = GraphBuilder(cfg, device)
        hetgnn_path = ckpt_dir / "hetgnn_best.pt"
        if hetgnn_path.exists():
            hetgnn.load_state_dict(torch.load(hetgnn_path, map_location=device, weights_only=True))
        hetgnn.eval()
    else:
        import torch.nn as nn
        hetgnn = nn.Sequential(
            nn.Linear(obs_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        ).to(device)
        graph_builder = None
        # Load MLP weights from ablation checkpoint
        agent_path = ckpt_dir / "agent_best.pt"
        # The MLP is saved as part of the trainer, load separately if available
        hetgnn_path = ckpt_dir / "hetgnn_best.pt"
        if hetgnn_path.exists():
            hetgnn.load_state_dict(torch.load(hetgnn_path, map_location=device, weights_only=True))
        hetgnn.eval()

    # SAC Agent
    agent = SACAgent(cfg, device)
    agent_path = ckpt_dir / "agent_best.pt"
    if agent_path.exists():
        agent.load(str(agent_path))

    return hetgnn, graph_builder, agent, use_hetgnn


def get_embedding(obs, hetgnn, graph_builder, cfg, device, use_hetgnn):
    """Get embedding from observation."""
    if use_hetgnn:
        graph = graph_builder.build(obs)
        with torch.no_grad():
            emb, _ = hetgnn(graph["node_features"], graph["positions"])
        return emb.cpu().numpy()
    else:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            emb = hetgnn(obs_t)
        return emb.cpu().numpy()


def evaluate_model_on_scenario(
    cfg, model_spec, scenario_overrides, device,
    num_episodes=20, seed_base=1000
):
    """
    Evaluate a model under a specific stress scenario.
    Returns detailed metrics.
    """
    hetgnn, graph_builder, agent, use_hetgnn = load_model(cfg, model_spec, device)

    rewards = []
    p99_latencies = []
    mean_latencies = []
    sla_50ms_pcts = []
    sla_30ms_pcts = []
    handover_counts = []
    connection_failures = []

    for ep in range(num_episodes):
        env = ICVEnvironment(cfg)
        apply_stress_overrides(env, scenario_overrides)
        np.random.seed(seed_base + ep)  # reproducible episodes

        obs, info = env.reset()
        ep_reward = 0
        ep_handovers = 0
        ep_failures = 0
        done = False

        while not done:
            emb = get_embedding(obs, hetgnn, graph_builder, cfg, device, use_hetgnn)
            emb_t = torch.tensor(emb, device=device)
            action = agent.select_action(emb_t, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if info.get("handover", False):
                ep_handovers += 1
            if info.get("latency_ms", 0) >= 999:
                ep_failures += 1

        rewards.append(ep_reward)
        handover_counts.append(ep_handovers)
        connection_failures.append(ep_failures)

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            p99_latencies.append(m.get("p99_latency_ms", 999))
            mean_latencies.append(m.get("mean_latency_ms", 999))
            sla_50ms_pcts.append(m.get("sla_50ms_met_pct", 0))
            sla_30ms_pcts.append(m.get("sla_30ms_met_pct", 0))

        env.close()

    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "p99_latency": np.mean(p99_latencies) if p99_latencies else 999,
        "mean_latency": np.mean(mean_latencies) if mean_latencies else 999,
        "sla_50ms": np.mean(sla_50ms_pcts) if sla_50ms_pcts else 0,
        "sla_30ms": np.mean(sla_30ms_pcts) if sla_30ms_pcts else 0,
        "handovers": np.mean(handover_counts),
        "conn_failures": np.mean(connection_failures),
    }


def main():
    args = parse_args()
    console.print("\n[bold cyan]═══ VehicleFormer Stress-Test Evaluation ═══[/bold cyan]\n")

    # Load config
    cfg_path = _project_root / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    # Check which models have checkpoints
    ckpt_base = Path(cfg['project']['checkpoint_dir'])
    available_models = []
    for m in MODELS:
        d = ckpt_base / m["tag"] if m["tag"] else ckpt_base
        agent_file = d / "agent_best.pt"
        if agent_file.exists():
            available_models.append(m)
            console.print(f"  [green]✓[/green] {m['name']}: {agent_file}")
        else:
            console.print(f"  [yellow]✗[/yellow] {m['name']}: {agent_file} (not found, skipping)")

    if not available_models:
        console.print("[red]No trained models found![/red]")
        return

    num_eval_episodes = args.episodes

    scenario_keys = list(STRESS_SCENARIOS.keys())
    if args.scenarios:
        invalid = [k for k in args.scenarios if k not in STRESS_SCENARIOS]
        if invalid:
            console.print(f"[red]Invalid scenarios:[/red] {', '.join(invalid)}")
            console.print(f"[yellow]Available:[/yellow] {', '.join(scenario_keys)}")
            return
        scenario_keys = args.scenarios

    selected_scenarios = {k: STRESS_SCENARIOS[k] for k in scenario_keys}
    console.print(f"Episodes per scenario: [cyan]{num_eval_episodes}[/cyan]")
    console.print(f"Scenarios: [cyan]{', '.join(selected_scenarios.keys())}[/cyan]")

    # ─── Run all evaluations ──────────────────────────────────────────
    # results[scenario_name][model_name] = metrics_dict
    results = {}

    for sc_name, sc_info in selected_scenarios.items():
        console.print(f"\n[bold yellow]▶ Scenario: {sc_info['desc']}[/bold yellow]")
        results[sc_name] = {}

        for model_spec in available_models:
            console.print(f"  Evaluating {model_spec['name']}...", end=" ")
            metrics = evaluate_model_on_scenario(
                cfg, model_spec, sc_info["overrides"], device,
                num_episodes=num_eval_episodes,
            )
            results[sc_name][model_spec["name"]] = metrics
            console.print(
                f"reward={metrics['reward_mean']:.1f}±{metrics['reward_std']:.1f} | "
                f"P99={metrics['p99_latency']:.1f}ms | "
                f"SLA50={metrics['sla_50ms']:.1f}%"
            )

    # ─── Print Results Tables ─────────────────────────────────────────
    console.print("\n\n[bold cyan]═══ RESULTS SUMMARY ═══[/bold cyan]\n")

    # Table 1: Reward comparison
    table1 = RichTable(title="Mean Episode Reward by Scenario", show_lines=True)
    table1.add_column("Model", style="bold")
    for sc_name, sc_info in selected_scenarios.items():
        table1.add_column(sc_info["desc"][:25], justify="center")

    for model_spec in available_models:
        row = [model_spec["name"]]
        for sc_name in selected_scenarios:
            if model_spec["name"] in results[sc_name]:
                m = results[sc_name][model_spec["name"]]
                row.append(f"{m['reward_mean']:.1f}±{m['reward_std']:.1f}")
            else:
                row.append("—")
        table1.add_row(*row)
    console.print(table1)

    # Table 2: P99 Latency comparison
    table2 = RichTable(title="P99 Latency (ms) by Scenario", show_lines=True)
    table2.add_column("Model", style="bold")
    for sc_name, sc_info in selected_scenarios.items():
        table2.add_column(sc_info["desc"][:25], justify="center")

    for model_spec in available_models:
        row = [model_spec["name"]]
        for sc_name in selected_scenarios:
            if model_spec["name"] in results[sc_name]:
                m = results[sc_name][model_spec["name"]]
                row.append(f"{m['p99_latency']:.1f}")
            else:
                row.append("—")
        table2.add_row(*row)
    console.print(table2)

    # Table 3: SLA compliance
    table3 = RichTable(title="SLA Compliance (50ms target, %) by Scenario", show_lines=True)
    table3.add_column("Model", style="bold")
    for sc_name, sc_info in selected_scenarios.items():
        table3.add_column(sc_info["desc"][:25], justify="center")

    for model_spec in available_models:
        row = [model_spec["name"]]
        for sc_name in selected_scenarios:
            if model_spec["name"] in results[sc_name]:
                m = results[sc_name][model_spec["name"]]
                row.append(f"{m['sla_50ms']:.1f}%")
            else:
                row.append("—")
        table3.add_row(*row)
    console.print(table3)

    # ─── Performance Degradation Table (key for paper) ────────────────
    if "standard" in results and "nightmare" in results:
        console.print("\n")
        table4 = RichTable(
            title="Performance Degradation: Standard → Nightmare",
            show_lines=True,
        )
        table4.add_column("Model", style="bold")
        table4.add_column("Standard Reward", justify="center")
        table4.add_column("Nightmare Reward", justify="center")
        table4.add_column("Δ Reward", justify="center")
        table4.add_column("Degradation %", justify="center")
        table4.add_column("Std P99 (ms)", justify="center")
        table4.add_column("Night P99 (ms)", justify="center")

        for model_spec in available_models:
            name = model_spec["name"]
            if name in results["standard"] and name in results["nightmare"]:
                std_m = results["standard"][name]
                ngt_m = results["nightmare"][name]
                delta = std_m["reward_mean"] - ngt_m["reward_mean"]
                pct = delta / std_m["reward_mean"] * 100 if std_m["reward_mean"] != 0 else 0

                # Color-code degradation
                if pct < 5:
                    pct_str = f"[green]{pct:.1f}%[/green]"
                elif pct < 15:
                    pct_str = f"[yellow]{pct:.1f}%[/yellow]"
                else:
                    pct_str = f"[red]{pct:.1f}%[/red]"

                table4.add_row(
                    name,
                    f"{std_m['reward_mean']:.1f}",
                    f"{ngt_m['reward_mean']:.1f}",
                    f"{delta:.1f}",
                    pct_str,
                    f"{std_m['p99_latency']:.1f}",
                    f"{ngt_m['p99_latency']:.1f}",
                )
        console.print(table4)

    # ─── Save results to CSV ──────────────────────────────────────────
    csv_path = _project_root / "logs" / "stress_eval_results.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scenario,model,reward_mean,reward_std,p99_latency,mean_latency,"
                "sla_50ms,sla_30ms,handovers,conn_failures\n")
        for sc_name in selected_scenarios:
            for model_spec in available_models:
                name = model_spec["name"]
                if name in results[sc_name]:
                    m = results[sc_name][name]
                    f.write(f"{sc_name},{name},{m['reward_mean']:.2f},{m['reward_std']:.2f},"
                            f"{m['p99_latency']:.2f},{m['mean_latency']:.2f},"
                            f"{m['sla_50ms']:.2f},{m['sla_30ms']:.2f},"
                            f"{m['handovers']:.1f},{m['conn_failures']:.1f}\n")
    console.print(f"\n[green]✓ Results saved to {csv_path}[/green]")

    console.print("\n[bold cyan]═══ Stress Evaluation Complete ═══[/bold cyan]\n")


if __name__ == "__main__":
    main()
