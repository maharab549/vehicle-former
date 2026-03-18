"""
VehicleFormer Diagnostic — Red Flag Detector
=============================================
Loads the best checkpoint, runs eval episodes, and checks for:
  🚩 1) V2X over-usage (degenerate "always-V2X" policy)
  🚩 2) Simulation too easy (loads never spike, latency never hard)
  🚩 3) Insufficient handovers (agent not making real trade-offs)

Usage:
    python scripts/diagnose.py
"""
import sys, os, yaml, torch
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vehicleformer.env.icv_env import ICVEnvironment
from vehicleformer.env.network_sim import NetworkType
from vehicleformer.models.hetgnn import HetGNNEncoder, GraphBuilder
from vehicleformer.models.sac_agent import SACAgent


# ── configuration ─────────────────────────────────────────────────────
CFG_PATH        = ROOT / "configs" / "default.yaml"
CKPT_DIR        = ROOT / "checkpoints"
NUM_EPISODES    = 10
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_NAMES       = {0: "5G", 1: "V2X", 2: "SAT", 3: "5G+V2X"}


def load_config():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(cfg):
    """Load HetGNN + SAC agent from best checkpoint."""
    hetgnn = HetGNNEncoder(cfg).to(DEVICE)
    agent  = SACAgent(cfg, DEVICE)

    hetgnn_path = CKPT_DIR / "hetgnn_best.pt"
    agent_path  = CKPT_DIR / "agent_best.pt"

    if not hetgnn_path.exists() or not agent_path.exists():
        print("ERROR: best checkpoints not found in", CKPT_DIR)
        sys.exit(1)

    hetgnn.load_state_dict(torch.load(hetgnn_path, map_location=DEVICE, weights_only=True))
    agent.load(str(agent_path))  # uses custom save/load format
    hetgnn.eval()
    agent.eval()
    return hetgnn, agent


def get_embedding(hetgnn, graph_builder, obs):
    with torch.no_grad():
        graph = graph_builder.build(obs)
        emb, _ = hetgnn(graph["node_features"], graph["positions"])
        return emb


def run_diagnostic():
    cfg = load_config()
    print("=" * 70)
    print("  VehicleFormer Diagnostic — Red Flag Detector")
    print("=" * 70)

    # ── load models ───────────────────────────────────────────────────
    print("\n[1/3] Loading best checkpoint …")
    hetgnn, agent = load_models(cfg)
    graph_builder = GraphBuilder(cfg, DEVICE)
    print("  ✓ Loaded hetgnn_best.pt + agent_best.pt")

    # ── run eval episodes ─────────────────────────────────────────────
    print(f"\n[2/3] Running {NUM_EPISODES} evaluation episodes …")
    env = ICVEnvironment(cfg)

    # accumulators
    all_latencies      = []
    all_networks       = []       # which network was chosen each step
    all_handovers      = []       # per-episode handover count
    all_bs_loads       = []       # every sampled BS load
    all_rsu_loads      = []       # every sampled RSU load
    all_rewards        = []
    all_v2x_available  = []       # was V2X available at each step
    all_v2x_distances  = []       # min RSU distance at each step
    per_net_latencies  = defaultdict(list)

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_handovers = 0
        prev_net = None

        while not done:
            # embed
            emb = get_embedding(hetgnn, graph_builder, obs)
            emb_t = emb.to(DEVICE)

            # deterministic action
            action = agent.select_action(emb_t, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # ── record network choice ──────────────────────────────
            net_idx = int(np.argmax(action[:4]))
            net_name = NET_NAMES[net_idx]
            all_networks.append(net_name)

            if prev_net is not None and net_idx != prev_net:
                ep_handovers += 1
            prev_net = net_idx

            # ── record latency by network ──────────────────────────
            lat = info.get("latency_ms", 999.0)
            all_latencies.append(lat)
            per_net_latencies[net_name].append(lat)

            # ── record infrastructure loads ────────────────────────
            all_bs_loads.append(env.net_sim.bs_loads.copy())
            all_rsu_loads.append(env.net_sim.rsu_loads.copy())

            # ── record V2X reachability ────────────────────────────
            ego = env._get_ego_vehicle(env.sumo.get_vehicle_states())
            if ego is not None:
                dists = np.linalg.norm(
                    np.array(cfg["simulation"]["rsu_positions"]) - ego.position,
                    axis=1,
                )
                min_rsu_dist = dists.min()
                all_v2x_distances.append(min_rsu_dist)
                all_v2x_available.append(min_rsu_dist <= 300.0)

        all_handovers.append(ep_handovers)
        all_rewards.append(ep_reward)
        print(f"  Ep {ep+1:>2}/{NUM_EPISODES}  reward={ep_reward:+8.1f}  "
              f"handovers={ep_handovers:>3}  steps={info.get('episode_step', '?')}")

    env.close()

    # ── analysis ──────────────────────────────────────────────────────
    print(f"\n[3/3] Analysis\n{'─' * 70}")

    lats = np.array(all_latencies)
    bs_loads = np.concatenate(all_bs_loads)          # (steps*n_bs,)
    rsu_loads = np.concatenate(all_rsu_loads)

    net_counts = Counter(all_networks)
    total_steps = len(all_networks)

    # ── Network usage distribution ────────────────────────────────────
    print("\n📊  Network Usage Distribution:")
    for net in ["5G", "V2X", "SAT", "5G+V2X"]:
        pct = net_counts.get(net, 0) / total_steps * 100
        bar = "█" * int(pct / 2)
        print(f"    {net:<8s}  {pct:5.1f}%  {bar}")

    v2x_pct = net_counts.get("V2X", 0) / total_steps * 100

    # ── Handovers ─────────────────────────────────────────────────────
    mean_ho = np.mean(all_handovers)
    print(f"\n🔄  Handovers per episode: mean={mean_ho:.1f}  "
          f"min={min(all_handovers)}  max={max(all_handovers)}")

    # ── Latency histogram ─────────────────────────────────────────────
    print("\n⏱️   Latency Summary:")
    print(f"    Mean:  {lats.mean():.1f} ms")
    print(f"    P50:   {np.percentile(lats, 50):.1f} ms")
    print(f"    P95:   {np.percentile(lats, 95):.1f} ms")
    print(f"    P99:   {np.percentile(lats, 99):.1f} ms")
    print(f"    P999:  {np.percentile(lats, 99.9):.1f} ms")
    print(f"    Max:   {lats.max():.1f} ms")

    bins = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
    hist, _ = np.histogram(lats, bins=bins)
    print("\n    Latency Distribution:")
    for i, count in enumerate(hist):
        pct = count / len(lats) * 100
        bar = "█" * int(pct / 2)
        print(f"    {bins[i]:>5d}-{bins[i+1]:<5d} ms: {pct:5.1f}%  {bar}")

    # ── Per-network latency ───────────────────────────────────────────
    print("\n📡  Latency by Network Type:")
    for net in ["5G", "V2X", "SAT", "5G+V2X"]:
        vals = per_net_latencies.get(net, [])
        if vals:
            arr = np.array(vals)
            print(f"    {net:<8s}  mean={arr.mean():6.1f}ms  "
                  f"P99={np.percentile(arr, 99):6.1f}ms  "
                  f"n={len(vals)}")

    # ── Infrastructure loads ──────────────────────────────────────────
    print("\n🏗️   Infrastructure Loads:")
    print(f"    BS  loads — mean={bs_loads.mean():.3f}  "
          f"max={bs_loads.max():.3f}  >0.8: {(bs_loads > 0.8).mean()*100:.1f}%")
    print(f"    RSU loads — mean={rsu_loads.mean():.3f}  "
          f"max={rsu_loads.max():.3f}  >0.8: {(rsu_loads > 0.8).mean()*100:.1f}%")

    # ── V2X reachability ──────────────────────────────────────────────
    v2x_avail_pct = np.mean(all_v2x_available) * 100 if all_v2x_available else 0
    v2x_dists = np.array(all_v2x_distances) if all_v2x_distances else np.array([999])
    print(f"\n📶  V2X Reachability:")
    print(f"    % steps within 300m of RSU: {v2x_avail_pct:.1f}%")
    print(f"    Min RSU distance — mean={v2x_dists.mean():.0f}m  "
          f"P50={np.percentile(v2x_dists, 50):.0f}m  "
          f"max={v2x_dists.max():.0f}m")

    # ── Reward ────────────────────────────────────────────────────────
    print(f"\n💰  Reward per episode: mean={np.mean(all_rewards):.1f}  "
          f"std={np.std(all_rewards):.1f}")

    # ═══════════════ VERDICT ═══════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  VERDICT")
    print("═" * 70)

    flags = []

    # Flag 1: V2X overuse
    if v2x_pct > 85:
        flags.append(
            f"🚩 RED FLAG 1 — V2X usage {v2x_pct:.0f}% (>85%). "
            f"Agent learned 'always pick V2X'. Degenerate policy.\n"
            f"   FIX: Reduce RSU count from 12→4, spread them out, "
            f"or add RSU outage periods."
        )
    elif v2x_pct > 60:
        flags.append(
            f"⚠️  YELLOW FLAG — V2X usage {v2x_pct:.0f}% is high. "
            f"Not necessarily degenerate, but check if it's V2X+5G combined."
        )

    # Flag 2: Too few handovers
    if mean_ho < 2:
        flags.append(
            f"🚩 RED FLAG 2 — Mean handovers {mean_ho:.1f}/episode (<2). "
            f"Agent never switches networks. No real decision-making.\n"
            f"   FIX: Add coverage gaps, RSU failures, or dynamic load spikes."
        )
    elif mean_ho < 5:
        flags.append(
            f"⚠️  YELLOW FLAG — Mean handovers {mean_ho:.1f}/episode. "
            f"Low but possibly OK if scenario is simple."
        )

    # Flag 3: Simulation too easy (loads never dangerous)
    if bs_loads.max() < 0.8 and rsu_loads.max() < 0.8:
        flags.append(
            f"🚩 RED FLAG 3 — Max BS load {bs_loads.max():.2f}, "
            f"Max RSU load {rsu_loads.max():.2f}. Loads never spike.\n"
            f"   FIX: Add congestion events or correlated load bursts."
        )

    # Flag 4: V2X always available
    if v2x_avail_pct > 98:
        flags.append(
            f"🚩 RED FLAG 4 — V2X reachable {v2x_avail_pct:.0f}% of steps. "
            f"12 RSUs blanket 800×800m grid. V2X is trivially always best.\n"
            f"   FIX: Reduce to 4-5 RSUs so vehicles must handle dead zones."
        )

    # Flag 5: P99 implausibly low
    p99 = np.percentile(lats, 99)
    if p99 < 30:
        flags.append(
            f"🚩 RED FLAG 5 — P99 latency {p99:.1f}ms (<30ms). "
            f"Beating the 2030 ICV target trivially suggests the sim is too easy.\n"
            f"   FIX: Add more realistic channel fading, interference, or congestion."
        )

    if flags:
        for f in flags:
            print(f"\n{f}")
        print(f"\n{'─' * 70}")
        print(f"  TOTAL FLAGS: {len(flags)}")
        print(f"  Likely cause: scenario design is too favorable, not a code bug.")
        print(f"  The agent IS learning — but it's learning a trivial problem.")
    else:
        print("\n✅  All checks passed. Agent appears to be making genuine trade-offs.")
        print("    Network usage is diverse, handovers are present, loads are challenging.")

    print("═" * 70)


if __name__ == "__main__":
    run_diagnostic()
