"""
VehicleFormer Installation Verifier
====================================
Run this before starting training to make sure everything works.

Usage: python scripts/verify_install.py
"""

import sys
import traceback
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

CHECKS = []

def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("Python version")
def check_python():
    assert sys.version_info >= (3, 10), f"Need Python 3.10+, got {sys.version}"
    return f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

@check("PyTorch + CUDA")
def check_torch():
    import torch
    cuda_ok = torch.cuda.is_available()
    ver = torch.__version__
    if cuda_ok:
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"torch {ver} | CUDA ✓ | {gpu} | {vram:.1f}GB"
    return f"torch {ver} | CPU only"

@check("NumPy")
def check_numpy():
    return f"numpy {np.__version__}"

@check("Gymnasium")
def check_gym():
    import gymnasium
    return f"gymnasium {gymnasium.__version__}"

@check("YAML config")
def check_config():
    import yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "hetgnn" in cfg
    return "configs/default.yaml loaded ✓"

@check("Network Simulator")
def check_net_sim():
    import yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.env.network_sim import NetworkSimulator, NetworkType
    sim = NetworkSimulator(cfg, rng_seed=42)
    pos = np.array([400.0, 400.0])
    states = sim.get_channel_states(pos, 10.0)
    assert NetworkType.G5 in states
    lat = states[NetworkType.G5].latency_ms
    return f"5G latency @ center: {lat:.1f}ms"

@check("SUMO Bridge (mock mode)")
def check_sumo():
    import yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.env.sumo_env import SUMOBridge
    bridge = SUMOBridge(cfg)
    bridge.start()
    bridge.step()
    vs = bridge.get_vehicle_states()
    bridge.close()
    return f"{len(vs)} vehicles in mock mode"

@check("ICVEnvironment (reset + step)")
def check_env():
    import yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.env.icv_env import ICVEnvironment
    env = ICVEnvironment(cfg)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)
    env.close()
    return f"obs_dim={obs.shape[0]} | reward={reward:.3f}"

@check("HetGNN Encoder")
def check_hetgnn():
    import torch, yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.models.hetgnn import HetGNNEncoder, GraphBuilder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HetGNNEncoder(cfg).to(device)
    builder = GraphBuilder(cfg, device)
    obs = np.random.rand(cfg['observation']['max_vehicles'] * cfg['observation']['vehicle_dim']
                         + cfg['observation']['max_rsus'] * cfg['observation']['rsu_dim']
                         + cfg['observation']['max_base_stations'] * cfg['observation']['bs_dim']
                         + 22).astype(np.float32)
    graph = builder.build(obs)
    emb, node_embs = model(graph["node_features"], graph["positions"])
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"emb_dim={emb.shape[0]} | params={params:.2f}M | device={device}"

@check("World Model (forward + loss)")
def check_world_model():
    import torch, yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.models.world_model import WorldModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg['world_model']['d_model']
    T = cfg['world_model']['max_seq_len']
    k = cfg['world_model']['prediction_horizon']
    model = WorldModel(cfg).to(device)
    h_seq = torch.randn(4, T, d, device=device)
    out = model(h_seq)
    assert out["mean_pred"].shape == (4, k, d)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"pred_shape={out['mean_pred'].shape} | params={params:.2f}M"

@check("SAC Agent (action selection)")
def check_sac():
    import torch, yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.models.sac_agent import SACAgent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(cfg, device)
    emb = torch.randn(cfg['hetgnn']['embedding_dim'], device=device)
    action = agent.select_action(emb, deterministic=False)
    params = sum(p.numel() for p in agent.actor.parameters()
                 ) + sum(p.numel() for p in agent.critic.parameters())
    return f"action_dim={len(action)} | actor+critic params={params/1e6:.2f}M"

@check("Replay Buffer")
def check_buffer():
    import torch, yaml
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from vehicleformer.training.replay_buffer import ReplayBuffer
    device = torch.device("cpu")
    obs_dim = 100
    act_dim = cfg['action']['action_dim']
    buf = ReplayBuffer(cfg, obs_dim, act_dim, device)
    for _ in range(200):
        buf.add(
            np.random.rand(obs_dim).astype(np.float32),
            np.random.rand(act_dim).astype(np.float32),
            float(np.random.rand()),
            np.random.rand(obs_dim).astype(np.float32),
            False,
        )
    batch = buf.sample(32)
    return f"capacity={buf.capacity} | size={len(buf)} | batch_keys={list(batch.keys())}"


def main():
    console.print("\n[bold cyan]═══ VehicleFormer Installation Verification ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Check",   width=32)
    table.add_column("Result",  width=55)
    table.add_column("Status",  width=10)

    all_passed = True
    for name, fn in CHECKS:
        try:
            result = fn()
            table.add_row(name, str(result), "[bold green]PASS[/bold green]")
        except Exception as e:
            table.add_row(name, f"[red]{str(e)[:50]}[/red]", "[bold red]FAIL[/bold red]")
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("\n[bold green]✓ All checks passed! Ready to train.[/bold green]")
        console.print("\n[yellow]Start training:[/yellow]")
        console.print("  [cyan]python train.py --config configs/default.yaml[/cyan]\n")
    else:
        console.print("\n[bold red]✗ Some checks failed. Fix errors above before training.[/bold red]")
        console.print("[yellow]Most common fix (activate venv first):[/yellow]")
        console.print("  [cyan]source venv/bin/activate[/cyan]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
