"""
VehicleFormer — Main Training Entry Point
==========================================
Usage:
    python train.py
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --device cuda
    python train.py --config configs/default.yaml --no-llm  # skip LLM prior
"""
import argparse
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Train VehicleFormer")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None, help="Override device: cuda/cpu")
    p.add_argument("--no-llm",    action="store_true", help="Disable LLM prior")
    p.add_argument("--no-wm",     action="store_true", help="Disable world model")
    p.add_argument("--no-hetgnn", action="store_true", help="Disable HetGNN (use flat MLP)")
    p.add_argument("--gui",       action="store_true", help="Enable SUMO GUI")
    p.add_argument("--steps",   type=int, default=None, help="Override total_timesteps")
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--resume",  default=None,
                   help="Resume from checkpoint dir or tag (e.g. 'best', 'step_150000', or full path)")
    p.add_argument("--tag",     default=None,
                   help="Run tag for separate checkpoint/log dirs (e.g. 'ablation_no_hetgnn')")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if args.device:
        cfg['project']['device'] = args.device
    if args.no_llm:
        cfg['llm_prior']['enabled'] = False
    if args.no_wm:
        cfg['world_model']['enabled'] = False
    if args.no_hetgnn:
        cfg['hetgnn']['enabled'] = False
    if args.gui:
        cfg['simulation']['use_gui'] = True
    if args.steps:
        cfg['training']['total_timesteps'] = args.steps
    if args.seed:
        cfg['project']['seed'] = args.seed
    if args.tag:
        cfg['project']['checkpoint_dir'] = f"checkpoints/{args.tag}"
        cfg['project']['log_dir'] = f"logs/{args.tag}"

    # Print header
    console.print(Panel.fit(
        "[bold cyan]VehicleFormer PhD Research[/bold cyan]\n"
        "[white]Graph-Augmented Causal Transformer for ICV Networks[/white]\n\n"
        f"[yellow]Config:[/yellow] {args.config}\n"
        f"[yellow]LLM Prior:[/yellow] {'[green]Enabled[/green]' if cfg['llm_prior']['enabled'] else '[red]Disabled[/red]'}\n"
        f"[yellow]Total Steps:[/yellow] {cfg['training']['total_timesteps']:,}",
        title="🚗 VehicleFormer",
        border_style="cyan",
    ))

    # Build SUMO scenario if not exists
    sumo_cfg_path = Path(cfg['simulation']['sumo_cfg'])
    if not sumo_cfg_path.exists():
        console.print("[yellow]Creating SUMO scenario...[/yellow]")
        import scripts.create_sumo_scenario as creator
        creator.create_network()
        creator.create_routes()
        creator.create_config()

    # Start training
    from vehicleformer.training.trainer import VehicleFormerTrainer
    trainer = VehicleFormerTrainer(cfg)

    # Resume from checkpoint if requested
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
