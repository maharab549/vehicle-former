# VehicleFormer

VehicleFormer is a research-grade framework for robust communication
orchestration in Intelligent Connected Vehicle (ICV) systems.

It combines graph-based state modeling, reinforcement learning control, and
robustness-oriented evaluation across heterogeneous networks (5G, C-V2X,
satellite) in SUMO simulation.

## Why VehicleFormer

Connected vehicles operate under dynamic channel quality, infrastructure load,
handover events, and strict latency/reliability constraints. VehicleFormer is
designed to study these trade-offs in a reproducible, end-to-end pipeline.

Main goals:

- Learn adaptive network-selection and control policies
- Evaluate robustness under outages and congestion bursts
- Produce publication-ready experiments (baseline, ablation, stress tests)

## Core Method

VehicleFormer integrates the following modules:

- HetGNN encoder for heterogeneous vehicle-infrastructure state representation
- SAC agent for continuous control and policy optimization
- Optional world-model auxiliary learning
- Optional LLM-prior KL regularization during training
- Stress-evaluation harness for scenario-level robustness analysis

## Repository Structure

```text
configs/
	default.yaml            # baseline configuration
	paper_robust.yaml       # robustness-focused paper configuration

data/sumo/                # SUMO network/route/config files

scripts/
	create_sumo_scenario.py # SUMO scenario generation
	verify_install.py       # environment and dependency checks
	run_ablations.py        # ablation batch runner
	resume_ablations.py     # continue interrupted ablations
	stress_eval.py          # robustness evaluation across stress scenarios
	diagnose.py             # diagnostics script

vehicleformer/
	env/                    # SUMO bridge and network environment logic
	models/                 # HetGNN, SAC, world model, LLM prior
	training/               # trainer and replay buffer
	utils/                  # logging and KPI utilities

train.py                  # training entry point
requirements.txt
```

## Installation

### Prerequisites

- Python 3.10+
- SUMO installed and available in the runtime environment
- CUDA-capable GPU recommended for training

### Setup

```bash
pip install -r requirements.txt
python scripts/verify_install.py
```

If verification passes, start training.

## Quick Start

```bash
# Baseline training
python train.py --config configs/default.yaml

# Robustness-focused training
python train.py --config configs/paper_robust.yaml
```

## Experiment Workflows

### 1) Baseline + Ablations

```bash
python scripts/run_ablations.py
python scripts/resume_ablations.py
```

### 2) Stress Robustness Evaluation

```bash
python scripts/stress_eval.py --episodes 20
```

### 3) Diagnostics

```bash
python scripts/diagnose.py
```

## Results Snapshot (Current Runs)

The following values are from completed local runs and are intended as a quick
reference for repository visitors:

| Experiment | Key Outcome |
|---|---|
| Full model (default) | Best eval reward: 799.053 |
| w/o HetGNN | Best eval reward: 793.944 |
| w/o World Model | Best eval reward: 795.097 |
| w/o LLM Prior | Best eval reward: 794.037 |
| Robust paper config | Best eval reward: 456.387 at 120K |

For full metrics and scenario breakdown, see generated files under `logs/`.

## Reproducibility

- Hyperparameters are versioned in `configs/*.yaml`
- Major experiment scripts are under `scripts/`
- Runtime artifacts (`logs/`, `checkpoints/`, `venv/`) are ignored by git
- Use fixed seeds from config for repeatability

## Development Notes

- Main training logic: `vehicleformer/training/trainer.py`
- Environment dynamics: `vehicleformer/env/icv_env.py`, `vehicleformer/env/network_sim.py`
- Logging outputs: run-specific folders under `logs/`

## Citation

If this repository contributes to your research, please cite the associated
paper once published.

## License

Add your license file before broader distribution.
