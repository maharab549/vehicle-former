# VehicleFormer

Graph-Augmented RL framework for robust communication orchestration in
Intelligent Connected Vehicle (ICV) systems.

## What This Project Does

VehicleFormer trains and evaluates policies that decide how vehicles use
multi-network connectivity (5G, C-V2X, satellite) under latency and reliability
constraints in a SUMO simulation environment.

Core stack:

- HetGNN for heterogeneous graph state encoding
- SAC for policy optimization
- Optional world-model auxiliary learning
- Optional LLM-prior regularization
- Stress-test benchmarking for robustness

## Highlights

- End-to-end train/eval pipeline for network selection + control actions
- Reproducible ablation workflows
- Robustness scenarios: outages, congestion bursts, sparse V2X coverage
- Paper-focused config included (`configs/paper_robust.yaml`)

## Project Layout

```text
configs/                 # experiment configs
data/sumo/               # SUMO scenario assets
scripts/                 # setup, ablation, stress-eval utilities
vehicleformer/env/       # SUMO bridge + network environment
vehicleformer/models/    # HetGNN, SAC, world model, LLM prior
vehicleformer/training/  # trainer + replay buffer
vehicleformer/utils/     # logging + metrics
train.py                 # entry point
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/verify_install.py
python train.py --config configs/default.yaml
```

## Common Commands

```bash
# Main training
python train.py --config configs/default.yaml

# Robust paper training
python train.py --config configs/paper_robust.yaml

# Ablations
python scripts/run_ablations.py
python scripts/resume_ablations.py

# Stress evaluation
python scripts/stress_eval.py --episodes 20
```

## Notes

- Runtime artifacts are excluded from git (`logs/`, `checkpoints/`, `venv/`).
- For publication experiments, prefer `configs/paper_robust.yaml`.

## Citation

If this repository helps your research, please cite the corresponding paper
when available.
