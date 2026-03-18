# VehicleFormer

Graph-Augmented Causal Transformer with world-model-assisted reinforcement
learning for multi-network Intelligent Connected Vehicle (ICV) systems.

## Overview

VehicleFormer is an end-to-end research codebase for training and evaluating
network orchestration policies across 5G, C-V2X, and satellite links in SUMO.
The framework combines:

- Heterogeneous graph encoding (HetGNN)
- Soft Actor-Critic (SAC) control
- World model auxiliary learning
- Optional LLM prior regularization
- Stress-test evaluation scenarios for robustness analysis

## Project Structure

```
vehicleformer/
  configs/
    default.yaml
    paper_robust.yaml
  data/sumo/
  scripts/
    create_sumo_scenario.py
    verify_install.py
    run_ablations.py
    resume_ablations.py
    stress_eval.py
  vehicleformer/
    env/
    models/
    training/
    utils/
  train.py
  requirements.txt
```

## Quick Start

```bash
# 1) Create and activate your Python environment

# 2) Install dependencies
pip install -r requirements.txt

# 3) Verify setup
python scripts/verify_install.py

# 4) Train default configuration
python train.py --config configs/default.yaml
```

## Reproducibility Commands

```bash
# Ablations
python scripts/run_ablations.py
python scripts/resume_ablations.py

# Stress evaluation
python scripts/stress_eval.py --episodes 20

# Robust paper-focused run
python train.py --config configs/paper_robust.yaml
```

## Notes

- Large artifacts (logs, checkpoints, virtual environments) are ignored via
  `.gitignore`.
- Use `configs/paper_robust.yaml` for the final robustness-focused training
  protocol.
