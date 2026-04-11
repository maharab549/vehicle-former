# VehicleFormer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#core-method)
[![SUMO](https://img.shields.io/badge/Simulator-SUMO-success.svg)](#installation)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](#project-status)

VehicleFormer is a research framework for robust communication orchestration in
Intelligent Connected Vehicle (ICV) systems. It combines graph-based state
representation, reinforcement learning control, and robustness-oriented
evaluation across heterogeneous networks (5G, C-V2X, satellite) in SUMO.

## Table of Contents

- [Overview](#overview)
- [Core Method](#core-method)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiment Workflows](#experiment-workflows)
- [Results Snapshot](#results-snapshot)
- [Reproducibility](#reproducibility)
- [Project Status](#project-status)
- [Citation](#citation)
- [License](#license)

## Overview

Connected vehicles face dynamic channel quality, variable infrastructure load,
handover events, and strict latency/reliability constraints. VehicleFormer is
designed to study these trade-offs in a reproducible, end-to-end setting.

Primary objectives:

- Learn adaptive network-selection and control policies
- Quantify robustness under outages and congestion bursts
- Support publication-ready workflows (baseline, ablation, stress evaluation)

## Core Method

VehicleFormer integrates the following modules:

- HetGNN encoder for heterogeneous vehicle-infrastructure state representation
- SAC policy optimization for continuous control actions
- Optional world-model auxiliary learning
- Optional LLM-prior KL regularization during training
- Hierarchical option-conditioned SAC actor for macro/micro control coupling
- Prioritized replay with importance sampling and TD-error feedback updates
- Distributional quantile critics with CVaR-aware risk-sensitive policy objective
- Adaptive Lagrangian risk constraints for online safety-risk balancing
- RND intrinsic motivation for deep exploration in sparse and shifted regimes
- Adaptive curriculum domain randomization driven by KPI performance
- Policy smoothness regularization against embedding perturbations
- Domain randomization and invariance regularization for OOD robustness
- Stress evaluation harness for scenario-level robustness analysis

## Repository Structure

```text
configs/
  default.yaml              # baseline configuration
  paper_robust.yaml         # robustness-focused paper configuration

data/sumo/                  # SUMO network/route/config files

scripts/
  create_sumo_scenario.py   # SUMO scenario generation
  verify_install.py         # environment and dependency checks
  run_ablations.py          # ablation batch runner
  resume_ablations.py       # continue interrupted ablations
  stress_eval.py            # robustness evaluation across stress scenarios
  diagnose.py               # diagnostics script

vehicleformer/
  env/                      # SUMO bridge and network environment logic
  models/                   # HetGNN, SAC, world model, LLM prior
  training/                 # trainer and replay buffer
  utils/                    # logging and KPI utilities

train.py                    # training entry point
requirements.txt
```

## Installation

Prerequisites:

- Python 3.10+
- SUMO installed and accessible from your runtime environment
- CUDA-capable GPU recommended for training

Setup:

```bash
pip install -r requirements.txt
python scripts/verify_install.py
```

If verification passes, proceed to training.

## Quick Start

```bash
# Baseline training
python train.py --config configs/default.yaml

# Robustness-focused training
python train.py --config configs/paper_robust.yaml
```

## Experiment Workflows

### Baseline + Ablations

```bash
python scripts/run_ablations.py
python scripts/resume_ablations.py
```

### Stress Robustness Evaluation

```bash
python scripts/stress_eval.py --episodes 20
```

### Diagnostics

```bash
python scripts/diagnose.py
```

## Results Snapshot

The following values are from completed local experiments and provided as a
quick reference for repository visitors.

| Experiment | Key Outcome |
|---|---|
| Full model (default) | Best eval reward: 799.053 |
| w/o HetGNN | Best eval reward: 793.944 |
| w/o World Model | Best eval reward: 795.097 |
| w/o LLM Prior | Best eval reward: 794.037 |
| Robust paper config | Best eval reward: 456.387 at 120K |

For complete metrics and scenario breakdowns, see generated outputs under
`logs/`.

## Reproducibility

- Hyperparameters are versioned in `configs/*.yaml`
- Experiment entry scripts are tracked under `scripts/`
- Runtime artifacts (`logs/`, `checkpoints/`, `venv/`) are excluded via `.gitignore`
- Fixed seeds are used via configuration files

## Project Status

Active research repository for model development, benchmarking, and paper
preparation.

## Citation

If this repository contributes to your research, please cite the associated
paper once published.

## License

Add a license file before broader external distribution.
