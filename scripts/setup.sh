#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# VehicleFormer - WSL2 Environment Setup Script
# Run: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ═══════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         VehicleFormer - Environment Setup            ║"
echo "║      PhD Research: ICV Multi-Network AI System       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── Step 1: System packages ───────────────────────────────────────────
echo -e "${YELLOW}[1/6] Installing system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    cmake \
    libopenmpi-dev
echo -e "${GREEN}✓ System packages installed${NC}"

# ─── Step 2: Python virtual environment ────────────────────────────────
echo -e "${YELLOW}[2/6] Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
source venv/bin/activate
pip install --upgrade pip -q
echo -e "${GREEN}✓ pip upgraded${NC}"

# ─── Step 3: PyTorch Geometric dependencies ────────────────────────────
echo -e "${YELLOW}[3/6] Installing PyTorch Geometric...${NC}"
# Detect CUDA version
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "cpu")
echo "    Detected CUDA: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "12"* ]]; then
    TORCH_CUDA="cu121"
elif [[ "$CUDA_VERSION" == "11"* ]]; then
    TORCH_CUDA="cu118"
else
    TORCH_CUDA="cpu"
fi

pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "https://data.pyg.org/whl/torch-2.2.0+${TORCH_CUDA}.html" -q
echo -e "${GREEN}✓ PyTorch Geometric installed${NC}"

# ─── Step 4: Main requirements ─────────────────────────────────────────
echo -e "${YELLOW}[4/6] Installing Python requirements...${NC}"
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Python requirements installed${NC}"

# ─── Step 5: SUMO Python bindings check ────────────────────────────────
echo -e "${YELLOW}[5/6] Checking SUMO installation...${NC}"
if command -v sumo &> /dev/null; then
    SUMO_VERSION=$(sumo --version 2>&1 | head -1)
    echo -e "${GREEN}✓ SUMO found: $SUMO_VERSION${NC}"
    # Set SUMO_HOME if not already set
    if [ -z "$SUMO_HOME" ]; then
        SUMO_PATH=$(which sumo)
        export SUMO_HOME=$(dirname $(dirname $SUMO_PATH))
        echo "export SUMO_HOME=$SUMO_HOME" >> ~/.bashrc
        echo "    SUMO_HOME set to: $SUMO_HOME"
    fi
else
    echo -e "${YELLOW}⚠ SUMO not found in PATH. Installing...${NC}"
    sudo apt-get install -y -qq sumo sumo-tools sumo-doc
    export SUMO_HOME=/usr/share/sumo
    echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc
fi

# Add SUMO_HOME to venv activate
echo "export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}" >> venv/bin/activate

# ─── Step 6: Create SUMO scenario files ────────────────────────────────
echo -e "${YELLOW}[6/6] Creating SUMO simulation scenario...${NC}"
python3 scripts/create_sumo_scenario.py
echo -e "${GREEN}✓ SUMO scenario created in data/sumo/${NC}"

# ─── Done ──────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║            Setup Complete! ✓                         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python scripts/verify_install.py"
echo "  3. python train.py --config configs/default.yaml"
echo ""
