#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pommerman
cd "$PROJECT_DIR"

python -m src.train_bc \
    --episodes 100 \
    --device cuda \
    --checkpoint checkpoints/bc_policy.pt

python -m src.train_ppo \
    --episodes 1000 \
    --opponent mixed \
    --device cuda \
    --load checkpoints/bc_policy.pt \
    --checkpoint checkpoints/ppo_policy.pt
