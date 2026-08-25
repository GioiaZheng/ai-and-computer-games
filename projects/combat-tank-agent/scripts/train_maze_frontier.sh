#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo
cd "$(dirname "$0")/.."

mkdir -p checkpoints/official-maze-frontier results/logs
timestamp="$(date +%Y%m%d_%H%M%S)"

PYTHONUNBUFFERED=1 python -m src.train_ppo \
  --load-model checkpoints/official-all18-hit-100k/final_model.zip \
  --action-set all \
  --opponent random \
  --reward-shaping \
  --exploration-pretraining \
  --shaping-scale 0.02 \
  --timesteps 600000 \
  --n-envs 6 \
  --vec-env subproc \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00005 \
  --gamma 0.995 \
  --clip-range 0.1 \
  --entropy-coefficient 0.025 \
  --checkpoint-frequency 50000 \
  --checkpoint-directory checkpoints/official-maze-frontier \
  --output checkpoints/official-maze-frontier/final_model \
  --run-name official_maze_frontier \
  --seed 200829 \
  --device cuda \
  2>&1 | tee "results/logs/maze_frontier_${timestamp}.log"
