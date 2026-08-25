#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo
cd "$(dirname "$0")/.."

mkdir -p checkpoints/official-combat-tactics results/logs
timestamp="$(date +%Y%m%d_%H%M%S)"
teacher="checkpoints/official-mixed-adaptive-50k/best_model.zip"

PYTHONUNBUFFERED=1 python -m src.train_ppo \
  --load-model checkpoints/official-maze-exploration/final_model.zip \
  --action-set all \
  --opponent mixed \
  --opponent-model "$teacher" \
  --opponent-action-set fire \
  --opponent-prefix-steps 0 \
  --opponent-device cpu \
  --opponent-stochastic \
  --reward-shaping \
  --tactical-pretraining \
  --shaping-scale 0.003 \
  --timesteps 500000 \
  --n-envs 6 \
  --vec-env subproc \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00003 \
  --gamma 0.995 \
  --clip-range 0.1 \
  --entropy-coefficient 0.015 \
  --checkpoint-frequency 50000 \
  --checkpoint-directory checkpoints/official-combat-tactics \
  --output checkpoints/official-combat-tactics/final_model \
  --run-name official_combat_tactics \
  --seed 200827 \
  --device cuda \
  2>&1 | tee "results/logs/combat_tactics_${timestamp}.log"
