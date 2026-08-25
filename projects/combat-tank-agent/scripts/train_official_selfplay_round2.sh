#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo

BEST_MODEL=checkpoints/official-pipeline-selfplay/official_pipeline_selfplay_250000_steps.zip

python -m src.official_environment
python -m src.train_official_ppo \
  --load-model "$BEST_MODEL" \
  --timesteps 50000 \
  --n-envs 4 \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.000025 \
  --entropy-coefficient 0.02 \
  --exploration-bonus-scale 0.0 \
  --idle-penalty-scale 0.0 \
  --opponent mixed \
  --opponent-model "$BEST_MODEL" \
  --opponent-model-probability 0.80 \
  --opponent-device cpu \
  --checkpoint-frequency 25000 \
  --checkpoint-directory checkpoints/official-pipeline-selfplay-round2 \
  --output checkpoints/official-pipeline-selfplay-round2/final_model \
  --run-name official_pipeline_selfplay_round2 \
  --seed 86026 \
  --device cuda
