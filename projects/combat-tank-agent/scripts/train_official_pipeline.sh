#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo

python -m src.official_environment
python -m src.train_official_ppo \
  --timesteps 500000 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --entropy-coefficient 0.02 \
  --checkpoint-frequency 100000 \
  --checkpoint-directory checkpoints/official-pipeline-random \
  --output checkpoints/official-pipeline-random/final_model \
  --run-name official_pipeline_random \
  --seed 82026 \
  --device cuda
