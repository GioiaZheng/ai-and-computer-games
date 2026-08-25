#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo

python -m src.official_environment
python -m src.train_official_ppo \
  --load-model checkpoints/official-pipeline-random/official_pipeline_random_100000_steps.zip \
  --timesteps 300000 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.0001 \
  --entropy-coefficient 0.02 \
  --exploration-bonus-scale 0.03 \
  --idle-penalty-scale 0.002 \
  --checkpoint-frequency 100000 \
  --checkpoint-directory checkpoints/official-pipeline-exploration \
  --output checkpoints/official-pipeline-exploration/final_model \
  --run-name official_pipeline_exploration \
  --seed 83026 \
  --device cuda
