#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo

python -m src.official_environment
python -m src.train_official_ppo \
  --load-model checkpoints/official-pipeline-exploration/official_pipeline_exploration_200000_steps.zip \
  --timesteps 100000 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00005 \
  --entropy-coefficient 0.01 \
  --exploration-bonus-scale 0.0 \
  --idle-penalty-scale 0.0 \
  --checkpoint-frequency 50000 \
  --checkpoint-directory checkpoints/official-pipeline-score-finetune \
  --output checkpoints/official-pipeline-score-finetune/final_model \
  --run-name official_pipeline_score_finetune \
  --seed 84026 \
  --device cuda
