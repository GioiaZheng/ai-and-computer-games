#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
base="checkpoints/official-pipeline-selfplay/official_pipeline_selfplay_250000_steps.zip"
first_source="checkpoints/official-pipeline-exploration/official_pipeline_exploration_200000_steps.zip"
second_source="checkpoints/official-pipeline-second-role/official_pipeline_second_role_275000_steps.zip"

cd "$repo"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo

python -m pytest -q
python -m src.official_environment

# Preserve combat ability while making unexplored maze cells valuable again.
# These rewards exist only in the training adapter; tournament rewards and the
# instructor-provided observation pipeline remain unchanged.
python -m src.train_official_ppo \
  --load-model "$first_source" \
  --timesteps 120000 \
  --n-envs 8 \
  --fixed-role first_0 \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00001 \
  --entropy-coefficient 0.025 \
  --exploration-bonus-scale 0.08 \
  --idle-penalty-scale 0.004 \
  --opponent mixed \
  --opponent-model "$base" \
  --opponent-model-probability 0.50 \
  --opponent-device cpu \
  --checkpoint-frequency 40000 \
  --checkpoint-directory checkpoints/official-navigation-preserve-first \
  --output checkpoints/official-navigation-preserve-first/final_model \
  --run-name official_navigation_first \
  --seed 91026 \
  --device cuda

python -m src.train_official_ppo \
  --load-model "$second_source" \
  --timesteps 120000 \
  --n-envs 8 \
  --fixed-role second_0 \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00001 \
  --entropy-coefficient 0.025 \
  --exploration-bonus-scale 0.08 \
  --idle-penalty-scale 0.004 \
  --opponent mixed \
  --opponent-model "$base" \
  --opponent-model-probability 0.50 \
  --opponent-device cpu \
  --checkpoint-frequency 40000 \
  --checkpoint-directory checkpoints/official-navigation-preserve-second \
  --output checkpoints/official-navigation-preserve-second/final_model \
  --run-name official_navigation_second \
  --seed 92026 \
  --device cuda
