#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo
cd "$(dirname "$0")/.."

base="checkpoints/official-navigation-fixed/final_model.zip"
opponent="checkpoints/official-mixed-adaptive-50k/best_model.zip"
output_dir="checkpoints/official-navigation-combat"
mkdir -p "$output_dir" results/logs
timestamp="$(date +%Y%m%d_%H%M%S)"

# Convert the broad stochastic navigation policy into a combat policy without
# returning to the high-entropy exploration-only objective.
PYTHONUNBUFFERED=1 python -m src.train_ppo \
  --load-model "$base" \
  --action-set all \
  --opponent mixed \
  --opponent-model "$opponent" \
  --opponent-action-set fire \
  --opponent-prefix-steps 0 \
  --opponent-device cpu \
  --opponent-stochastic \
  --reward-shaping \
  --tactical-pretraining \
  --shaping-scale 0.01 \
  --timesteps 200000 \
  --n-envs 6 \
  --vec-env subproc \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00001 \
  --gamma 0.995 \
  --clip-range 0.1 \
  --entropy-coefficient 0.01 \
  --checkpoint-frequency 50000 \
  --checkpoint-directory "$output_dir" \
  --output "$output_dir/final_model" \
  --run-name official_navigation_combat \
  --seed 200835 \
  --device cuda \
  2>&1 | tee "results/logs/navigation_combat_${timestamp}.log"

echo "Navigation-combat training finished: $output_dir/final_model.zip"
