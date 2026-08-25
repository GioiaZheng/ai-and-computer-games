#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo
cd "$(dirname "$0")/.."

base="checkpoints/official-mixed-adaptive-50k/best_model.zip"
output_dir="checkpoints/official-fire-score-continuation"
mkdir -p "$output_dir" results/logs
timestamp="$(date +%Y%m%d_%H%M%S)"

# Continue the strongest retained nine-action fire policy. Official hit rewards
# remain dominant; shaping only discourages idling and repeated blind actions.
PYTHONUNBUFFERED=1 python -m src.train_ppo \
  --load-model "$base" \
  --action-set fire \
  --opponent mixed \
  --opponent-model "$base" \
  --opponent-action-set fire \
  --opponent-prefix-steps 0 \
  --opponent-device cpu \
  --opponent-stochastic \
  --reward-shaping \
  --tactical-pretraining \
  --shaping-scale 0.0015 \
  --timesteps 1000000 \
  --n-envs 6 \
  --vec-env subproc \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00002 \
  --gamma 0.995 \
  --clip-range 0.1 \
  --entropy-coefficient 0.01 \
  --checkpoint-frequency 50000 \
  --checkpoint-directory "$output_dir" \
  --output "$output_dir/final_model" \
  --run-name official_fire_score \
  --seed 200828 \
  --device cuda \
  2>&1 | tee "results/logs/fire_score_${timestamp}.log"

echo "Score continuation finished: $output_dir/final_model.zip"
