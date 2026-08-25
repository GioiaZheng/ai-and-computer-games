#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo
cd "$(dirname "$0")/.."

base="checkpoints/official-all18-hit-100k/final_model.zip"
selfplay_teacher="checkpoints/official-mixed-adaptive-50k/best_model.zip"
output_dir="checkpoints/official-navigation-fixed"
mkdir -p "$output_dir" results/logs
timestamp="$(date +%Y%m%d_%H%M%S)"

# Fail early if the closed-loop teacher cannot pass both maze gates.
python -m pytest tests/test_environment.py -q
python -m src.benchmark_waypoint_coverage --seed 200831 --minimum-waypoints 4

# Retain the existing all-action policy and teach it visually grounded routes
# through both the upper and lower passages for both player roles.
PYTHONUNBUFFERED=1 python -m src.pretrain_bc \
  --load-model "$base" \
  --expert waypoint \
  --action-set all \
  --opponent random \
  --episodes-per-role 8 \
  --steps-per-episode 6000 \
  --samples-per-action 2048 \
  --epochs 8 \
  --batch-size 256 \
  --learning-rate 0.00001 \
  --entropy-coefficient 0.003 \
  --seed 200832 \
  --device cuda \
  --output "$output_dir/navigation_bc" \
  2>&1 | tee "results/logs/navigation_bc_${timestamp}.log"

# Continue the user's policy against a mixture that includes a frozen copy of
# its strongest retained scorer. Official rules and all 18 actions are kept.
PYTHONUNBUFFERED=1 python -m src.train_ppo \
  --load-model "$output_dir/navigation_bc.zip" \
  --action-set all \
  --opponent mixed \
  --opponent-model "$selfplay_teacher" \
  --opponent-action-set fire \
  --opponent-prefix-steps 0 \
  --opponent-device cpu \
  --opponent-stochastic \
  --reward-shaping \
  --exploration-pretraining \
  --shaping-scale 0.015 \
  --timesteps 600000 \
  --n-envs 6 \
  --vec-env subproc \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00003 \
  --gamma 0.995 \
  --clip-range 0.1 \
  --entropy-coefficient 0.025 \
  --checkpoint-frequency 50000 \
  --checkpoint-directory "$output_dir" \
  --output "$output_dir/final_model" \
  --run-name official_navigation_fixed \
  --seed 200833 \
  --device cuda \
  2>&1 | tee "results/logs/navigation_rl_${timestamp}.log"

echo "Navigation training finished: $output_dir/final_model.zip"
