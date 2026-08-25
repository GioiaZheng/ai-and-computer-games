#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo

mkdir -p checkpoints/official-sweep-ppo-overnight results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/overnight_${STAMP}.log"
START_MODEL="${1:-}"

echo "Offline Combat Tank PPO training"
echo "start_model=${START_MODEL:-fresh policy}"
echo "log=$LOG"
LOAD_ARGS=()
if [[ -n "$START_MODEL" ]]; then
  LOAD_ARGS=(--load-model "$START_MODEL")
fi
python -u -m src.train_ppo \
  "${LOAD_ARGS[@]}" \
  --timesteps 10000000 \
  --n-envs 6 \
  --vec-env subproc \
  --start-method spawn \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 2 \
  --learning-rate 0.00005 \
  --entropy-coefficient 0.001 \
  --action-set sweep \
  --scripted-prefix-steps 2024 \
  --checkpoint-frequency 250000 \
  --checkpoint-directory checkpoints/official-sweep-ppo-overnight \
  --output checkpoints/official-sweep-ppo-overnight/final_model \
  --run-name official_sweep_ppo_overnight \
  --seed 21101 \
  --device cuda 2>&1 | tee "$LOG"

echo "Training complete. Final model: checkpoints/official-sweep-ppo-overnight/final_model.zip"
