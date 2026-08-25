#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
base="checkpoints/official-pipeline-selfplay/official_pipeline_selfplay_250000_steps.zip"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="logs/final_candidate_selection_${stamp}"

cd "$repo"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate boxing-ppo
mkdir -p "$output_dir"

evaluate_candidate() {
    local label="$1"
    local role="$2"
    local model="$3"
    local opponent="$4"
    local log="$output_dir/${role}_${label}_${opponent}.log"

    printf '\n===== %s role=%s opponent=%s =====\n' "$label" "$role" "$opponent" | tee "$log"
    args=(
        --model "$model"
        --role "$role"
        --games-per-role 5
        --max-steps 8000
        --seed 310826
        --device cuda
    )
    if [[ "$opponent" == "baseline" ]]; then
        args+=(--opponent-model "$base" --opponent-device cpu)
    fi
    python -m src.evaluate_official_ppo "${args[@]}" | tee -a "$log"
}

first_candidates=(
    "base|$base"
    "exploration_champion|checkpoints/official-pipeline-exploration/official_pipeline_exploration_200000_steps.zip"
    "overnight_diverse|checkpoints/overnight-balanced-diverse/final_model.zip"
    "overnight_first|checkpoints/overnight-first-role/final_model.zip"
)

second_candidates=(
    "base|$base"
    "old_second_champion|checkpoints/official-pipeline-second-role/official_pipeline_second_role_275000_steps.zip"
    "overnight_exploit|checkpoints/overnight-balanced-exploit/final_model.zip"
)

for entry in "${first_candidates[@]}"; do
    IFS='|' read -r label model <<< "$entry"
    evaluate_candidate "$label" "first_0" "$model" random
    evaluate_candidate "$label" "first_0" "$model" baseline
done

for entry in "${second_candidates[@]}"; do
    IFS='|' read -r label model <<< "$entry"
    evaluate_candidate "$label" "second_0" "$model" random
    evaluate_candidate "$label" "second_0" "$model" baseline
done

printf '%s\n' "$output_dir" > logs/final_candidate_selection_last_run.txt
printf '\nFinal candidate screening finished. Logs: %s\n' "$output_dir"
