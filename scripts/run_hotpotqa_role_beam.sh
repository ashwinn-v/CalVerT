#!/usr/bin/env bash
# Paired role vs no_telemetry evaluation on HotpotQA-distractor at N=300 dev.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_role_beam}"
INDEX_DIR="${INDEX_DIR:-data/hotpotqa_distractor_validation_bm25_index}"
N_LIMIT="${N_LIMIT:-300}"
SEED="${SEED:-42}"

mkdir -p "$RESULTS_ROOT"

for telemetry_mode in role no_telemetry; do
    out_jsonl="$RESULTS_ROOT/${telemetry_mode}_s${SEED}.jsonl"
    out_summary="$RESULTS_ROOT/${telemetry_mode}_s${SEED}.summary.json"
    echo "=== arm: $telemetry_mode ==="
    python -m telemetry_agent.runners.hotpotqa_role_beam \
        --dataset_subset distractor \
        --split validation \
        --seed "$SEED" \
        --limit "$N_LIMIT" \
        --index_dir "$INDEX_DIR" \
        --model_name "$MODEL" \
        --qwen_backend vllm \
        --agent_prompt_mode stateless \
        --agent_telemetry_mode "$telemetry_mode" \
        --agent_max_turns 6 \
        --agent_max_new_tokens 4096 \
        --use_guided_json \
        --output_jsonl "$out_jsonl" \
        --summary_json "$out_summary"
done
echo "Done. Results in $RESULTS_ROOT."
