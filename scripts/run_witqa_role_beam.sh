#!/usr/bin/env bash
# Paired role vs no_telemetry evaluation on WiTQA at N=300.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
WITQA_TSV="${WITQA_TSV:-data/witqa/witqa.tsv}"
INDEX_DIR="${INDEX_DIR:-data/witqa_self_bm25_index}"
RESULTS_ROOT="${RESULTS_ROOT:-results/witqa_role_beam}"
N_LIMIT="${N_LIMIT:-300}"
SEED="${SEED:-42}"

mkdir -p "$RESULTS_ROOT"

for telemetry_mode in role no_telemetry; do
    out_jsonl="$RESULTS_ROOT/${telemetry_mode}_s${SEED}.jsonl"
    out_summary="$RESULTS_ROOT/${telemetry_mode}_s${SEED}.summary.json"
    echo "=== arm: $telemetry_mode ==="
    python -m telemetry_agent.runners.witqa_role_beam \
        --witqa_tsv "$WITQA_TSV" \
        --seed "$SEED" \
        --limit "$N_LIMIT" \
        --index_dir "$INDEX_DIR" \
        --model_name "$MODEL" \
        --qwen_backend vllm \
        --agent_telemetry_mode "$telemetry_mode" \
        --output_jsonl "$out_jsonl" \
        --summary_json "$out_summary"
done
echo "Done. Results in $RESULTS_ROOT."
