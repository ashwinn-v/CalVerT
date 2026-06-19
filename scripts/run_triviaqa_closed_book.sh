#!/usr/bin/env bash
# Closed-book DINCO on TriviaQA rc.nocontext. The paper's calibration
# appendix uses Qwen3-32B with K=5 beams and N_SC=5 stochastic samples on a
# random N=300 sample (seed=42).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
RESULTS_ROOT="${RESULTS_ROOT:-results/triviaqa_closed_book}"
N_LIMIT="${N_LIMIT:-300}"
SEED="${SEED:-42}"
K="${K:-5}"
N_SC="${N_SC:-5}"

mkdir -p "$RESULTS_ROOT"
out_jsonl="$RESULTS_ROOT/dinco_s${SEED}.jsonl"
out_summary="$RESULTS_ROOT/dinco_s${SEED}.summary.json"

python -m telemetry_agent.runners.triviaqa_closed_book \
    --model_name "$MODEL" \
    --split validation \
    --seed "$SEED" \
    --limit "$N_LIMIT" \
    --num_beams "$K" \
    --n_sample "$N_SC" \
    --output_jsonl "$out_jsonl" \
    --summary_json "$out_summary"
echo "Done. Results in $RESULTS_ROOT."
