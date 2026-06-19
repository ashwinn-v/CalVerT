#!/usr/bin/env bash
# Build a BM25 index over the HotpotQA-distractor per-question paragraph pool.
# The result lands under data/hotpotqa_distractor_validation_bm25_index/.
set -euo pipefail
mkdir -p data
python -m telemetry_agent.retrieval.build_index \
    --dataset hotpotqa/hotpot_qa \
    --config distractor \
    --split validation \
    --output_dir data/hotpotqa_distractor_validation_bm25_index
echo "HotpotQA BM25 index built."
