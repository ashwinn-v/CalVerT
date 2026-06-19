#!/usr/bin/env bash
# Build the WiTQA self-BM25 index from the TSV.
set -euo pipefail
WITQA_TSV="${WITQA_TSV:-data/witqa/witqa.tsv}"
mkdir -p data
python -m telemetry_agent.retrieval.build_index \
    --witqa_tsv "$WITQA_TSV" \
    --output_dir data/witqa_self_bm25_index
echo "WiTQA BM25 index built."
