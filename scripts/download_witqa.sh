#!/usr/bin/env bash
# WiTQA ships as a single TSV per the original release. Either fetch from the
# upstream URL into ``data/witqa/witqa.tsv``, or point WITQA_TSV at your own
# copy. The runner only needs the TSV path; no HF download is required.
set -euo pipefail
mkdir -p data/witqa
DEFAULT_TSV="data/witqa/witqa.tsv"
TARGET="${WITQA_TSV:-$DEFAULT_TSV}"
if [ ! -f "$TARGET" ]; then
    echo "WiTQA TSV not found at $TARGET."
    echo "Download the file from the WiTQA paper release and place it at $TARGET,"
    echo "or export WITQA_TSV=<your path> and rerun."
    exit 1
fi
echo "WiTQA available at $TARGET."
