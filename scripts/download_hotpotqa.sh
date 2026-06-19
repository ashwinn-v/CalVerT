#!/usr/bin/env bash
# Pull HotpotQA-distractor (validation + train) from HuggingFace Hub. The
# datasets library caches under $HF_HOME so subsequent runs are no-ops.
set -euo pipefail
python -c "from datasets import load_dataset; load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation'); load_dataset('hotpotqa/hotpot_qa', 'distractor', split='train')"
echo "HotpotQA-distractor cached."
