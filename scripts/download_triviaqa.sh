#!/usr/bin/env bash
set -euo pipefail
python -c "from datasets import load_dataset; load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')"
echo "TriviaQA rc.nocontext cached."
