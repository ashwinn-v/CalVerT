# Tinker GRPO training for the role-beam agent

Trains a rank-16 LoRA adapter on the per-turn agent of the role-beam loop,
using a Search-R1-derived reward and Tinker's PPO loss function.

## Setup

```bash
# 0. Install with the GRPO extras.
pip install -e ".[grpo]"
export TELEMETRY_AGENT_ROOT="$PWD"

# 1. Build a training parquet from the public HotpotQA dev split.
python -m grpo.build_train_data \
  --hotpotqa_split train \
  --n_hard 5000 --n_med 2500 --n_easy 2500 \
  --output data/train.parquet

# 2. (Optional) Build the SFT cold-start corpus from your own trajectory
# release. Set HF_HOTPOT_TRAJECTORIES to the HF dataset id holding paired
# role-beam trajectory rows.
export HF_HOTPOT_TRAJECTORIES="${ORG}/${REPO}"
python -m grpo.build_sft_corpus \
  --hf_hotpot "$HF_HOTPOT_TRAJECTORIES" \
  --top_quartile_only \
  --output data/sft_corpus.parquet

# 3. (Optional) SFT cold-start with TRL.
python -m grpo.sft_train \
  --sft_corpus data/sft_corpus.parquet \
  --base_model Qwen/Qwen3-8B \
  --output_dir runs/sft

# 4. GRPO training on Tinker.
python -m grpo.tinker_train \
  --train_parquet data/train.parquet \
  --output_dir runs/grpo \
  --base_model Qwen/Qwen3-8B \
  --rank 16 \
  --steps 200 \
  --rollouts_per_prompt 4 \
  --prompts_per_batch 10
```

## Environment variables

| Variable | Purpose |
|---|---|
| `TINKER_API_KEY` | Tinker SDK authentication |
| `WANDB_API_KEY` | Optional; enables wandb logging when present |
| `WANDB_PROJECT` | Override the default wandb project name (`telemetry-agent`) |
| `HF_TOKEN` | Required when the base model is gated ||
| `HF_HOTPOT_TRAJECTORIES` | HF dataset id for the SFT cold-start trajectories |

## Reward

Reward in ``grpo.reward`` is

```
r(tau) = alpha * F1(a_pred, a_gold) - beta * min(total_lm_calls / budget, 1)
```

with `alpha = 2.0`, `beta = 0.15`, `budget = 9`. A verifier-gated F1-substring
cap (`f1_substring_cap = 0.3`) prevents an answer-padding attack: when the
predicted answer is a strict substring of gold (or vice versa), EM is zero,
and MiniCheck grounding is below 0.5, F1 is clipped to the cap. The reward
takes a ``Rollout`` and returns a scalar; see ``grpo/tinker_train.py``.


