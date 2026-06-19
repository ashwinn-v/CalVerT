# CalVerT

Code for the paper **"CalVerT: Augmenting Agents with Calibrated Verifier Telemetry Improves Action and Learning in Knowledge-Intensive Tasks"**


## Install and Setup

```bash
git clone https://github.com/ashwinn-v/CalVerT.git
cd CalVerT
python -m venv .venv && source .venv/bin/activate
pip install -e ".[grpo,test]"
export TELEMETRY_AGENT_ROOT="$PWD"
```

The eval runners assume vLLM compatible GPUs. CPU-only smoke runs work for
the tests in `tests/` but not for full DINCO + MiniCheck inference.

## CalVerT on HotpotQA-distractor

```bash
# 1. Pull HotpotQA distractor split and build the per-question BM25 index.
bash scripts/download_hotpotqa.sh
bash scripts/build_hotpotqa_index.sh

Change dataset name to evaluate on 2Wiki (2WikiMultihopQA) and MuSiQue


# 2. Paired role vs no_telemetry evaluation on Qwen3-32B at N=300 dev.
bash scripts/run_hotpotqa_role_beam.sh
```

## CalVerT on WiTQA

```bash
bash scripts/download_witqa.sh
bash scripts/build_witqa_index.sh
bash scripts/run_witqa_role_beam.sh
```

## closed-book DINCO on TriviaQA

```bash
bash scripts/download_triviaqa.sh
bash scripts/run_triviaqa_closed_book.sh
```

## GRPO training

See `grpo/README.md` for the Tinker SDK path (rank-16 LoRA on Qwen3-8B or
Qwen3-30B-A3B MoE).

## Environment variables

| Variable | When needed |
|---|---|
| `HF_TOKEN` | Gated models (e.g., MiniCheck weights); reading datasets |
| `TINKER_API_KEY` | GRPO training via Tinker SDK |
| `WANDB_API_KEY` | Optional; enables wandb logging during training |
| `CLUSTER_NAME` | Optional label written into result JSONLs for when evaluating on clusters |
| `ENV_VARS` | All other API keys and VARS setup |

## Models


- **Generator:** Qwen3-32B, Qwen3-8B; Mistral-Small-24B-Instruct; any
  vLLM-compatible chat model with HF chat templates is compatible for eval.
- **Grounding verifier:** `https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B`.
- **NLI for DINCO:** `https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

## License

MIT. See `LICENSE`.


