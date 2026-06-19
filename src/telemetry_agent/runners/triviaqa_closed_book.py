#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle as pkl
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers

from telemetry_agent._common import (
    BASE_DIR,
    file_stem_for_subset,
    load_hf_dataset_split,
    limit_suffix,
    maybe_select_subset,
    normalize_limit,
    slugify_name,
    write_json,
)

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Missing dependency 'datasets'. Install via `pip install -e .` in this repo."
    ) from exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Missing dependency 'transformers'. Install via `pip install -e .` in this repo."
    ) from exc

from telemetry_agent.dinco import triviaqa as dinco_base
from telemetry_agent.dinco import _gpt_oss_triviaqa as gpt_oss_dinco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-book DINCO on TriviaQA.")
    parser.add_argument("--dataset_name", type=str, default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset_config", type=str, default="rc.nocontext")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local path created with datasets.save_to_disk(). Overrides remote dataset loading.",
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=1000, help="<= 0 means full split")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle before selecting the subset.",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--n_sample", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--sc_match_threshold", type=float, default=0.9)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to results/closed_book/<run_id>",
    )
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace, limit: Optional[int]) -> Path:
    subset_stem = file_stem_for_subset(args.dataset_config, args.split, args.seed, limit)
    run_id = f"{subset_stem}_{slugify_name(args.model_name.split('/')[-1])}"
    return BASE_DIR / "results" / "closed_book" / run_id


def load_split(args: argparse.Namespace, limit: Optional[int]):
    try:
        split = load_hf_dataset_split(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            dataset_path=args.dataset_path,
        )
    except Exception as exc:  # pragma: no cover - network/cache dependent
        source = args.dataset_path or f"{args.dataset_name}:{args.dataset_config}"
        raise RuntimeError(
            f"Failed to load TriviaQA split from {source}. If you are offline, pre-cache the dataset first."
        ) from exc
    return maybe_select_subset(split, limit=limit, shuffle=args.shuffle, seed=args.seed)


def validate_gpt_oss_runtime(model_name: str) -> None:
    issues: List[str] = []
    if not torch.cuda.is_available():
        issues.append("No CUDA device is visible in this shell.")

    try:
        from transformers.utils import is_accelerate_available, is_kernels_available, is_triton_available
    except Exception:  # pragma: no cover - transformers import path changed
        is_accelerate_available = None
        is_kernels_available = None
        is_triton_available = None

    if is_accelerate_available is not None and not is_accelerate_available():
        issues.append("Missing or outdated `accelerate` (required for `device_map=\"auto\"`).")
    if is_triton_available is not None and not is_triton_available("3.4.0"):
        issues.append("Missing or incompatible `triton` runtime for MXFP4 kernels.")
    if is_kernels_available is not None and not is_kernels_available():
        issues.append("Missing `kernels`, which GPT-OSS MXFP4 loading depends on.")
    elif importlib.util.find_spec("kernels") is None:
        issues.append("Missing `kernels`, which GPT-OSS MXFP4 loading depends on.")

    if not issues:
        return

    formatted = "\n".join(f"  - {issue}" for issue in issues)
    raise RuntimeError(
        f"Cannot load {model_name} in this environment.\n"
        f"{formatted}\n"
        "The Hugging Face GPT-OSS checkpoints store MoE weights in MXFP4. When the MXFP4 runtime is missing, "
        "Transformers falls back to dequantizing those weights to bf16 during `from_pretrained`, which usually "
        "causes CUDA OOM for `openai/gpt-oss-120b`.\n"
        "Install the GPT-OSS runtime dependencies first and rerun. The official model card documents at least "
        "`pip install -U transformers kernels torch`; with this script, `accelerate` also needs to be available."
    )


def load_generator_model(model_name: str):
    lower_name = model_name.lower()
    if gpt_oss_dinco.supports_model(model_name):
        validate_gpt_oss_runtime(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if gpt_oss_dinco.supports_model(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )
    elif "gemma" in lower_name:
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
    else:
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
        )

    if "llama" in lower_name and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if "gemma" in lower_name:
        tokenizer.eos_token_id = 106

    return model, tokenizer


def dinco_backend_for_model(model_name: str):
    if gpt_oss_dinco.supports_model(model_name):
        return gpt_oss_dinco
    return dinco_base


def tensor_to_list(tensor: torch.Tensor, valid_length: Optional[int] = None) -> List[float]:
    if valid_length is not None:
        tensor = tensor[:valid_length]
    return [float(x) for x in tensor.detach().cpu().tolist()]


def write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pkl.dump(obj, f)


def write_predictions_jsonl(
    ds: Any,
    beam_strs: List[List[str]],
    ptrues: torch.Tensor,
    nvcs: torch.Tensor,
    sc_confs: torch.Tensor,
    dinco_confs: torch.Tensor,
    sampled_strs: List[List[str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex_i, example in enumerate(ds):
            candidates = [dinco_base.clean_str(candidate) for candidate in beam_strs[ex_i]]
            sampled = [dinco_base.clean_str(candidate) for candidate in sampled_strs[ex_i]]
            answer = example.get("answer", {}) or {}
            row: Dict[str, Any] = {
                "question_id": str(example.get("question_id") or ""),
                "question": str(example.get("question") or ""),
                "question_source": str(example.get("question_source") or ""),
                "gold_answer": str(answer.get("value") or ""),
                "gold_aliases": [str(v) for v in answer.get("aliases", []) or []],
                "gold_normalized_aliases": [str(v) for v in answer.get("normalized_aliases", []) or []],
                "closed_book_answer": candidates[0] if candidates else "",
                "candidate_answers": candidates,
                "ptrues": tensor_to_list(ptrues[ex_i], valid_length=len(candidates)),
                "nvc_pre": float(nvcs[ex_i].item()),
                "sc_conf": float(sc_confs[ex_i].item()),
                "dinco_conf": float(dinco_confs[ex_i].item()),
                "sampled_answers": sampled,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    limit = normalize_limit(args.limit)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)

    ds = load_split(args, limit=limit)
    print(f"Loaded {len(ds)} TriviaQA examples from {args.dataset_config}/{args.split}")

    print(f"Loading generator model {args.model_name}")
    model, tokenizer = load_generator_model(args.model_name)
    dinco_backend = dinco_backend_for_model(args.model_name)
    backend_name = "gpt_oss_transformers" if dinco_backend is gpt_oss_dinco else "reference_transformers"

    print("Running beam search")
    beam_strs, beam_lls = dinco_backend.beam_search(
        ds,
        model,
        tokenizer,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    write_pickle(beam_strs, output_dir / "beam_strs.pkl")
    torch.save(beam_lls, output_dir / "beam_lls.pth")

    print("Running lexical cleaning")
    beam_strs, beam_lls = dinco_base.lexical_cleaning(beam_strs, beam_lls)
    write_pickle(beam_strs, output_dir / "beam_strs_cleaned.pkl")
    torch.save(beam_lls, output_dir / "beam_lls_cleaned.pth")

    print("Collecting P(true) estimates")
    ptrues = dinco_backend.get_ptrue(ds, model, tokenizer, beam_strs)
    torch.save(ptrues, output_dir / "ptrues.pth")

    print("Running answer-pair NLI")
    nlis = dinco_base.run_nli(ds, beam_strs)
    torch.save(nlis, output_dir / "nlis.pth")

    print("Computing normalized verbalized confidence")
    nvcs = dinco_base.get_normalized_verbalized_confidence(ptrues, nlis)
    torch.save(nvcs, output_dir / "nvcs.pth")

    print("Sampling generations for self-consistency")
    sampled_strs = dinco_backend.sample_generations(
        ds,
        model,
        tokenizer,
        n_sample=args.n_sample,
        max_new_tokens=args.max_new_tokens,
    )
    write_pickle(sampled_strs, output_dir / "sampled_strs.pkl")

    print("Running self-consistency NLI")
    main_strs = [candidates[0] if candidates else "" for candidates in beam_strs]
    sc_nlis = dinco_base.run_sc_nli(ds, main_strs, sampled_strs)
    torch.save(sc_nlis, output_dir / "sc_nlis_raw.pth")

    sc_nlis_with_main = torch.cat((sc_nlis, torch.ones(len(ds), 1)), dim=-1)
    sc_confs = torch.mean((sc_nlis_with_main > args.sc_match_threshold).float(), dim=-1)
    torch.save(sc_confs, output_dir / "sc_confs.pth")

    dinco_confs = (nvcs + sc_confs) / 2
    torch.save(dinco_confs, output_dir / "dinco_confs.pth")

    predictions_path = output_dir / "predictions.jsonl"
    write_predictions_jsonl(
        ds=ds,
        beam_strs=beam_strs,
        ptrues=ptrues,
        nvcs=nvcs,
        sc_confs=sc_confs,
        dinco_confs=dinco_confs,
        sampled_strs=sampled_strs,
        output_path=predictions_path,
    )

    question_ids = [str(example.get("question_id") or "") for example in ds]
    write_json(question_ids, output_dir / "question_ids.json")

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "limit": limit,
        "backend": backend_name,
        "model_name": args.model_name,
        "num_examples": len(ds),
        "num_beams": args.num_beams,
        "n_sample": args.n_sample,
        "max_new_tokens": args.max_new_tokens,
        "sc_match_threshold": args.sc_match_threshold,
        "output_dir": str(output_dir),
        "predictions_path": str(predictions_path),
        "mean_nvc_pre": float(nvcs.mean().item()) if len(ds) else 0.0,
        "mean_sc_conf": float(sc_confs.mean().item()) if len(ds) else 0.0,
        "mean_dinco_conf": float(dinco_confs.mean().item()) if len(ds) else 0.0,
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
