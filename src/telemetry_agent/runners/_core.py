#!/usr/bin/env python3
"""
Shared core for DINCO + MiniCheck calibrated-retrieval runners.

Extracted from run_qampari_retrieval.py so that WiTQA and OverSearchQA runners
can reuse the critical shared infrastructure:

  1. condition_has_telemetry() — the has_tel fix. `"notelemetry" not in condition`.
  2. condition_is_agent() / condition_is_react() — condition classification.
  3. build_vllm_engine() — your GPU-safe defaults (enforce_eager, spawn).
  4. make_minicheck_grounder() — MiniCheck with max_model_len=4096 default.
  5. apply_chat_template_safe() — enable_thinking=False with graceful fallback.
  6. load_completed_keys() / append_jsonl_record() — resume + atomic append.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# your GPU-safe defaults (baked in to protect against regressions)
# ---------------------------------------------------------------------------

GH200_GPU_MEM_GENERATOR_DEFAULT = 0.72
GH200_GPU_MEM_MINICHECK_DEFAULT = 0.18
MINICHECK_MAX_MODEL_LEN_DEFAULT = 4096
VLLM_DTYPE_DEFAULT = "bfloat16"


# ---------------------------------------------------------------------------
# Condition parser — THE has_tel fix
# ---------------------------------------------------------------------------

def condition_has_telemetry(condition: str) -> bool:
    """Return True iff `condition` should show telemetry (DINCO + MiniCheck) to the agent.

    Regression note: the naive `condition.endswith("telemetry")` is BUGGY because
    "notelemetry".endswith("telemetry") is True. This function uses the safe form.

    True:  agent_telemetry, react_telemetry, agent_stateless_telemetry,
           agent_telemetry_premise_aware
    False: agent_notelemetry, react_notelemetry, agent_stateless_notelemetry,
           closed_book, bm25_top20, gold_context_ceiling, origbeam_threshold
    """
    if "notelemetry" in condition:
        return False
    return "telemetry" in condition


def condition_is_agent(condition: str) -> bool:
    return condition.startswith("agent")


def condition_is_react(condition: str) -> bool:
    return condition.startswith("react")


def condition_is_multi_turn(condition: str) -> bool:
    """Agent or ReAct conditions are multi-turn; everything else is single-shot."""
    return condition_is_agent(condition) or condition_is_react(condition)


# ---------------------------------------------------------------------------
# vLLM bootstrap (your GPU defaults)
# ---------------------------------------------------------------------------

def build_vllm_engine(
    model_name: str,
    *,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = GH200_GPU_MEM_GENERATOR_DEFAULT,
    dtype: str = VLLM_DTYPE_DEFAULT,
    cache_dir: Optional[str] = None,
    enforce_eager: bool = True,
    tensor_parallel_size: int = 1,
) -> Tuple[Any, Any]:
    """Build a vLLM LLM + tokenizer with your GPU-safe defaults.

    Prints effective settings so canary logs capture the VRAM split at startup.
    Must be called AFTER `VLLM_WORKER_MULTIPROC_METHOD=spawn` is set in env.
    """
    from vllm import LLM  # noqa: E402

    print(
        f"[vLLM] Loading {model_name} — dtype={dtype} "
        f"max_model_len={max_model_len} "
        f"gpu_memory_utilization={gpu_memory_utilization} "
        f"enforce_eager={enforce_eager} "
        f"tensor_parallel_size={tensor_parallel_size}",
        flush=True,
    )
    llm = LLM(
        model=model_name,
        dtype=dtype,
        download_dir=cache_dir,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


# ---------------------------------------------------------------------------
# MiniCheck factory
# ---------------------------------------------------------------------------

def make_minicheck_grounder(
    *,
    gpu_memory_utilization: float = GH200_GPU_MEM_MINICHECK_DEFAULT,
    max_model_len: int = MINICHECK_MAX_MODEL_LEN_DEFAULT,
    cache_dir: Optional[str] = None,
    model_name: str = "Bespoke-MiniCheck-7B",
    enable_prefix_caching: bool = True,
    allow_cpu_fallback: bool = False,
):
    """Create a MiniCheckGrounder with safe defaults (max_model_len=4096).

    Imports the grounder class from multihop_dinco_minicheck_hotpotqa lazily
    so callers that never use MiniCheck (closed_book-only) don't pay the import.
    """
    # Ensure the runner package is on sys.path.
    pkg_dir = Path(__file__).resolve().parent.parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    import multihop_dinco_minicheck_hotpotqa as hotpot_utils  # noqa: E402

    print(
        f"[MiniCheck] Loading {model_name} — "
        f"gpu_memory_utilization={gpu_memory_utilization} "
        f"max_model_len={max_model_len}",
        flush=True,
    )
    return hotpot_utils.MiniCheckGrounder(
        model_name=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        cache_dir=cache_dir,
        allow_cpu_fallback=allow_cpu_fallback,
    )


# ---------------------------------------------------------------------------
# Chat-template helper (enable_thinking=False, graceful fallback)
# ---------------------------------------------------------------------------

def apply_chat_template_safe(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Apply chat template with thinking disabled; fall back for older tokenizers."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# Resume + append-only writer
# ---------------------------------------------------------------------------

def resume_key(question_id: str, condition: str) -> str:
    return f"{question_id}+{condition}"


def load_completed_keys(
    output_path: Path,
    *,
    qid_field: str = "question_id",
    condition_field: str = "condition",
) -> Set[str]:
    """Scan existing JSONL output and return the set of completed (qid+condition) keys.

    Error records (rows that include an 'error' field) are NOT counted as completed,
    so a retry will re-run the same (qid, condition) pair.
    """
    completed: Set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            qid = rec.get(qid_field, "")
            cond = rec.get(condition_field, "")
            if qid and cond:
                completed.add(resume_key(str(qid), str(cond)))
    return completed


def append_jsonl_record(writer, record: Dict[str, Any]) -> None:
    """Append one record to an open JSONL file and flush immediately."""
    writer.write(json.dumps(record, ensure_ascii=True) + "\n")
    writer.flush()


def build_error_record(
    *, question_id: str, condition: str, exc: BaseException,
) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "condition": condition,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Schema adapter helpers (used by benchmark-specific runners)
# ---------------------------------------------------------------------------

REQUIRED_CANONICAL_FIELDS_SINGLE_ANSWER = (
    "qid",
    "question_text",
    "gold_answer",
)


def assert_canonical_fields(
    example: Dict[str, Any],
    required: Iterable[str],
    *,
    idx: int = -1,
) -> None:
    """Raise ValueError if any required field is missing or None.

    Called BEFORE any generation starts (fail-fast). The first schema bug in
    QAMPARI cost 4 iterations — this mitigates that class of failure.
    """
    missing = [f for f in required if example.get(f) in (None, "")]
    if missing:
        raise ValueError(
            f"Schema-adapter check failed for example idx={idx} qid="
            f"{example.get('qid')!r}: missing/null fields {missing}. "
            f"Available keys: {sorted(example.keys())}"
        )


# ---------------------------------------------------------------------------
# MiniCheck wrapper — matches MiniCheckGrounder.score(passages, claims) API
# ---------------------------------------------------------------------------

def score_single_claim_against_passages(
    grounder,
    claim: str,
    passages: List[Dict[str, Any]],
) -> Optional[float]:
    """Score one claim against a list of passages using MiniCheckGrounder.

    The real `MiniCheckGrounder.score()` takes `Sequence[Passage], Sequence[str]`
    and returns `(mean_prob, List[float])` where the list has one score per
    claim. We pass a list of `Passage(index, title, text)` objects built from
    the runner's dict-shaped passages.

    Returns the single claim's grounding probability (`[0, 1]`) or None if
    MiniCheck cannot be invoked.
    """
    if grounder is None or not claim or not passages:
        return None
    pkg_dir = Path(__file__).resolve().parent.parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    try:
        from telemetry_agent.runners._hotpot_utils import Passage  # noqa: E402
    except Exception:
        return None

    packed: List[Any] = []
    for i, p in enumerate(passages):
        title = p.get("title", "") or ""
        text = p.get("text") or p.get("passage") or p.get("content") or ""
        packed.append(Passage(index=i + 1, title=title, text=text))

    try:
        _mean, probs = grounder.score(passages=packed, claims=[claim])
    except Exception as e:
        # Loud failure — silently returning None lets the validator rubber-stamp
        # a canary where MiniCheck was effectively disabled.
        print(
            f"[MiniCheck] WARN: grounder.score failed: {type(e).__name__}: {e}",
            flush=True,
        )
        return None
    if not probs:
        return None
    try:
        return float(probs[0])
    except (TypeError, ValueError):
        return None
