#!/usr/bin/env python3
"""
WiTQA calibrated-retrieval runner.

Single-answer factoid QA over 14,837 Wikipedia-triple questions. Tests the
DINCO+MiniCheck gating hypothesis on a benchmark stratified by subject
popularity (S-R_count) — retrieval helps on tail entities and can hurt on
head entities. The agent should learn when to retrieve.

Conditions (9):
  1. closed_book              — parametric knowledge only
  2. always_retrieve_top5     — top-5 BM25 passages
  3. oracle_gold_passage      — `supporting_passage` field from the TSV
  4. oracle_toolneeded        — retrieve iff empirical `tool_needed == True`
  5. threshold_dinco_0.3      — retrieve iff DINCO < 0.3
  6. threshold_dinco_0.5      — retrieve iff DINCO < 0.5
  7. threshold_dinco_0.7      — retrieve iff DINCO < 0.7
  8. agent_telemetry          — agent sees DINCO (single-turn decision)
  9. agent_notelemetry        — agent decides without DINCO

Usage:
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  python run_witqa_retrieval.py \\
    --data_path data/witqa/witqa.tsv \\
    --toolneeded_path data/witqa/witqa_toolneeded_labels.jsonl \\
    --index_dir data/retrieval_index \\
    --condition closed_book \\
    --n_examples 5 \\
    --output_path results/witqa_canary.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CALIBANDRETRIEVE_DIR = Path(__file__).resolve().parent
LLMRECOURSE_DIR = CALIBANDRETRIEVE_DIR.parent

for _p in (str(CALIBANDRETRIEVE_DIR), str(LLMRECOURSE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from witqa_utils import (  # noqa: E402
    load_witqa_tsv,
    assert_witqa_schema,
    load_toolneeded_labels,
    attach_toolneeded_labels,
    split_aliases,
    compute_em,
    compute_token_f1,
    build_closed_book_messages,
    CLOSED_BOOK_SYSTEM,
    POPULARITY_BUCKET_NAMES,
)
from qampari_dinco import compute_single_answer_dinco  # noqa: E402
from calibrated_retrieval_core import (  # noqa: E402
    condition_has_telemetry,
    build_vllm_engine,
    make_minicheck_grounder,
    apply_chat_template_safe,
    load_completed_keys,
    append_jsonl_record,
    build_error_record,
    resume_key,
    score_single_claim_against_passages,
    GH200_GPU_MEM_GENERATOR_DEFAULT,
    GH200_GPU_MEM_MINICHECK_DEFAULT,
    MINICHECK_MAX_MODEL_LEN_DEFAULT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
    "closed_book",
    "always_retrieve_top5",
    "oracle_gold_passage",
    "oracle_toolneeded",
    "threshold_dinco_0.3",
    "threshold_dinco_0.5",
    "threshold_dinco_0.7",
    "agent_role",            # Signal-Roles prompt + beam-search DINCO (paper headline)
    "agent_notelemetry",
    "agent_telemetry",       # Legacy directive prompt — kept for backward-compat
    "agent_dinco_only",      # Single-signal ablation: role prompt with grounding signal suppressed
    "agent_minicheck_only",  # Single-signal ablation: role prompt with DINCO signal suppressed
]

THRESHOLD_CONDITIONS = {
    "threshold_dinco_0.3": 0.3,
    "threshold_dinco_0.5": 0.5,
    "threshold_dinco_0.7": 0.7,
}
AGENT_CONDITIONS = {
    "agent_role",
    "agent_telemetry",
    "agent_notelemetry",
    "agent_dinco_only",
    "agent_minicheck_only",
}
# Conditions that need DINCO compute (closed-book beam + SC sampling).
# agent_minicheck_only skips DINCO entirely; agent_notelemetry would too,
# but the existing code path handles that via `has_tel`.
DINCO_REQUIRED_AGENT_CONDITIONS = {
    "agent_role",
    "agent_telemetry",
    "agent_dinco_only",
}
# Conditions that need MiniCheck (post-retrieval grounding).
# agent_dinco_only skips MiniCheck entirely.
MINICHECK_REQUIRED_AGENT_CONDITIONS = {
    "agent_role",
    "agent_telemetry",
    "agent_minicheck_only",
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CLOSED_BOOK_USER = """\
Answer this question with a SHORT direct answer — one entity, name, date, or \
phrase. Do not add hedges or explanations. If unsure, still give your best \
single-answer guess.

Question: {question}
Answer:"""

WITH_EVIDENCE_USER = """\
Answer this question using the provided passages. Give a SHORT direct answer — \
one entity, name, date, or phrase. Do not add hedges or explanations.

Passages:
{passages}

Question: {question}
Answer:"""

AGENT_SYSTEM_TEL = """\
You are a calibrated retrieval agent deciding whether to search Wikipedia.
You will see a candidate closed-book answer and DINCO stability telemetry
(the fraction of 10 sampled generations that produced the same answer).

Return STRICT JSON on a single line:
  {"action": "commit", "reason": "..."} — keep the closed-book answer
  {"action": "retrieve", "reason": "..."} — request top-5 Wikipedia passages

Decision principles:
- High DINCO (>= 0.7) is a strong signal the parametric answer is stable.
- Low DINCO means retrieval is likely to help.
- Do not retrieve on obvious commonsense questions. Do not skip retrieval on \
questions you do not recognize."""

AGENT_SYSTEM_NOTEL = """\
You are a retrieval agent deciding whether to search Wikipedia for this \
question.

Return STRICT JSON on a single line:
  {"action": "commit", "reason": "..."}
  {"action": "retrieve", "reason": "..."}"""


# Signal-Roles system prompt — paper headline framing. No baked-in numeric
# thresholds; names DINCO and MiniCheck families separately and asks the agent
# to reason about each.
AGENT_SYSTEM_ROLE = """\
You are a retrieval policy controller for single-hop factoid QA over Wikipedia.

After producing a closed-book candidate answer, you receive numerical telemetry \
from two distinct models — a self-confidence model and (after retrieval) a \
grounding model — and decide whether to keep the candidate (commit) or fetch \
supporting passages (retrieve).

## Available actions

Return STRICT JSON on a single line with exactly one action:
  {"action": "commit", "reason": "..."}    — keep the closed-book answer
  {"action": "retrieve", "reason": "..."}  — request top-5 Wikipedia passages

## Signal Roles (read this carefully)

You will see two families of signals. They measure different things and should \
be reasoned about separately:

- **DINCO family (DINCO confidence, NVC, SC):** these come from the GENERATOR \
reasoning about its own answer. They measure SELF-CONFIDENCE — how much the \
model agrees with itself across alternatives it can construct via beam search. \
A high DINCO does NOT mean the answer is correct; it means the model has a \
stable internal belief. Models can be confidently wrong, especially on tail \
entities or ambiguous referents.
- **MiniCheck family (g_mean, g_min, per-claim grounding):** these come from a \
SEPARATE GROUNDING MODEL that asks: do the retrieved passages support the \
claim in the current answer? MiniCheck is a GROUNDING signal, not a \
reasoning-quality score. MiniCheck only becomes available AFTER you retrieve \
— at decision time it is forfeit if you commit.
- **These two families are orthogonal.** All four combinations occur \
(high/low DINCO × high/low grounding) and each carries different information.

## Decision principles

1. Reason about self-confidence (DINCO) explicitly in the "reason" field.
2. If the question references a tail entity or rare topic, retrieval is \
informative even at high DINCO — the model can be confidently wrong.
3. If you commit, the grounding signal is forfeit — choose commit only when \
you trust the parametric answer regardless of whether evidence would support it.
4. There is no fixed numeric threshold; weigh the signals and explain your \
reasoning before acting.

Return STRICT JSON only — no markdown, no extra text outside the object."""


REACT_EVIDENCE_REVIEW_USER_TEL = """\
You previously decided to retrieve. Here are the top-5 BM25 passages. Produce \
the final short answer using this evidence.

Passages:
{passages}

Candidate pre-retrieval answer: "{candidate}"  ({confidence_label}={confidence_value:.2f})

Question: {question}
Final Answer:"""

REACT_EVIDENCE_REVIEW_USER_NOTEL = """\
You previously decided to retrieve. Here are the top-5 BM25 passages. Produce \
the final short answer using this evidence.

Passages:
{passages}

Question: {question}
Final Answer:"""


# ---------------------------------------------------------------------------
# Agent action parsing
# ---------------------------------------------------------------------------

_VALID_WITQA_ACTIONS = frozenset({"commit", "retrieve"})


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_witqa_agent_action(raw: str) -> Dict[str, str]:
    parsed = _extract_json_dict(raw)
    if parsed is None:
        return {"action": "retrieve", "reason": "parse_failure", "raw": raw}
    action = str(parsed.get("action", "")).strip().lower()
    if action not in _VALID_WITQA_ACTIONS:
        action = "retrieve"
    return {
        "action": action,
        "reason": str(parsed.get("reason", "")).strip(),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Passage formatting
# ---------------------------------------------------------------------------

def format_passages(passages: Sequence[Dict[str, Any]], *, max_chars: int = 600) -> str:
    if not passages:
        return "(no passages available)"
    lines: List[str] = []
    for i, p in enumerate(passages, 1):
        title = p.get("title") or p.get("doc_title") or ""
        text = p.get("text") or p.get("passage") or p.get("content") or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        header = f"[{i}] {title}" if title else f"[{i}]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def format_gold_passage(passage_text: str, *, max_chars: int = 1200) -> str:
    if not passage_text:
        return "(no gold passage available)"
    if len(passage_text) > max_chars:
        passage_text = passage_text[:max_chars] + "..."
    return f"[Gold]\n{passage_text}"


# ---------------------------------------------------------------------------
# vLLM generation helpers
# ---------------------------------------------------------------------------

def _gen_greedy(llm, tokenizer, messages: List[Dict[str, str]], max_new: int,
                json_schema: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Greedy decode. If `json_schema` is provided, use vLLM guided JSON to
    constrain output to the schema — useful for the agent decision step where
    JSON parse failures would otherwise force a fallback action.
    """
    from vllm import SamplingParams
    prompt = apply_chat_template_safe(tokenizer, messages)
    kwargs = {"temperature": 0.0, "max_tokens": max_new, "n": 1}
    if json_schema is not None:
        # vLLM 0.16+ uses StructuredOutputsParams; older vLLM uses GuidedDecodingParams.
        # Try new API first, fall back to old. Silent fallback (no constraint) is a bug
        # we previously hit — log the choice so it's visible in the run log.
        _gen_greedy._json_logged = getattr(_gen_greedy, "_json_logged", False)
        try:
            from vllm.sampling_params import StructuredOutputsParams
            kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
            if not _gen_greedy._json_logged:
                print("[WiTQA] guided JSON: using StructuredOutputsParams (vLLM 0.16+)", flush=True)
                _gen_greedy._json_logged = True
        except ImportError:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
                if not _gen_greedy._json_logged:
                    print("[WiTQA] guided JSON: using legacy GuidedDecodingParams", flush=True)
                    _gen_greedy._json_logged = True
            except ImportError:
                if not _gen_greedy._json_logged:
                    print("[WiTQA] WARN: no guided-decoding API found in vLLM; running unconstrained", flush=True)
                    _gen_greedy._json_logged = True
    sp = SamplingParams(**kwargs)
    out = llm.generate([prompt], sp, use_tqdm=False)[0]
    text = out.outputs[0].text.strip()
    finish = out.outputs[0].finish_reason or ""
    return text, finish


# Cross-encoder reranker — applied AFTER BM25 to lift precision of top-K.
# Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M params, ~50ms/query).
# Loaded lazily (see `_load_reranker`) so non-rerank runs stay zero-cost.
_RERANKER_CACHE: Dict[str, Any] = {}


def _load_reranker(model_name: str, cache_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load (tokenizer, model) for the cross-encoder reranker; cached by model_name."""
    if not model_name or model_name.lower() in ("none", "off", ""):
        return None
    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        print(f"[WiTQA] WARN: reranker import failed: {e}; falling back to BM25-only", flush=True)
        return None
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=torch.float16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[WiTQA] CrossEncoder reranker loaded: {model_name} on {device}", flush=True)
    bundle = {"tokenizer": tok, "model": model, "device": device}
    _RERANKER_CACHE[model_name] = bundle
    return bundle


def _rerank_passages(reranker: Optional[Dict[str, Any]], query: str,
                     passages: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Rerank a list of passages by cross-encoder relevance to `query`; keep top-K.

    No-op if `reranker` is None or `passages` is short. Score = model logit.
    """
    if not reranker or not passages or len(passages) <= top_k:
        return passages[:top_k] if len(passages) > top_k else passages
    import torch
    tok = reranker["tokenizer"]; model = reranker["model"]; device = reranker["device"]
    queries = [query] * len(passages)
    docs = [(p.get("text") or p.get("passage") or "") for p in passages]
    with torch.no_grad():
        enc = tok(queries, docs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits.squeeze(-1).float().cpu().tolist()
    ranked = sorted(zip(passages, logits), key=lambda x: -x[1])
    out = []
    for p, score in ranked[:top_k]:
        p2 = dict(p)
        p2["rerank_score"] = float(score)
        out.append(p2)
    return out


# JSON schema for the WiTQA / single-turn agent's decision (commit | retrieve).
WITQA_AGENT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["commit", "retrieve"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


def _gen_stochastic(llm, tokenizer, messages: List[Dict[str, str]], *, n: int, temperature: float, max_new: int) -> List[str]:
    """Stochastic samples (used for self-consistency, NOT beam search)."""
    from vllm import SamplingParams
    prompt = apply_chat_template_safe(tokenizer, messages)
    sp = SamplingParams(temperature=temperature, max_tokens=max_new, n=n, top_p=0.95)
    out = llm.generate([prompt], sp, use_tqdm=False)[0]
    return [o.text.strip() for o in out.outputs]


_gen_beam = _gen_stochastic  # backwards-compat alias (semantic was always stochastic)


def _beam_search_candidates(
    llm, tokenizer, messages: List[Dict[str, str]],
    *, beam_width: int, max_new: int, length_penalty: float = 0.0,
) -> List[str]:
    """True beam search via vLLM BeamSearchParams (mirrors HotpotQA pipeline)."""
    from vllm.sampling_params import BeamSearchParams
    prompt = apply_chat_template_safe(tokenizer, messages)
    params = BeamSearchParams(
        beam_width=int(max(1, beam_width)),
        max_tokens=max_new,
        ignore_eos=False,
        temperature=0.0,
        length_penalty=length_penalty,
    )
    outs = llm.beam_search([{"prompt": prompt}], params=params)
    out = outs[0]
    candidates: List[str] = []
    for seq in out.sequences:
        full_text = seq.text or ""
        if full_text.startswith(prompt):
            new_text = full_text[len(prompt):]
        else:
            for marker in ("<|im_start|>assistant", "[/INST]", "<|assistant|>"):
                if marker in full_text:
                    new_text = full_text.rsplit(marker, 1)[-1].lstrip("\n")
                    break
            else:
                new_text = full_text
        for end_tok in ("<|im_end|>", "</s>", "<|eot_id|>"):
            if end_tok in new_text:
                new_text = new_text.split(end_tok)[0]
        candidates.append(new_text.strip())
    return candidates


def generate_short_answer(
    llm, tokenizer, question: str, passages_text: Optional[str], max_new: int,
) -> Tuple[str, str, str]:
    """Return (parsed_answer, raw_text, finish_reason) for a single-answer prompt."""
    if passages_text:
        user = WITH_EVIDENCE_USER.format(question=question, passages=passages_text)
    else:
        user = CLOSED_BOOK_USER.format(question=question)
    messages = [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw, finish = _gen_greedy(llm, tokenizer, messages, max_new)
    parsed = parse_short_answer(raw)
    return parsed, raw, finish


_HARMONY_CHANNEL_RE = re.compile(
    r"(?is)(?:<\|channel\|>[^<]*<\|message\|>|<\|[^|>]+\|>)"
)


def _strip_gptoss_harmony(text: str) -> str:
    """Extract the assistant-final payload from gpt-oss harmony output.

    The gpt-oss family emits channels like
        <|channel|>analysis<|message|>...<|channel|>final<|message|>{answer}<|end|>
    When special tokens are stripped at decode, this collapses to inline
    text like "analysis...assistantfinal{answer}". We split on the
    `final<|message|>` marker if present, otherwise on the textual
    "assistantfinal" / "final<|message|>" forms, and keep what follows.
    """
    if not text:
        return text
    # Explicit channel form (special tokens preserved).
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>", 1)[1]
    # Tokens-stripped form: split on the textual marker.
    elif "assistantfinal" in text:
        text = text.rsplit("assistantfinal", 1)[1]
    elif "final<|message|>" in text:
        text = text.split("final<|message|>", 1)[1]
    if "<|end|>" in text:
        text = text.split("<|end|>", 1)[0]
    # Strip residual special tokens / channel headers.
    text = _HARMONY_CHANNEL_RE.sub("", text)
    return text


def parse_short_answer(raw: str) -> str:
    """Extract a short answer string from the model output.

    Strip <think>...</think>, gpt-oss harmony channels, NFKC normalize, take
    the first non-empty line, strip surrounding quotes and leading 'Answer:',
    'The answer is', 'It is', numbering, bullets.
    """
    import unicodedata
    if not raw:
        return ""
    cleaned = _strip_gptoss_harmony(raw)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        return ""
    cleaned = unicodedata.normalize("NFKC", cleaned)
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    first = re.sub(
        r"^\s*(?:the\s+answer\s+is|answer)\s*[:\-]?\s*",
        "",
        first,
        flags=re.IGNORECASE,
    )
    first = re.sub(r"^\s*(?:it\s+is)\s+", "", first, flags=re.IGNORECASE)
    first = re.sub(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)", "", first)
    if len(first) >= 2 and first[0] in "\"'" and first[-1] == first[0]:
        first = first[1:-1].strip()
    # Drop trailing period / quote if exactly one token and ends with punctuation
    first = first.strip().strip(",")
    return first


def compute_dinco_single(
    llm, tokenizer, question: str, candidate_answer: str, passages_text: Optional[str],
    *, num_beams: int, num_sc: int, beam_t: float, sc_t: float, max_new: int,
) -> Dict[str, Any]:
    """Compute DINCO for a single-answer factoid question.

    Uses `compute_single_answer_dinco` which treats each beam/SC output as one
    whole string — critical for WiTQA answers containing commas (e.g.
    "Washington, D.C."). The QAMPARI per-answer variant would comma-split
    these into multiple tokens and collapse DINCO to zero.
    """
    if passages_text:
        user = WITH_EVIDENCE_USER.format(question=question, passages=passages_text)
    else:
        user = CLOSED_BOOK_USER.format(question=question)
    messages = [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": user},
    ]
    beam_out = _beam_search_candidates(
        llm, tokenizer, messages, beam_width=num_beams, max_new=max_new,
    )
    sc_out = _gen_stochastic(
        llm, tokenizer, messages, n=num_sc, temperature=sc_t, max_new=max_new,
    )
    return compute_single_answer_dinco(
        primary_answer=candidate_answer,
        beam_outputs=beam_out,
        sc_outputs=sc_out,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _bm25_top_k(index, question_id: str, query: str, k: int = 5,
                reranker: Optional[Dict[str, Any]] = None,
                bm25_top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch top-k passages from the BM25 index, optionally reranked.

    If `reranker` is provided, BM25 returns the top `bm25_top_k` (default
    `max(25, k*5)`), the cross-encoder rescores all of them, and the final
    `k` are returned by reranker score. If no reranker, plain BM25 top-k.
    """
    if index is None:
        return []
    bm25_k = bm25_top_k if bm25_top_k is not None else (max(25, k * 5) if reranker else k)
    try:
        hits = index.search_global(query, top_k=bm25_k)
    except AttributeError:
        # Older BM25Index without search_global — try per-question, then bail.
        try:
            hits = index.search(question_id, query, top_k=bm25_k)
        except (KeyError, TypeError):
            return []
    out: List[Dict[str, Any]] = []
    for h in hits:
        row = getattr(h, "row", None)
        if isinstance(row, dict):
            out.append({
                "title": row.get("title") or row.get("doc_title") or row.get("chunk_title") or "",
                "text": (
                    row.get("chunk_body_text")
                    or row.get("chunk_text")
                    or row.get("text")
                    or row.get("passage")
                    or ""
                ),
                "id": row.get("chunk_id") or row.get("doc_id") or row.get("id") or "",
                "score": float(getattr(h, "score", 0.0)),
            })
        elif isinstance(h, dict):
            out.append(h)
        else:
            out.append({
                "title": getattr(h, "title", "") or "",
                "text": getattr(h, "text", "") or getattr(h, "passage", "") or "",
                "id": getattr(h, "doc_id", "") or getattr(h, "id", ""),
                "score": float(getattr(h, "score", 0.0)),
            })
    if reranker:
        out = _rerank_passages(reranker, query, out, top_k=k)
    else:
        out = out[:k]
    return out


# ---------------------------------------------------------------------------
# MiniCheck grounding (single answer)
# ---------------------------------------------------------------------------

def compute_minicheck_single(
    grounder, question: str, answer: str, passages: Sequence[Dict[str, Any]],
) -> Optional[float]:
    """Return MiniCheck grounding probability for the single-answer claim.

    Uses the shared `score_single_claim_against_passages` helper, which matches
    the real `MiniCheckGrounder.score(passages, claims) -> (mean, List[float])`
    API. Returns None if `grounder` is unavailable or inputs are empty.
    """
    if grounder is None or not answer or not passages:
        return None
    claim = f"{question.strip()} {answer.strip()}"
    return score_single_claim_against_passages(grounder, claim, list(passages))


# ---------------------------------------------------------------------------
# Prompt dump (validation criterion #8 — no DINCO leak into notelemetry)
# ---------------------------------------------------------------------------

_PROMPT_DUMPED: set = set()


def _maybe_dump_prompt(
    args, condition: str, qid: str,
    system_prompt: str, user_prompt: str, *, has_tel: bool,
) -> None:
    path = getattr(args, "dump_prompts_path", None)
    if not path:
        return
    if condition in _PROMPT_DUMPED:
        return
    _PROMPT_DUMPED.add(condition)
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "condition": condition,
                "qid": qid,
                "has_tel": bool(has_tel),
                "system": system_prompt,
                "user": user_prompt,
                "contains_dinco": ("DINCO" in user_prompt) or ("DINCO" in system_prompt),
            }, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"[WiTQA] WARN: prompt-dump failed for {condition}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Condition handlers
# ---------------------------------------------------------------------------

def _run_closed_book(llm, tokenizer, grounder, example, args) -> Dict[str, Any]:
    q = example["question_text"]
    _maybe_dump_prompt(
        args, "closed_book", example["qid"],
        CLOSED_BOOK_SYSTEM, CLOSED_BOOK_USER.format(question=q), has_tel=False,
    )
    pred, raw, finish = generate_short_answer(llm, tokenizer, q, None, args.answer_max_new_tokens)
    dinco = compute_dinco_single(
        llm, tokenizer, q, pred, None,
        num_beams=args.num_beams, num_sc=args.num_sc_samples,
        beam_t=args.beam_temperature, sc_t=args.sc_temperature,
        max_new=args.answer_max_new_tokens,
    )
    em = compute_em(pred, example["gold_answer"])
    f1 = compute_token_f1(pred, example["gold_answer"])
    return _build_record(
        example, "closed_book", pred, raw, finish,
        em=em, f1=f1, dinco=dinco, grounding=None,
        retrieved=False, n_passages=0, agent_raw=None, agent_action=None,
    )


def _run_always_retrieve(llm, tokenizer, grounder, example, index, args) -> Dict[str, Any]:
    q = example["question_text"]
    passages = _bm25_top_k(index, example["qid"], q, k=getattr(args, "final_top_k", 5), reranker=getattr(args, "_reranker", None))
    pt = format_passages(passages)
    pred, raw, finish = generate_short_answer(llm, tokenizer, q, pt, args.answer_max_new_tokens)
    dinco = compute_dinco_single(
        llm, tokenizer, q, pred, pt,
        num_beams=args.num_beams, num_sc=args.num_sc_samples,
        beam_t=args.beam_temperature, sc_t=args.sc_temperature,
        max_new=args.answer_max_new_tokens,
    )
    g = compute_minicheck_single(grounder, q, pred, passages)
    em = compute_em(pred, example["gold_answer"])
    f1 = compute_token_f1(pred, example["gold_answer"])
    return _build_record(
        example, "always_retrieve_top5", pred, raw, finish,
        em=em, f1=f1, dinco=dinco, grounding=g,
        retrieved=True, n_passages=len(passages), agent_raw=None, agent_action=None,
        passages=passages,
    )


def _run_oracle_gold(llm, tokenizer, grounder, example, args) -> Dict[str, Any]:
    q = example["question_text"]
    gold_text = example.get("supporting_passage") or ""
    if not gold_text:
        pt = None
        passages = []
    else:
        pt = format_gold_passage(gold_text)
        passages = [{"title": example.get("subject") or "", "text": gold_text, "id": "gold"}]
    pred, raw, finish = generate_short_answer(llm, tokenizer, q, pt, args.answer_max_new_tokens)
    dinco = compute_dinco_single(
        llm, tokenizer, q, pred, pt,
        num_beams=args.num_beams, num_sc=args.num_sc_samples,
        beam_t=args.beam_temperature, sc_t=args.sc_temperature,
        max_new=args.answer_max_new_tokens,
    )
    g = compute_minicheck_single(grounder, q, pred, passages)
    em = compute_em(pred, example["gold_answer"])
    f1 = compute_token_f1(pred, example["gold_answer"])
    return _build_record(
        example, "oracle_gold_passage", pred, raw, finish,
        em=em, f1=f1, dinco=dinco, grounding=g,
        retrieved=bool(passages), n_passages=len(passages),
        agent_raw=None, agent_action=None, passages=passages,
    )


def _run_oracle_toolneeded(llm, tokenizer, grounder, example, index, args) -> Dict[str, Any]:
    tn = example.get("tool_needed")
    if tn is None:
        tn = example.get("paper_tool_needed", True)
    if tn:
        record = _run_always_retrieve(llm, tokenizer, grounder, example, index, args)
        record["condition"] = "oracle_toolneeded"
        record["oracle_tool_needed"] = True
    else:
        record = _run_closed_book(llm, tokenizer, grounder, example, args)
        record["condition"] = "oracle_toolneeded"
        record["oracle_tool_needed"] = False
    return record


def _run_threshold(llm, tokenizer, grounder, example, index, condition, args) -> Dict[str, Any]:
    threshold = THRESHOLD_CONDITIONS[condition]
    q = example["question_text"]
    # Closed-book first
    cb_pred, cb_raw, cb_finish = generate_short_answer(llm, tokenizer, q, None, args.answer_max_new_tokens)
    cb_dinco = compute_dinco_single(
        llm, tokenizer, q, cb_pred, None,
        num_beams=args.num_beams, num_sc=args.num_sc_samples,
        beam_t=args.beam_temperature, sc_t=args.sc_temperature,
        max_new=args.answer_max_new_tokens,
    )
    avg = float(cb_dinco.get("avg_answer_dinco", 0.0))
    if avg < threshold:
        passages = _bm25_top_k(index, example["qid"], q, k=getattr(args, "final_top_k", 5), reranker=getattr(args, "_reranker", None))
        pt = format_passages(passages)
        pred, raw, finish = generate_short_answer(llm, tokenizer, q, pt, args.answer_max_new_tokens)
        post_dinco = compute_dinco_single(
            llm, tokenizer, q, pred, pt,
            num_beams=args.num_beams, num_sc=args.num_sc_samples,
            beam_t=args.beam_temperature, sc_t=args.sc_temperature,
            max_new=args.answer_max_new_tokens,
        )
        g = compute_minicheck_single(grounder, q, pred, passages)
        em = compute_em(pred, example["gold_answer"])
        f1 = compute_token_f1(pred, example["gold_answer"])
        return _build_record(
            example, condition, pred, raw, finish,
            em=em, f1=f1, dinco=post_dinco, grounding=g,
            retrieved=True, n_passages=len(passages),
            agent_raw=None, agent_action=None, passages=passages,
            extra={
                "pre_retrieval_dinco": avg,
                "pre_retrieval_answer": cb_pred,
                "pre_retrieval_raw": cb_raw,
                "threshold": threshold,
            },
        )
    # Commit closed-book
    em = compute_em(cb_pred, example["gold_answer"])
    f1 = compute_token_f1(cb_pred, example["gold_answer"])
    return _build_record(
        example, condition, cb_pred, cb_raw, cb_finish,
        em=em, f1=f1, dinco=cb_dinco, grounding=None,
        retrieved=False, n_passages=0,
        agent_raw=None, agent_action=None,
        extra={
            "pre_retrieval_dinco": avg,
            "threshold": threshold,
        },
    )


# ---------------------------------------------------------------------------
# Single-signal ablation prompts (mirror of run_agent_gated_retrieval_hotpotqa.py
# Block B). AGENT_SYSTEM_DINCO_ONLY strips MiniCheck/grounding refs;
# AGENT_SYSTEM_MINICHECK_ONLY strips DINCO/NVC/SC/beam refs. Used by the new
# conditions agent_dinco_only and agent_minicheck_only.
# ---------------------------------------------------------------------------

AGENT_SYSTEM_DINCO_ONLY = """\
You are a retrieval policy controller for single-hop factoid QA over Wikipedia.

After producing a closed-book candidate answer, you receive numerical telemetry \
from a self-confidence model and decide whether to keep the candidate (commit) \
or fetch supporting passages (retrieve).

## Available actions

Return STRICT JSON on a single line with exactly one action:
  {"action": "commit", "reason": "..."}    — keep the closed-book answer
  {"action": "retrieve", "reason": "..."}  — request top-5 Wikipedia passages

## Signal Role (read this carefully)

You will see one family of signals from a self-confidence channel. The DINCO \
family (DINCO confidence, NVC, SC) comes from the GENERATOR reasoning about its \
own answer. It measures SELF-CONFIDENCE — how much the model agrees with itself \
across alternatives constructed via beam search. A high DINCO does NOT mean the \
answer is correct; it means the model has a stable internal belief. Models can \
be confidently wrong, especially on tail entities or ambiguous referents.

## Decision principles

1. Reason about self-confidence (DINCO) explicitly in the "reason" field.
2. If the question references a tail entity or rare topic, retrieval is \
informative even at high DINCO — the model can be confidently wrong.
3. There is no fixed numeric threshold; weigh the signal and explain your \
reasoning before acting.

Return STRICT JSON only — no markdown, no extra text outside the object."""


AGENT_SYSTEM_MINICHECK_ONLY = """\
You are a retrieval policy controller for single-hop factoid QA over Wikipedia.

You decide whether to fetch supporting passages (retrieve) before committing to \
an answer. A separate GROUNDING MODEL (MiniCheck) is available to score whether \
retrieved passages support the candidate answer — but its signal only becomes \
visible AFTER you retrieve.

## Available actions

Return STRICT JSON on a single line with exactly one action:
  {"action": "commit", "reason": "..."}    — keep the closed-book answer
  {"action": "retrieve", "reason": "..."}  — request top-5 Wikipedia passages

## Signal Role (read this carefully)

You will see one family of signals from a SEPARATE GROUNDING MODEL that asks: \
do the retrieved passages support the claim in the current answer? MiniCheck is \
a GROUNDING signal, not a reasoning-quality score. MiniCheck only becomes \
available AFTER you retrieve — at decision time it is forfeit if you commit.

## Decision principles

1. Before any retrieval has happened, no grounding signal is available — \
default to retrieving for non-trivial factoid questions, especially on \
tail entities or rare topics.
2. If you commit, the grounding signal is forfeit — choose commit only when \
you are confident in the closed-book answer regardless of whether evidence \
would support it.
3. There is no fixed numeric threshold for grounding; weigh the signal at \
evidence-review time and explain your reasoning.

Return STRICT JSON only — no markdown, no extra text outside the object."""


def _select_agent_system_prompt(condition: str, has_tel: bool) -> str:
    if condition == "agent_role":
        return AGENT_SYSTEM_ROLE
    if condition == "agent_dinco_only":
        return AGENT_SYSTEM_DINCO_ONLY
    if condition == "agent_minicheck_only":
        return AGENT_SYSTEM_MINICHECK_ONLY
    if not has_tel:
        return AGENT_SYSTEM_NOTEL
    return AGENT_SYSTEM_TEL


_WITQA_VERBAL_PROB_RE = re.compile(r"(\d+\.\d+|0|1)")
_WITQA_VERBAL_PROB_SYSTEM = (
    "You are a self-evaluating answerer. When asked, you respond with a single "
    "number between 0 and 1 representing your confidence in your current answer. "
    "No words, no explanation — just the number."
)


def _elicit_verbal_confidence_witqa(llm, tokenizer, question: str, answer: str,
                                    max_new_tokens: int = 16) -> float:
    """Per-turn verbal-confidence elicitation for the WiTQA `nvc` mode.

    Asks the model to rate its confidence in *answer* for *question* as a
    number in [0, 1]. Returns 0.5 on parse failure.
    """
    user_msg = (
        f"Question: {question}\n"
        f"Your current answer: {answer!r}\n\n"
        f"On a scale of 0 to 1, how confident are you that this answer is "
        f"correct? Respond with the number ONLY (e.g. 0.73)."
    )
    try:
        text, _ = _gen_greedy(
            llm, tokenizer,
            [
                {"role": "system", "content": _WITQA_VERBAL_PROB_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_new=max_new_tokens,
        )
    except Exception:
        return 0.5
    m = _WITQA_VERBAL_PROB_RE.search(text or "")
    if not m:
        return 0.5
    try:
        val = float(m.group(1))
    except ValueError:
        return 0.5
    if val > 1.0:
        return 1.0
    if val < 0.0:
        return 0.0
    return val


def _select_confidence_signal_witqa(cb_dinco: Optional[Dict[str, Any]],
                                     verbal_prob: Optional[float],
                                     mode: str) -> tuple:
    """Pick (label, value) for the role-prompt confidence slot per mode.

    Returns (label_str, value_float) suitable for `{confidence_label}` and
    `{confidence_value}` slots in the WiTQA agent + followup prompts.
    """
    if mode == "dinco":
        val = float(cb_dinco.get("avg_answer_dinco", 0.0)) if cb_dinco else 0.0
        return ("DINCO", val)
    if mode == "nvc":
        return ("Verbal confidence", float(verbal_prob if verbal_prob is not None else 0.5))
    if mode == "sc":
        sc = 0.0
        if cb_dinco:
            per_ans = cb_dinco.get("per_answer_scores") or []
            if per_ans:
                sc = float(per_ans[0].get("sc_stability") or 0.0)
        return ("Self-consistency", sc)
    raise ValueError(f"unknown confidence_injection_mode: {mode!r}")


def _run_agent(llm, tokenizer, grounder, example, index, condition, args) -> Dict[str, Any]:
    has_tel = condition_has_telemetry(condition)
    # Single-signal ablation gates. agent_minicheck_only suppresses DINCO compute
    # and the confidence injection in the user prompt; agent_dinco_only suppresses
    # MiniCheck grounder (handled at boot via DisabledMiniCheckGrounder).
    needs_dinco = condition in DINCO_REQUIRED_AGENT_CONDITIONS or (has_tel and condition not in AGENT_CONDITIONS)
    q = example["question_text"]
    # Phase 1: closed-book candidate + (conditional) DINCO via beam search
    cand, cand_raw, cand_finish = generate_short_answer(llm, tokenizer, q, None, args.answer_max_new_tokens)
    cb_dinco: Optional[Dict[str, Any]] = None
    beam_candidates_text: Optional[List[str]] = None
    if needs_dinco:
        cb_messages = [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": CLOSED_BOOK_USER.format(question=q)},
        ]
        beam_candidates_text = _beam_search_candidates(
            llm, tokenizer, cb_messages,
            beam_width=args.num_beams, max_new=args.answer_max_new_tokens,
        )
        sc_candidates_text = _gen_stochastic(
            llm, tokenizer, cb_messages,
            n=args.num_sc_samples, temperature=args.sc_temperature,
            max_new=args.answer_max_new_tokens,
        )
        cb_dinco = compute_single_answer_dinco(
            primary_answer=cand,
            beam_outputs=beam_candidates_text,
            sc_outputs=sc_candidates_text,
        )
    # Phase 2: single-turn agent decision
    sys_prompt = _select_agent_system_prompt(condition, has_tel)
    confidence_injection_mode = getattr(args, "confidence_injection_mode", "dinco")
    verbal_prob: Optional[float] = None
    if has_tel and confidence_injection_mode == "nvc":
        verbal_prob = _elicit_verbal_confidence_witqa(
            llm, tokenizer, q, cand, max_new_tokens=16,
        )
    if has_tel and cb_dinco is not None:
        conf_label, conf_value = _select_confidence_signal_witqa(
            cb_dinco, verbal_prob, confidence_injection_mode,
        )
        tel_line = f"{conf_label}: {conf_value:.2f}"
        user = (
            f'Question: "{q}"\n'
            f'Candidate closed-book answer: "{cand}"\n'
            f'{tel_line}\n\n'
            'Return STRICT JSON: {"action": "...", "reason": "..."}'
        )
    else:
        user = (
            f'Question: "{q}"\n'
            f'Candidate closed-book answer: "{cand}"\n\n'
            'Return STRICT JSON: {"action": "...", "reason": "..."}'
        )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]
    _maybe_dump_prompt(args, condition, example["qid"], sys_prompt, user, has_tel=has_tel)
    schema = WITQA_AGENT_JSON_SCHEMA if getattr(args, "use_guided_json", True) else None
    agent_raw, agent_finish = _gen_greedy(
        llm, tokenizer, messages, args.agent_max_new_tokens, json_schema=schema,
    )
    action = parse_witqa_agent_action(agent_raw)

    if action["action"] == "retrieve":
        passages = _bm25_top_k(index, example["qid"], q, k=getattr(args, "final_top_k", 5), reranker=getattr(args, "_reranker", None))
        pt = format_passages(passages)
        if has_tel and cb_dinco is not None:
            post_label, post_value = _select_confidence_signal_witqa(
                cb_dinco, verbal_prob, confidence_injection_mode,
            )
            post_user = REACT_EVIDENCE_REVIEW_USER_TEL.format(
                passages=pt, candidate=cand,
                confidence_label=post_label,
                confidence_value=post_value,
                question=q,
            )
        else:
            post_user = REACT_EVIDENCE_REVIEW_USER_NOTEL.format(passages=pt, question=q)
        post_messages = [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": post_user},
        ]
        post_raw, post_finish = _gen_greedy(llm, tokenizer, post_messages, args.answer_max_new_tokens)
        pred = parse_short_answer(post_raw)
        # Post-retrieval DINCO is suppressed when the condition disables DINCO
        # entirely (agent_minicheck_only); otherwise compute as usual.
        if needs_dinco:
            post_dinco = compute_dinco_single(
                llm, tokenizer, q, pred, pt,
                num_beams=args.num_beams, num_sc=args.num_sc_samples,
                beam_t=args.beam_temperature, sc_t=args.sc_temperature,
                max_new=args.answer_max_new_tokens,
            )
        else:
            post_dinco = None
        # MiniCheck call is a no-op when grounder is a DisabledMiniCheckGrounder
        # (instantiated at boot for condition=agent_dinco_only); returns None.
        g = compute_minicheck_single(grounder, q, pred, passages)
        em = compute_em(pred, example["gold_answer"])
        f1 = compute_token_f1(pred, example["gold_answer"])
        return _build_record(
            example, condition, pred, post_raw, post_finish,
            em=em, f1=f1, dinco=post_dinco, grounding=g,
            retrieved=True, n_passages=len(passages),
            agent_raw=agent_raw, agent_action=action, passages=passages,
            extra={
                "pre_retrieval_dinco": (
                    float(cb_dinco.get("avg_answer_dinco", 0.0)) if cb_dinco else None
                ),
                "pre_retrieval_answer": cand,
                "pre_retrieval_raw": cand_raw,
                "beam_candidates_text": beam_candidates_text,
            },
        )
    # commit
    em = compute_em(cand, example["gold_answer"])
    f1 = compute_token_f1(cand, example["gold_answer"])
    return _build_record(
        example, condition, cand, cand_raw, cand_finish,
        em=em, f1=f1, dinco=cb_dinco, grounding=None,
        retrieved=False, n_passages=0,
        agent_raw=agent_raw, agent_action=action,
        extra={"beam_candidates_text": beam_candidates_text} if beam_candidates_text else None,
    )


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _build_record(
    example: Dict[str, Any], condition: str,
    answer: str, raw: str, finish_reason: str,
    *, em: float, f1: float,
    dinco: Optional[Dict[str, Any]], grounding: Optional[float],
    retrieved: bool, n_passages: int,
    agent_raw: Optional[str], agent_action: Optional[Dict[str, Any]],
    passages: Optional[Sequence[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "question_id": example["qid"],
        "question_text": example["question_text"],
        "gold_answer": example["gold_answer"],
        "subject": example.get("subject", ""),
        "relation": example.get("relation", ""),
        "sr_count": example.get("sr_count", 0),
        "s_count": example.get("s_count", 0),
        "popularity_bucket": example.get("popularity_bucket", ""),
        "paper_tool_needed": example.get("paper_tool_needed"),
        "tool_needed": example.get("tool_needed"),
        "condition": condition,
        "model_response": raw,
        "parsed_answer": answer,
        "finish_reason": finish_reason,
        "em": float(em),
        "token_f1": float(f1),
        "retrieved": bool(retrieved),
        "n_passages": int(n_passages),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if dinco is not None:
        rec["avg_answer_dinco"] = float(dinco.get("avg_answer_dinco", 0.0))
        rec["min_answer_dinco"] = float(dinco.get("min_answer_dinco", 0.0))
        rec["list_completeness"] = float(dinco.get("list_completeness", 0.0))
        rec["dinco_per_answer"] = dinco.get("per_answer_scores", [])
        rec["beam_list_lengths"] = dinco.get("beam_list_lengths", [])
        rec["sc_list_lengths"] = dinco.get("sc_list_lengths", [])
    if grounding is not None:
        rec["post_retrieval_minicheck_grounding"] = float(grounding)
    if agent_raw is not None:
        rec["agent_raw"] = agent_raw
    if agent_action is not None:
        rec["agent_action"] = agent_action["action"]
        rec["agent_reason"] = agent_action.get("reason", "")
    if passages is not None:
        rec["passage_ids"] = [str(p.get("id", "")) for p in passages]
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def run_example(llm, tokenizer, grounder, example, index, condition, args) -> Dict[str, Any]:
    if condition == "closed_book":
        return _run_closed_book(llm, tokenizer, grounder, example, args)
    if condition == "always_retrieve_top5":
        return _run_always_retrieve(llm, tokenizer, grounder, example, index, args)
    if condition == "oracle_gold_passage":
        return _run_oracle_gold(llm, tokenizer, grounder, example, args)
    if condition == "oracle_toolneeded":
        return _run_oracle_toolneeded(llm, tokenizer, grounder, example, index, args)
    if condition in THRESHOLD_CONDITIONS:
        return _run_threshold(llm, tokenizer, grounder, example, index, condition, args)
    if condition in AGENT_CONDITIONS:
        return _run_agent(llm, tokenizer, grounder, example, index, condition, args)
    raise ValueError(f"Unknown WiTQA condition: {condition}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WiTQA calibrated retrieval runner")
    p.add_argument("--data_path", required=True, help="Path to witqa.tsv")
    p.add_argument("--toolneeded_path", default=None, help="Path to witqa_toolneeded_labels.jsonl")
    p.add_argument("--index_dir", default=None, help="Path to BM25 index dir")
    p.add_argument("--condition", required=True, choices=ALL_CONDITIONS + ["all"])
    p.add_argument("--n_examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify_by_bucket", action="store_true")
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--cache_dir", default=None)
    # Generation params
    p.add_argument("--answer_max_new_tokens", type=int, default=256)
    p.add_argument("--agent_max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--num_sc_samples", type=int, default=5)
    p.add_argument(
        "--confidence_injection_mode", type=str, default="dinco",
        choices=["dinco", "nvc", "sc"],
        help="Confidence-metric ablation knob. 'dinco' = composite (default, "
             "matches cross-bench production). 'nvc' = per-turn verbal-prob "
             "elicitation. 'sc' = self-consistency stability only. See "
             "experiment confidence-ablation-witqa-hotpot.",
    )
    p.add_argument("--beam_temperature", type=float, default=0.3)
    p.add_argument("--sc_temperature", type=float, default=0.7)
    # vLLM params
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--gpu_memory_utilization", type=float, default=GH200_GPU_MEM_GENERATOR_DEFAULT)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    # MiniCheck params
    p.add_argument("--no_minicheck", action="store_true",
                   help="Skip MiniCheck (speeds up pure-closed-book canary)")
    p.add_argument("--minicheck_max_model_len", type=int, default=MINICHECK_MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--minicheck_gpu_mem", type=float, default=GH200_GPU_MEM_MINICHECK_DEFAULT)
    # Debug / validation
    p.add_argument("--dump_prompts_path", default=None,
                   help="If set, append first-row system+user prompts per condition to this JSONL file "
                        "for notelemetry-leak validation.")
    p.add_argument("--use_guided_json", action="store_true", default=True,
                   help="Constrain the agent decision step to a JSON schema via vLLM guided decoding "
                        "(recommended for models that don't reliably emit valid JSON, e.g. Mistral "
                        "under the Signal-Roles role prompt).")
    p.add_argument("--no_guided_json", dest="use_guided_json", action="store_false",
                   help="Disable JSON schema constraint (fall back to free-form parsing).")
    # CrossEncoder reranker (BM25 → cross-encoder → top-K). Off by default for
    # backward compat with v2 runs; pass --reranker_model_name to enable.
    p.add_argument("--reranker_model_name", default=None,
                   help="HF model id of a cross-encoder reranker (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2'). "
                        "If unset, uses pure BM25.")
    p.add_argument("--final_top_k", type=int, default=5,
                   help="Final number of passages returned after reranking (default 5).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load + stratify
    print(f"[WiTQA] Loading {args.data_path} (n_examples={args.n_examples}, "
          f"stratify={args.stratify_by_bucket})", flush=True)
    examples = load_witqa_tsv(
        args.data_path,
        n_examples=args.n_examples,
        seed=args.seed,
        stratify_by_bucket=args.stratify_by_bucket,
    )
    if not examples:
        raise RuntimeError("No WiTQA examples loaded")

    # Attach empirical tool_needed labels (optional)
    if args.toolneeded_path:
        labels = load_toolneeded_labels(args.toolneeded_path)
        if labels:
            examples = attach_toolneeded_labels(examples, labels)
            print(f"[WiTQA] Attached tool_needed labels for "
                  f"{sum(1 for e in examples if e.get('tool_needed') is not None)}/"
                  f"{len(examples)} examples", flush=True)
        else:
            print(f"[WiTQA] WARNING: {args.toolneeded_path} empty/missing — "
                  f"oracle_toolneeded will fall back to paper_tool_needed", flush=True)

    # Fail-fast schema check
    for i, ex in enumerate(examples):
        assert_witqa_schema(ex, idx=i)

    # Distribute conditions
    conditions = ALL_CONDITIONS if args.condition == "all" else [args.condition]
    print(f"[WiTQA] Running conditions: {conditions}", flush=True)

    # Resume
    completed = load_completed_keys(output_path)
    print(f"[WiTQA] Resume: {len(completed)} rows already complete in {output_path}", flush=True)

    # BM25 index — load and sanity-probe BEFORE vLLM/MiniCheck so a misconfigured
    # index fails in seconds rather than after ~6 minutes of model loading.
    needs_retrieval = any(
        c in ("always_retrieve_top5", "oracle_toolneeded",
              "threshold_dinco_0.3", "threshold_dinco_0.5", "threshold_dinco_0.7",
              "agent_role", "agent_telemetry", "agent_notelemetry",
              # Single-signal ablation arms also retrieve via the agent action
              # path and MUST rerank identically to agent_role/agent_notelemetry,
              # else the dinco_only/minicheck_only cells are confounded by missing
              # cross-encoder reranking vs their baselines.
              "agent_dinco_only", "agent_minicheck_only")
        for c in conditions
    )
    index = None
    if args.index_dir:
        try:
            from retrieval_index import BM25Index  # noqa: E402
            index = BM25Index.load(Path(args.index_dir))
            print(f"[WiTQA] Loaded BM25 index from {args.index_dir}", flush=True)
        except Exception as e:
            # If the user explicitly passed --index_dir AND a retrieval
            # condition is selected, an unreadable index is a hard error.
            if needs_retrieval:
                raise RuntimeError(
                    f"[WiTQA] Could not load BM25 index from {args.index_dir}: {e}. "
                    f"Retrieval conditions {conditions} require a working index."
                )
            print(f"[WiTQA] WARNING: could not load BM25 index: {e}", flush=True)

        # Sanity-check: run search_global on the first WiTQA question and
        # confirm at least one non-empty passage comes back. If the index was
        # built as per-question pools (QAMPARI-style), global search will
        # degenerate to irrelevant results and the canary will silently
        # produce garbage retrievals.
        if index is not None and needs_retrieval and examples:
            probe_q = examples[0]["question_text"]
            try:
                probe_hits = _bm25_top_k(index, examples[0]["qid"], probe_q, k=3)
            except Exception as e:
                raise RuntimeError(
                    f"[WiTQA] BM25 sanity-check failed on '{probe_q[:80]}...': {e}"
                )
            nonempty = [h for h in probe_hits if (h.get("text") or "").strip()]
            if not nonempty:
                raise RuntimeError(
                    f"[WiTQA] BM25 sanity-check returned no non-empty passages for "
                    f"'{probe_q[:80]}...'. Confirm {args.index_dir} was built over a "
                    f"shared Wikipedia corpus (not per-question candidate pools)."
                )
            print(
                f"[WiTQA] BM25 sanity OK — top title: "
                f"{(nonempty[0].get('title') or '(no title)')[:80]}",
                flush=True,
            )
    elif needs_retrieval:
        raise RuntimeError(
            f"[WiTQA] Conditions {conditions} require retrieval but --index_dir "
            f"was not supplied."
        )

    # vLLM engine
    llm, tokenizer = build_vllm_engine(
        args.model_name,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_dir=args.cache_dir,
        enforce_eager=args.enforce_eager,
    )

    # MiniCheck (only if any condition needs it).
    # `agent_dinco_only` is in AGENT_CONDITIONS but explicitly suppresses MiniCheck
    # (Block A3 ablation); `agent_minicheck_only` keeps MiniCheck.
    minicheck_conditions = (
        "always_retrieve_top5", "oracle_gold_passage", "oracle_toolneeded",
        "threshold_dinco_0.3", "threshold_dinco_0.5", "threshold_dinco_0.7",
        "agent_role", "agent_telemetry", "agent_notelemetry", "agent_minicheck_only",
    )
    needs_grounding = any(c in minicheck_conditions for c in conditions)
    grounder = None
    if needs_grounding and not args.no_minicheck:
        grounder = make_minicheck_grounder(
            gpu_memory_utilization=args.minicheck_gpu_mem,
            max_model_len=args.minicheck_max_model_len,
            cache_dir=args.cache_dir,
        )
    elif any(c == "agent_dinco_only" for c in conditions):
        # Substitute disabled stand-in so downstream code keeps the contract.
        try:
            import multihop_dinco_minicheck_hotpotqa as _mhot
            grounder = _mhot.DisabledMiniCheckGrounder()
            print(
                "[WiTQA] MiniCheck disabled for condition=agent_dinco_only; "
                "using DisabledMiniCheckGrounder stand-in (no MiniCheck-7B load).",
                flush=True,
            )
        except Exception:
            grounder = None

    # CrossEncoder reranker — applied AFTER BM25 to improve recall on tail
    # entities. Stashed on args so _bm25_top_k can pick it up via
    # getattr(args, "_reranker", None).
    args._reranker = None
    rerank_name = getattr(args, "reranker_model_name", None)
    if rerank_name and needs_retrieval:
        args._reranker = _load_reranker(rerank_name, cache_dir=args.cache_dir)

    # Execute
    n_written = 0
    n_skipped = 0
    n_errors = 0
    with output_path.open("a", encoding="utf-8") as writer:
        for i, ex in enumerate(examples):
            for cond in conditions:
                key = resume_key(ex["qid"], cond)
                if key in completed:
                    n_skipped += 1
                    continue
                t0 = time.time()
                try:
                    rec = run_example(llm, tokenizer, grounder, ex, index, cond, args)
                    rec["elapsed_s"] = round(time.time() - t0, 3)
                    append_jsonl_record(writer, rec)
                    n_written += 1
                    if n_written % 10 == 0:
                        print(f"[WiTQA] wrote {n_written} rows (idx={i} cond={cond})", flush=True)
                except Exception as e:
                    n_errors += 1
                    err = build_error_record(
                        question_id=ex["qid"], condition=cond, exc=e,
                    )
                    err["traceback"] = traceback.format_exc()[:4000]
                    append_jsonl_record(writer, err)
                    print(f"[WiTQA] ERROR at qid={ex['qid']} cond={cond}: {e}", flush=True)

    print(
        f"[WiTQA] DONE — {n_written} rows written, {n_skipped} skipped (resume), "
        f"{n_errors} errors. Output: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
