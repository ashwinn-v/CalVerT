#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import (
    BASE_DIR,
    limit_suffix,
    load_hf_dataset_split,
    maybe_select_subset,
    normalize_limit,
    slugify_name,
    summarize_counts,
    write_json,
)
from retrieval_index import BM25Index, SearchHit

LLMRECOURSE_DIR = Path(__file__).resolve().parents[1]
if str(LLMRECOURSE_DIR) not in sys.path:
    sys.path.insert(0, str(LLMRECOURSE_DIR))

from telemetry_agent.runners import _hotpot_utils as hotpot_utils  # noqa: E402
from telemetry_agent.planner import qwen_planner as planner_utils  # noqa: E402


ANSWER_STATEMENT_PROMPT = """
You are given a HotpotQA subquestion and a candidate short answer.
Rewrite the pair as one answer-critical declarative statement that directly asserts the answer.

Subquestion:
{question}

Candidate answer:
{answer}

Return STRICT JSON only:
{{
  "answer_statement": "single declarative answer-critical statement"
}}

Rules:
- Return exactly one statement, or an empty string if the answer is too vague to support directly.
- The statement must assert exactly one fact.
- The statement must directly assert only the answer-bearing slot asked by the subquestion.
- Use the answer explicitly in the statement.
- Keep the statement self-contained and decontextualized.
- Do not add side facts, dates, locations, roles, or conjunctions unless strictly necessary to make that one answer-critical fact grammatical and identifiable.
- If the answer is abstention-style, meta, or does not directly fill the subquestion slot, return an empty string.
- Do not output explanations, lists, or markdown.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multihop DINCO-gated calibrated retrieval with memory on HotpotQA."
    )
    parser.add_argument("--dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument(
        "--dataset_config",
        "--dataset_subset",
        dest="dataset_config",
        type=str,
        default="distractor",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=2000, help="<= 0 means full split")
    parser.add_argument(
        "--indexed_pool_limit",
        type=int,
        default=None,
        help=(
            "If set, first restrict the dataset to the first N examples before optional "
            "shuffle/limit. Use 2000 to sample from the indexed validation pool."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Defaults to false so the first validation examples are used.",
    )
    parser.add_argument(
        "--example_id",
        type=str,
        default=None,
        help="Run exactly one example id. Overrides --limit/--shuffle subset selection.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(BASE_DIR / "data" / "hotpotqa_distractor_validation_s0_n2000_chunks_bm25_index"),
    )
    parser.add_argument("--retrieval_top_k", type=int, default=8)
    parser.add_argument("--audit_top_k", type=int, default=8)
    parser.add_argument(
        "--retry_on_low_support",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry_extra_top_k", type=int, default=4)
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="dinco_gate",
        choices=["dinco_gate", "always_retrieve", "closed_book_only"],
    )
    parser.add_argument(
        "--gate_on",
        type=str,
        default="dinco",
        choices=["dinco", "nvc"],
    )
    parser.add_argument("--gate_threshold", type=float, default=0.80)
    parser.add_argument("--support_mean_threshold", type=float, default=0.70)
    parser.add_argument("--support_min_threshold", type=float, default=0.50)
    parser.add_argument(
        "--root_subquestion_policy",
        type=str,
        default="allow_closed_book_commit",
        choices=["always_retrieve", "allow_closed_book_commit", "skip_without_commit"],
    )
    parser.add_argument("--max_initial_subquestions", type=int, default=4)
    parser.add_argument("--max_subquestion_depth", type=int, default=2)
    parser.add_argument("--max_subquestion_nodes", type=int, default=12)
    parser.add_argument("--planner_max_new_tokens", type=int, default=800)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--n_distractors", type=int, default=5)
    parser.add_argument("--n_sc_samples", type=int, default=5)
    parser.add_argument("--sc_match_threshold", type=float, default=0.90)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--generator_device_map", type=str, default="auto")
    parser.add_argument(
        "--generator_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    parser.add_argument("--minicheck_model_name", type=str, default="Bespoke-MiniCheck-7B")
    parser.add_argument(
        "--allow_minicheck_cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minicheck_cpu_fallback_model_name", type=str, default="roberta-large")
    parser.add_argument("--minicheck_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--minicheck_max_model_len", type=int, default=None)
    parser.add_argument("--minicheck_gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument(
        "--noground",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated no-op. Retrieval now always uses MiniCheck grounding.",
    )
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument(
        "--printbad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the failing question/subquestion context if the DINCO pre-call raises.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def file_stem_for_subset(dataset_config: str, split: str, seed: int, limit: Optional[int]) -> str:
    return f"hotpotqa_{slugify_name(dataset_config)}_{slugify_name(split)}_s{seed}_{limit_suffix(limit)}"


def default_output_paths(args: argparse.Namespace, limit: Optional[int]) -> Tuple[Path, Path]:
    subset_stem = file_stem_for_subset(args.dataset_config, args.split, args.seed, limit)
    run_name = (
        f"{subset_stem}_multihop_{args.routing_mode}_{args.gate_on}_t{str(args.gate_threshold).replace('.', '_')}_"
        f"{slugify_name(args.model_name.split('/')[-1])}"
    )
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else BASE_DIR / "results" / f"{run_name}.jsonl"
    summary_json = (
        Path(args.summary_json) if args.summary_json else BASE_DIR / "results" / f"{run_name}.summary.json"
    )
    return output_jsonl, summary_json


def _is_musique_dataset(dataset_name: str) -> bool:
    """True iff `dataset_name` points at a MuSiQue HF mirror."""
    if not dataset_name:
        return False
    return "musique" in dataset_name.lower()


def _adapt_musique_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a MuSiQue (bdsaglam/musique, dgslibisey/MuSiQue) row to HotpotQA-distractor schema.

    MuSiQue ships:
        paragraphs = [{idx, title, paragraph_text, is_supporting}, ...]   # 20 per row
        question, answer, answer_aliases, answerable, question_decomposition
    We map to:
        _id              = id
        question         = question
        answer           = answer (string)
        context          = {"title": [...], "sentences": [[paragraph_text], ...]}   # one-sentence list per para
        supporting_facts = {"title": [supporting titles], "sent_id": [0, 0, ...]}   # paragraph-level
    """
    out = dict(example)
    out["_id"] = out.get("_id") or out.get("id")
    paragraphs = out.get("paragraphs") or []
    if isinstance(paragraphs, list) and paragraphs and isinstance(paragraphs[0], dict):
        titles = [p.get("title", "") for p in paragraphs]
        sentences = [[p.get("paragraph_text", "")] for p in paragraphs]
        out["context"] = {"title": titles, "sentences": sentences}
        sf_titles = [p.get("title", "") for p in paragraphs if p.get("is_supporting")]
        out["supporting_facts"] = {"title": sf_titles, "sent_id": [0] * len(sf_titles)}
    # Surface MuSiQue's answer_aliases as a flat list of acceptable gold strings.
    # MuSiQue ships `answer` (string) + `answer_aliases` (list) — official scoring
    # uses max-over-aliases. Without this the EM/F1 are conservatively biased low.
    aliases_raw = out.get("answer_aliases") or []
    aliases = [str(a) for a in aliases_raw if a]
    answer = str(out.get("answer", "") or "")
    if answer and answer not in aliases:
        aliases.insert(0, answer)
    out["answer_aliases"] = aliases
    return out


def _is_2wiki_dataset(dataset_name: str) -> bool:
    """True iff `dataset_name` points at a 2WikiMultihopQA HF mirror."""
    if not dataset_name:
        return False
    n = dataset_name.lower()
    return "2wikimultihop" in n or "2wiki_multihop" in n or "2wiki-multihop" in n


def _adapt_2wiki_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a 2WikiMultihopQA row to HotpotQA-distractor schema.

    2WikiMultihop (voidful/2wikimultihopqa) ships:
        context           = [[title, [sent1, sent2, ...]], ...]
        supporting_facts  = [[title, sent_idx], ...]
    HotpotQA (hotpotqa/hotpot_qa, distractor) ships:
        context          = {"title": [...], "sentences": [[...], ...]}
        supporting_facts = {"title": [...], "sent_id": [...]}

    `build_passages` and the supporting-facts join in `multihop_dinco_minicheck_hotpotqa`
    expect the HotpotQA shape, so we rewrite in place.
    """
    out = dict(example)
    ctx = out.get("context")
    if isinstance(ctx, list):  # 2Wiki list-of-pairs
        titles = [pair[0] for pair in ctx if isinstance(pair, (list, tuple)) and len(pair) >= 1]
        sentences = [list(pair[1]) if len(pair) >= 2 else [] for pair in ctx if isinstance(pair, (list, tuple)) and len(pair) >= 1]
        out["context"] = {"title": titles, "sentences": sentences}
    sf = out.get("supporting_facts")
    if isinstance(sf, list):  # 2Wiki list-of-pairs
        sf_titles = [pair[0] for pair in sf if isinstance(pair, (list, tuple)) and len(pair) >= 1]
        sf_sids = [int(pair[1]) for pair in sf if isinstance(pair, (list, tuple)) and len(pair) >= 2]
        out["supporting_facts"] = {"title": sf_titles, "sent_id": sf_sids}
    return out


def _load_2wiki_split(dataset_name: str, split: str):
    """Bypass pyarrow JSON schema inference for voidful/2wikimultihopqa.

    pyarrow's JSON loader fails on the `context` column (list-of-pairs that
    can't be unified across rows). We hand-stream the raw JSON via
    huggingface_hub and convert with pre-adapted dicts so HF's `Dataset.from_list`
    sees uniform Python objects.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    import json
    # voidful/2wikimultihopqa ships dev.json/test.json/train.json. HF's
    # `load_dataset` auto-aliases validation→dev; replicate that here.
    split_aliases = {
        "validation": ["dev", "validation"],
        "val": ["dev", "val"],
        "dev": ["dev", "validation"],
        "test": ["test"],
        "train": ["train"],
    }.get(split, [split])
    candidates = []
    for s in split_aliases:
        candidates += [f"{s}.json", f"data/{s}.json", f"{s}.jsonl"]
    last_err = None
    local_path = None
    for fname in candidates:
        try:
            local_path = hf_hub_download(
                repo_id=dataset_name, filename=fname, repo_type="dataset",
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if local_path is None:
        raise RuntimeError(
            f"Could not locate any of {candidates} in {dataset_name}: {last_err}"
        )
    print(f"[2wiki] streaming raw JSON from {local_path}", flush=True)
    if local_path.endswith(".jsonl"):
        with open(local_path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with open(local_path, "r") as f:
            records = json.load(f)
    print(f"[2wiki] loaded {len(records)} raw rows; pre-adapting before HF Dataset", flush=True)
    adapted = [_adapt_2wiki_example(r) for r in records]
    return Dataset.from_list(adapted)


def _load_musique_split(dataset_name: str, split: str):
    """Stream MuSiQue JSONL and pre-adapt to HotpotQA-distractor schema.

    bdsaglam/musique ships musique_ans_v1.0_{train,dev}.jsonl. We use the
    answerable-only split for the recipe paper (the recipe doesn't address
    abstention; OverSearchQA-style scope was dropped earlier).
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    import json
    split_aliases = {
        "validation": ["dev", "validation"],
        "val": ["dev"],
        "dev": ["dev"],
        "test": ["test", "dev"],  # MuSiQue test labels are private; dev is the public eval split
        "train": ["train"],
    }.get(split, [split])
    candidates: List[str] = []
    for s in split_aliases:
        candidates += [
            f"musique_ans_v1.0_{s}.jsonl",
            f"musique_full_v1.0_{s}.jsonl",
            f"{s}.jsonl",
        ]
    last_err = None
    local_path = None
    for fname in candidates:
        try:
            local_path = hf_hub_download(repo_id=dataset_name, filename=fname, repo_type="dataset")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if local_path is None:
        raise RuntimeError(f"Could not locate any of {candidates} in {dataset_name}: {last_err}")
    print(f"[musique] streaming raw JSONL from {local_path}", flush=True)
    with open(local_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"[musique] loaded {len(records)} raw rows; pre-adapting before HF Dataset", flush=True)
    adapted = [_adapt_musique_example(r) for r in records]
    return Dataset.from_list(adapted)


def load_examples(args: argparse.Namespace, limit: Optional[int]):
    # 2WikiMultihop on HF (voidful/2wikimultihopqa) has no config; force None
    # to avoid `load_dataset` rejecting an irrelevant 'distractor' config.
    dataset_config = None if _is_2wiki_dataset(args.dataset_name) else args.dataset_config
    if _is_musique_dataset(args.dataset_name):
        split = _load_musique_split(args.dataset_name, args.split)
    elif _is_2wiki_dataset(args.dataset_name):
        # voidful/2wikimultihopqa fails pyarrow JSON schema inference (`context`
        # column has list-of-pairs that pyarrow can't unify across rows). Bypass
        # the fast-path by streaming the raw JSON ourselves; the helper applies
        # the HotpotQA-schema adapter before constructing the HF Dataset.
        split = _load_2wiki_split(args.dataset_name, args.split)
    else:
        split = load_hf_dataset_split(
            dataset_name=args.dataset_name,
            dataset_config=dataset_config,
            split=args.split,
            dataset_path=args.dataset_path,
        )
    if args.example_id:
        target = str(args.example_id).strip()
        matches = [
            idx
            for idx, example in enumerate(split)
            if str(example.get("_id") or example.get("id") or example.get("question_id") or "") == target
        ]
        if not matches:
            raise ValueError(f"Example id '{target}' not found in {args.dataset_name}:{args.dataset_config}:{args.split}")
        return split.select([matches[0]])
    indexed_pool_limit = normalize_limit(getattr(args, "indexed_pool_limit", None))
    if indexed_pool_limit is not None:
        split = maybe_select_subset(split, limit=indexed_pool_limit, shuffle=False, seed=args.seed)
    return maybe_select_subset(split, limit=limit, shuffle=args.shuffle, seed=args.seed)


def clean_short_answer(qwen_model: Any, question: str, answer: str) -> str:
    text = str(answer or "").strip()
    if hasattr(qwen_model, "shorten_answer_for_hotpot"):
        try:
            text = qwen_model.shorten_answer_for_hotpot(question=question, answer=text)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(qwen_model, "clean_answer_for_dinco"):
        try:
            text = qwen_model.clean_answer_for_dinco(text)
        except Exception:  # noqa: BLE001
            pass
    text = str(text or "").strip()
    return text or "insufficient evidence"


def hit_to_passage(hit: SearchHit) -> hotpot_utils.Passage:
    return hotpot_utils.Passage(
        index=0,
        title=str(hit.row.get("title") or ""),
        text=str(hit.row.get("chunk_body_text") or hit.row.get("chunk_text") or ""),
    )


def reindex_passages(passages: Sequence[hotpot_utils.Passage]) -> List[hotpot_utils.Passage]:
    return [
        hotpot_utils.Passage(index=i, title=str(p.title), text=str(p.text))
        for i, p in enumerate(passages)
    ]


def search_passages(index: BM25Index, question_id: str, query: str, top_k: int) -> List[SearchHit]:
    if top_k <= 0:
        return []
    try:
        return index.search(question_id=question_id, query=query, top_k=top_k)
    except KeyError:
        return []


def parse_claims_from_output(text: str) -> List[str]:
    parsed = hotpot_utils.extract_json_dict(text)
    claims: List[str] = []
    if parsed and isinstance(parsed.get("answer_statement"), str):
        statement = parsed["answer_statement"].strip()
        if statement:
            claims.append(statement)
    elif parsed and isinstance(parsed.get("answer_support_claims"), list):
        for item in parsed["answer_support_claims"]:
            if isinstance(item, dict):
                claim = str(item.get("claim", "")).strip()
                if claim:
                    claims.append(claim)
                    break
            elif isinstance(item, str):
                item = item.strip()
                if item:
                    claims.append(item)
                    break
    return hotpot_utils.unique_keep_order(claims)


def generate_claims_for_answer(qwen_model: Any, question: str, answer: str) -> List[str]:
    if not str(answer or "").strip():
        return []
    if not hasattr(qwen_model, "generate"):
        return [str(answer).strip()]
    prompt = ANSWER_STATEMENT_PROMPT.format(question=question, answer=answer)
    raw = qwen_model.generate(prompt, max_new_tokens=220)
    claims = parse_claims_from_output(raw)
    return hotpot_utils.unique_keep_order(claims)


def score_claims_max_over_passages(
    grounder: Any,
    passages: Sequence[hotpot_utils.Passage],
    claims: Sequence[str],
) -> Dict[str, Any]:
    clean_claims = [str(claim).strip() for claim in claims if str(claim).strip()]
    if not clean_claims or not passages:
        return {
            "g_mean": 0.0,
            "g_min": 0.0,
            "claim_supports": [],
            "passage_support_matrix": [],
        }

    if not hasattr(grounder, "model"):
        g, claim_scores = grounder.score(passages, clean_claims)
        # DisabledMiniCheckGrounder (single-signal ablation) returns (None, []).
        # Treat that as a no-signal zero-dict so downstream code paths that read
        # `support["g_mean"]` keep working without injecting a fake numeric score.
        if g is None:
            return {
                "g_mean": 0.0,
                "g_min": 0.0,
                "claim_supports": [],
                "passage_support_matrix": [],
            }
        return {
            "g_mean": float(g),
            "g_min": float(min(claim_scores)) if claim_scores else 0.0,
            "claim_supports": [float(x) for x in claim_scores],
            "passage_support_matrix": [],
        }

    docs: List[str] = []
    claim_inputs: List[str] = []
    pair_map: List[Tuple[int, int]] = []
    for claim_i, claim in enumerate(clean_claims):
        for passage_i, passage in enumerate(passages):
            docs.append(f"Title: {passage.title}\n\n{passage.text}")
            claim_inputs.append(claim)
            pair_map.append((claim_i, passage_i))

    _, probs, _, _ = grounder.model.score(docs=docs, claims=claim_inputs)
    matrix = np.zeros((len(clean_claims), len(passages)), dtype=np.float32)
    for (claim_i, passage_i), prob in zip(pair_map, probs):
        matrix[claim_i, passage_i] = float(prob)

    claim_supports = matrix.max(axis=1).tolist()
    return {
        "g_mean": float(np.mean(claim_supports)) if claim_supports else 0.0,
        "g_min": float(np.min(claim_supports)) if claim_supports else 0.0,
        "claim_supports": [float(x) for x in claim_supports],
        "passage_support_matrix": matrix.tolist(),
    }


def is_supported(score_dict: Dict[str, Any], mean_threshold: float, min_threshold: float) -> bool:
    return float(score_dict.get("g_mean", 0.0)) >= mean_threshold and float(score_dict.get("g_min", 0.0)) >= min_threshold


def skipped_support_result(reason: str) -> Dict[str, Any]:
    return {
        "g_mean": None,
        "g_min": None,
        "claim_supports": [],
        "passage_support_matrix": [],
        "skipped": True,
        "reason": reason,
    }


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def choose_route(routing_mode: str, gate_score: float, gate_threshold: float) -> str:
    if routing_mode == "always_retrieve":
        return "retrieve"
    if routing_mode == "closed_book_only":
        return "skip"
    return "retrieve" if gate_score < gate_threshold else "skip"


def select_claims_for_attempt(
    qwen_model: Any,
    question: str,
    answer: str,
    support_claims: Sequence[str],
) -> List[str]:
    claims = hotpot_utils.unique_keep_order([str(claim).strip() for claim in support_claims if str(claim).strip()])
    if claims:
        return claims
    return generate_claims_for_answer(qwen_model=qwen_model, question=question, answer=answer)


class _RetryExampleFromStart(RuntimeError):
    def __init__(self, trace_event: Dict[str, Any]) -> None:
        super().__init__(str(trace_event.get("error", "")))
        self.trace_event = dict(trace_event)


def _is_retryable_dinco_azure_failure(exc: Exception) -> bool:
    seen: set[int] = set()
    stack: List[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current)
        if "Azure Responses request failed after retries:" in message:
            return True
        if "Azure Chat Completions request failed after retries:" in message:
            return True
        if "Network error calling Azure Responses API:" in message:
            return True
        if "calling Azure Responses API:" in message and "HTTP " in message:
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)
    return False


class CalibratedPlannerMemoryRunner(planner_utils.PlannerMemoryRunner):
    def __init__(
        self,
        *,
        planner: Any,
        qwen_model: Any,
        subquestion_qwen_model: Optional[Any] = None,
        dinco: Any,
        grounder: Any,
        index: BM25Index,
        gate_on: str,
        gate_threshold: float,
        support_mean_threshold: float,
        support_min_threshold: float,
        routing_mode: str,
        retry_on_low_support: bool,
        retry_extra_top_k: int,
        audit_top_k: int,
        root_subquestion_policy: str,
        max_initial_subquestions: int,
        max_subquestion_depth: int,
        max_subquestion_nodes: int,
        retrieval_top_k: int,
        n_distractors: int,
        printbad: bool = False,
        final_answer_reasoning_effort: Optional[str] = None,
        retry_example_on_dinco_azure_failure: bool = False,
        force_retrieval_on_repeat_dinco_azure_failure: bool = False,
        noground: bool = False,
        enable_sampling_dinco_telemetry: bool = False,
        sampling_dinco_n_samples: int = 10,
    ) -> None:
        super().__init__(
            planner=planner,
            qwen_model=qwen_model,
            subquestion_qwen_model=subquestion_qwen_model,
            dinco=dinco,
            max_initial_subquestions=max_initial_subquestions,
            max_subquestion_depth=max_subquestion_depth,
            max_subquestion_nodes=max_subquestion_nodes,
            confidence_threshold=gate_threshold,
            retrieval_top_k=retrieval_top_k,
            n_distractors=n_distractors,
        )
        self.grounder = grounder
        self.index = index
        self.gate_on = gate_on
        self.gate_threshold = float(gate_threshold)
        self.support_mean_threshold = float(support_mean_threshold)
        self.support_min_threshold = float(support_min_threshold)
        self.routing_mode = routing_mode
        self.retry_on_low_support = bool(retry_on_low_support)
        self.retry_extra_top_k = max(0, int(retry_extra_top_k))
        self.audit_top_k = max(int(audit_top_k), int(retrieval_top_k) + self.retry_extra_top_k)
        self.root_subquestion_policy = str(root_subquestion_policy)
        self.printbad = bool(printbad)
        self.final_answer_reasoning_effort = (
            None if final_answer_reasoning_effort is None else str(final_answer_reasoning_effort)
        )
        self.retry_example_on_dinco_azure_failure = bool(retry_example_on_dinco_azure_failure)
        self.force_retrieval_on_repeat_dinco_azure_failure = bool(force_retrieval_on_repeat_dinco_azure_failure)
        self.noground = bool(noground)
        self.enable_sampling_dinco_telemetry = bool(enable_sampling_dinco_telemetry)
        self.sampling_dinco_n_samples = max(2, int(sampling_dinco_n_samples))
        self._current_question_id = ""
        self._current_example_attempt = 0

    @staticmethod
    def _set_runtime(node: planner_utils.SubquestionNode, **updates: Any) -> None:
        runtime = dict(getattr(node, "_calib_runtime", {}))
        runtime.update(updates)
        setattr(node, "_calib_runtime", runtime)

    @staticmethod
    def _get_runtime(node: planner_utils.SubquestionNode) -> Dict[str, Any]:
        return dict(getattr(node, "_calib_runtime", {}))

    def _emit_printbad(
        self,
        *,
        question: str,
        node: planner_utils.SubquestionNode,
        execution_subquestion: str,
        dependency_entries: Sequence[Dict[str, Any]],
        pre_route: str,
        exc: Exception,
    ) -> None:
        if not self.printbad:
            return
        dependency_ids = [str(entry.get("node_id", "")) for entry in dependency_entries]
        fields = {
            "question_id": self._current_question_id,
            "question": question,
            "node_id": node.id,
            "node_subquestion": node.subquestion,
            "execution_subquestion": execution_subquestion,
            "pre_route": pre_route,
            "dependency_ids": dependency_ids,
            "error": str(exc),
        }
        print("[printbad] DINCO pre-call failed", file=sys.stderr)
        for key, value in fields.items():
            print(f"[printbad] {key}={json.dumps(value, ensure_ascii=False)}", file=sys.stderr)
        sys.stderr.flush()

    @staticmethod
    def _make_dinco_azure_fallback_attempt(error_text: str) -> planner_utils.AttemptResult:
        return planner_utils.AttemptResult(
            answer="",
            raw_answer="",
            nvc=None,
            sc_conf=None,
            dinco_conf=None,
            source="question_only_dinco_azure_failed",
            support_claims=[],
            explanation=f"DINCO Azure request failed; forcing retrieval. {error_text}",
            dinco_candidates=[],
            dinco_ptrues=[],
            available=False,
        )

    def _gate_score(self, *, nvc: Optional[float], dinco_conf: Optional[float]) -> float:
        if self.gate_on == "nvc":
            return float(nvc or 0.0)
        return float(dinco_conf or 0.0)

    def _answer_with_passages(
        self,
        *,
        question: str,
        passages: Sequence[hotpot_utils.Passage],
        source: str,
        unavailable_explanation: str,
        use_dinco: bool = False,
        reasoning_effort: Optional[str] = None,
        refinement: bool = False,
        previous_answer: str = "",
        previous_claims: Optional[Sequence[str]] = None,
    ) -> planner_utils.AttemptResult:
        return self.answer_engine.answer_with_passages(
            question=question,
            passages=passages,
            source=source,
            unavailable_explanation=unavailable_explanation,
            use_dinco=use_dinco,
            reasoning_effort=reasoning_effort,
            refinement=refinement,
            previous_answer=previous_answer,
            previous_claims=previous_claims,
        )

    def _question_only_attempt(self, subquestion: str) -> planner_utils.AttemptResult:
        pre = self.dinco.compute(question=subquestion, answer="", passages=[], n_distractors=self.n_distractors)
        answer = clean_short_answer(
            self.subquestion_qwen_model,
            question=subquestion,
            answer=pre.candidates[0] if pre.candidates else "",
        )
        claims = select_claims_for_attempt(
            qwen_model=self.subquestion_qwen_model,
            question=subquestion,
            answer=answer,
            support_claims=[],
        )
        return planner_utils.AttemptResult(
            answer=answer,
            raw_answer=answer,
            nvc=float(pre.nvc),
            sc_conf=float(pre.sc_conf),
            dinco_conf=float(pre.final_conf),
            source="question_only",
            support_claims=claims,
            explanation="Question-only DINCO pre-answer.",
            dinco_candidates=list(pre.candidates),
            dinco_ptrues=[float(x) for x in pre.ptrues],
            available=bool(answer.strip()),
        )

    def _dependency_memory_attempt(
        self,
        subquestion: str,
        dependency_passages: Sequence[hotpot_utils.Passage],
    ) -> planner_utils.AttemptResult:
        attempt = self._answer_with_passages(
            question=subquestion,
            passages=reindex_passages(dependency_passages),
            source="dependency_memory",
            unavailable_explanation="No dependency memory was available for this subquestion.",
            use_dinco=False,
        )
        if attempt.available:
            attempt.support_claims = select_claims_for_attempt(
                qwen_model=self.subquestion_qwen_model,
                question=subquestion,
                answer=attempt.answer,
                support_claims=attempt.support_claims,
            )
        return attempt

    def _score_retrieval_attempt(
        self,
        *,
        subquestion: str,
        dependency_passages: Sequence[hotpot_utils.Passage],
        retrieved_passages: Sequence[hotpot_utils.Passage],
        fallback_answer: str,
        evidence_chunk_ids: Sequence[str],
        evidence_titles: Sequence[str],
        evidence_scores: Sequence[float],
        source: str,
        refinement: bool = False,
        previous_answer: str = "",
        previous_claims: Optional[Sequence[str]] = None,
        compute_sampling_dinco: bool = False,
    ) -> Dict[str, Any]:
        combined_passages = reindex_passages(list(dependency_passages) + list(retrieved_passages))
        attempt = self._answer_with_passages(
            question=subquestion,
            passages=combined_passages,
            source=source,
            unavailable_explanation="No passages were available for this subquestion.",
            use_dinco=False,
            refinement=refinement,
            previous_answer=previous_answer,
            previous_claims=previous_claims,
        )
        answer = attempt.answer if attempt.available else fallback_answer
        claims = select_claims_for_attempt(
            qwen_model=self.subquestion_qwen_model,
            question=subquestion,
            answer=answer,
            support_claims=attempt.support_claims,
        )
        support = score_claims_max_over_passages(self.grounder, combined_passages, claims)
        attempt.support_claims = list(claims)
        attempt.retrieved_passage_indices = list(range(len(retrieved_passages)))
        attempt.retrieved_titles = [str(title) for title in evidence_titles]

        sampling_dinco_result: Optional[Dict[str, Any]] = None
        if (
            compute_sampling_dinco
            and self.enable_sampling_dinco_telemetry
            and self.dinco is not None
            and attempt.available
            and answer
        ):
            try:
                sd = self.dinco.compute_post_retrieval_sampling(
                    question=subquestion,
                    answer=answer,
                    passages=combined_passages,
                    n_samples=self.sampling_dinco_n_samples,
                )
                attempt.sampling_dinco_conf = float(sd.sampling_dinco_conf)
                attempt.sampling_dinco_degenerate = bool(sd.degenerate)
                attempt.sampling_dinco_agreement_rate = float(sd.agreement_rate)
                attempt.sampling_dinco_n_unique = int(sd.n_unique_distractors)
                attempt.sampling_distractors = list(sd.candidates[1:])
                attempt.sampling_ptrues = [float(p) for p in sd.ptrues[1:]]
                sampling_dinco_result = {
                    "sampling_dinco_conf": float(sd.sampling_dinco_conf),
                    "degenerate": bool(sd.degenerate),
                    "agreement_rate": float(sd.agreement_rate),
                    "n_unique_distractors": int(sd.n_unique_distractors),
                    "candidates": list(sd.candidates),
                    "raw_samples": list(sd.raw_samples),
                    "ptrues": [float(p) for p in sd.ptrues],
                    "raw_verbal_ptrue": float(sd.raw_verbal_ptrue),
                }
            except Exception as exc:  # noqa: BLE001 — telemetry must not crash the loop
                # Sampling-DINCO is optional telemetry. Log and continue with
                # null fields rather than aborting the agent's decision step.
                print(
                    f"[sampling-dinco] WARN: compute_post_retrieval_sampling failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        return {
            "attempt": attempt,
            "support": support,
            "dependency_passages": reindex_passages(dependency_passages),
            "retrieved_passages": reindex_passages(retrieved_passages),
            "combined_passages": combined_passages,
            "evidence_chunk_ids": [str(chunk_id) for chunk_id in evidence_chunk_ids],
            "evidence_titles": [str(title) for title in evidence_titles],
            "evidence_scores": [float(score) for score in evidence_scores],
            "sampling_dinco": sampling_dinco_result,
        }

    def _retrieval_attempt(
        self,
        *,
        subquestion: str,
        dependency_passages: Sequence[hotpot_utils.Passage],
        hits: Sequence[SearchHit],
        fallback_answer: str,
        source: str = "retrieval",
        refinement: bool = False,
        previous_answer: str = "",
        previous_claims: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        retrieved_passages = [hit_to_passage(hit) for hit in hits]
        return self._score_retrieval_attempt(
            subquestion=subquestion,
            dependency_passages=dependency_passages,
            retrieved_passages=retrieved_passages,
            fallback_answer=fallback_answer,
            evidence_chunk_ids=[str(hit.row.get("chunk_id") or "") for hit in hits],
            evidence_titles=[str(hit.row.get("title") or "") for hit in hits],
            evidence_scores=[float(hit.score) for hit in hits],
            source=source,
            refinement=refinement,
            previous_answer=previous_answer,
            previous_claims=previous_claims,
        )

    def _has_weak_hop_claims(self, support: Dict[str, Any]) -> bool:
        claim_supports = [float(score) for score in list(support.get("claim_supports", []) or []) if score is not None]
        return bool(claim_supports) and any(score < self.support_min_threshold for score in claim_supports)

    def _maybe_run_strict_grounding_retry(
        self,
        *,
        subquestion: str,
        dependency_passages: Sequence[hotpot_utils.Passage],
        hits: Sequence[SearchHit],
        fallback_answer: str,
        first_attempt: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, Optional[Dict[str, Any]]]:
        if not self.retry_on_low_support:
            return first_attempt, False, None
        first_result = first_attempt["attempt"]
        if not first_result.available or not first_result.support_claims:
            return first_attempt, False, None
        if not self._has_weak_hop_claims(first_attempt["support"]):
            return first_attempt, False, None

        retry_attempt = self._retrieval_attempt(
            subquestion=subquestion,
            dependency_passages=dependency_passages,
            hits=hits,
            fallback_answer=fallback_answer,
            source="retrieval_strict_grounding_retry",
            refinement=True,
            previous_answer=first_result.answer,
            previous_claims=first_result.support_claims,
        )
        return retry_attempt, True, retry_attempt

    def _append_success_entry(
        self,
        state: planner_utils.PipelineState,
        question: str,
        node: planner_utils.SubquestionNode,
    ) -> None:
        runtime = self._get_runtime(node)
        node.facts = self._build_facts(question=question, node=node)
        node.summary = self._build_summary(question=question, node=node)
        memory_text = str(node.summary or "").strip()
        entry = {
            "type": "resolved",
            "node_id": node.id,
            "subquestion": node.subquestion,
            "resolved_subquestion": self._effective_subquestion(node),
            "retrieve": bool(node.retrieve),
            "route_taken": runtime.get("route_taken"),
            "purpose": node.purpose,
            "answer": node.answer,
            "answer_source": node.answer_source,
            "summary": node.summary,
            "facts": list(node.facts),
            "support_claims": list(node.support_claims),
            "selected_nvc": None if node.selected_nvc is None else float(node.selected_nvc),
            "selected_sc_conf": None if node.selected_sc_conf is None else float(node.selected_sc_conf),
            "selected_dinco_conf": None if node.selected_dinco_conf is None else float(node.selected_dinco_conf),
            "retrieved_passage_indices": list(node.retrieved_passage_indices),
            "retrieved_titles": list(node.retrieved_titles),
            "support": {
                "g_mean": runtime.get("online_g_mean"),
                "g_min": runtime.get("online_g_min"),
                "claim_supports": list(runtime.get("online_claim_supports", [])),
            },
            "supported": runtime.get("online_supported"),
            "pre_route": runtime.get("pre_route"),
            "memory_text": memory_text,
        }
        state.appended_entries.append(entry)
        state.planning_trace.append(
            {
                "event": "append_success",
                "node_id": node.id,
                "source": node.answer_source,
                "route_taken": runtime.get("route_taken"),
                "nvc": node.selected_nvc,
                "sc_conf": node.selected_sc_conf,
                "dinco_conf": node.selected_dinco_conf,
                "g_mean": runtime.get("online_g_mean"),
                "g_min": runtime.get("online_g_min"),
                "fact_count": len(node.facts),
            }
        )

    def _run_subquestion(
        self,
        state: planner_utils.PipelineState,
        question: str,
        node: planner_utils.SubquestionNode,
        running_id_counter: int,
        full_passages: Sequence[hotpot_utils.Passage],
    ) -> int:
        del full_passages
        node.status = "running"
        execution_subquestion = self._resolve_dependency_subquestion(state=state, question=question, node=node)
        dependency_entries = self._dependency_memory_entries(state=state, node=node)
        dependency_passages = self._entries_to_passages(dependency_entries)
        state.execution_order.append(node.id)

        pre_attempt: planner_utils.AttemptResult
        pre_route = "dependency_rewrite_question_only" if dependency_passages else "question_only"
        gate_score_pre: Optional[float]
        forced_retrieval_due_to_dinco_error = False
        dinco_failure_error: Optional[str] = None
        try:
            pre_attempt = self._question_only_attempt(execution_subquestion)
            gate_score_pre = self._gate_score(nvc=pre_attempt.nvc, dinco_conf=pre_attempt.dinco_conf)
            route_taken = choose_route(self.routing_mode, gate_score_pre, self.gate_threshold)
        except Exception as exc:
            self._emit_printbad(
                question=question,
                node=node,
                execution_subquestion=execution_subquestion,
                dependency_entries=dependency_entries,
                pre_route=pre_route,
                exc=exc,
            )
            if not _is_retryable_dinco_azure_failure(exc):
                raise
            dinco_failure_error = str(exc)
            if self.retry_example_on_dinco_azure_failure and self._current_example_attempt == 0:
                raise _RetryExampleFromStart(
                    {
                        "event": "dinco_azure_failure_restart_example",
                        "question_id": self._current_question_id,
                        "node_id": node.id,
                        "resolved_subquestion": execution_subquestion,
                        "pre_route": pre_route,
                        "example_attempt_index": self._current_example_attempt,
                        "error": dinco_failure_error,
                    }
                ) from exc
            if not self.force_retrieval_on_repeat_dinco_azure_failure:
                raise
            forced_retrieval_due_to_dinco_error = True
            pre_attempt = self._make_dinco_azure_fallback_attempt(dinco_failure_error)
            gate_score_pre = None
            route_taken = "retrieve"
            state.planning_trace.append(
                {
                    "event": "dinco_azure_failure_force_retrieval",
                    "question_id": self._current_question_id,
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "pre_route": pre_route,
                    "example_attempt_index": self._current_example_attempt,
                    "error": dinco_failure_error,
                }
            )
        if not dependency_passages and self.root_subquestion_policy == "always_retrieve":
            route_taken = "retrieve"

        all_hits = (
            search_passages(self.index, self._current_question_id, execution_subquestion, top_k=self.audit_top_k)
            if route_taken == "retrieve"
            else []
        )

        self._set_runtime(
            node,
            execution_subquestion=execution_subquestion,
            planner_retrieve_hint=bool(node.retrieve),
            pre_route=pre_route,
            route_taken=route_taken,
            pre_answer=pre_attempt.answer,
            pre_nvc=pre_attempt.nvc,
            pre_sc_conf=pre_attempt.sc_conf,
            pre_dinco_conf=pre_attempt.dinco_conf,
            gate_score_pre=gate_score_pre,
            dependency_memory_count=len(dependency_passages),
            dependency_memory_titles=[p.title for p in dependency_passages],
            retrieved_chunk_ids_stage1=[],
            retrieved_titles_stage1=[],
            retrieved_scores_stage1=[],
            retrieved_chunk_ids_online=[],
            retrieved_titles_online=[],
            retrieved_scores_online=[],
            online_claims=[],
            online_g_mean=None,
            online_g_min=None,
            online_claim_supports=[],
            online_supported=None,
            grounding_skipped=None,
            grounding_mode=None,
            retry_used=False,
            strict_grounding_retry_used=False,
            strict_grounding_retry_triggered_by_weak_claim=False,
            strict_retry_answer="",
            strict_retry_claims=[],
            strict_retry_g_mean=None,
            strict_retry_g_min=None,
            strict_retry_claim_supports=[],
            root_closed_book_commit=False,
            dinco_azure_failure_forced_retrieval=forced_retrieval_due_to_dinco_error,
            dinco_azure_failure_error=dinco_failure_error,
            example_attempt_index=self._current_example_attempt,
        )

        state.planning_trace.append(
            {
                "event": "execute_subquestion",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "depends_on": list(node.depends_on),
                "planner_retrieve_hint": bool(node.retrieve),
                "pre_route": pre_route,
                "route_taken": route_taken,
                "gate_score_pre": gate_score_pre,
                "dependency_memory_count": len(dependency_passages),
                "rewritten_from_dependencies": bool(dependency_passages),
                "route_forced_by_evidence": False,
                "route_forced_by_dinco_azure_failure": forced_retrieval_due_to_dinco_error,
                "example_attempt_index": self._current_example_attempt,
            }
        )

        if route_taken == "skip":
            skip_pass = bool(
                pre_attempt.available
                and pre_attempt.dinco_conf is not None
                and pre_attempt.dinco_conf >= self.gate_threshold
                and hotpot_utils.normalize_answer(pre_attempt.answer) != hotpot_utils.normalize_answer("insufficient evidence")
            )
            allow_closed_book_commit = True
            grounding_mode = "skipped_dependency_rewrite_closed_book" if dependency_passages else "skipped_closed_book"
            explanation = (
                "Dependent subquestion committed from a high-confidence closed-book answer after dependency rewrite."
                if dependency_passages
                else "Root subquestion committed from a high-confidence closed-book answer."
            )
            if not dependency_passages:
                if self.root_subquestion_policy == "skip_without_commit":
                    skip_pass = False
                allow_closed_book_commit = self.root_subquestion_policy == "allow_closed_book_commit"
            if skip_pass and allow_closed_book_commit:
                closed_book_attempt = planner_utils.AttemptResult(
                    answer=pre_attempt.answer,
                    raw_answer=pre_attempt.raw_answer,
                    nvc=pre_attempt.nvc,
                    sc_conf=pre_attempt.sc_conf,
                    dinco_conf=pre_attempt.dinco_conf,
                    source="closed_book_commit",
                    support_claims=list(pre_attempt.support_claims),
                    explanation=explanation,
                    dinco_candidates=list(pre_attempt.dinco_candidates),
                    dinco_ptrues=list(pre_attempt.dinco_ptrues),
                    available=True,
                )
                self._set_runtime(
                    node,
                    online_claims=list(closed_book_attempt.support_claims),
                    grounding_skipped=True,
                    grounding_mode=grounding_mode,
                    root_closed_book_commit=not dependency_passages,
                )
                state.planning_trace.append(
                    {
                        "event": "subquestion_scored",
                        "node_id": node.id,
                        "resolved_subquestion": execution_subquestion,
                        "source": "closed_book_commit",
                        "answer": closed_book_attempt.answer,
                        "nvc": closed_book_attempt.nvc,
                        "sc_conf": closed_book_attempt.sc_conf,
                        "dinco_conf": closed_book_attempt.dinco_conf,
                        "g_mean": None,
                        "g_min": None,
                        "grounding_skipped": True,
                        "grounding_mode": grounding_mode,
                        "rewritten_from_dependencies": bool(dependency_passages),
                        "passed": True,
                    }
                )
                self._commit_success(node=node, attempt=closed_book_attempt, retrieved_titles=[])
                self._append_success_entry(state=state, question=question, node=node)
                return running_id_counter

            state.planning_trace.append(
                {
                    "event": "subquestion_scored",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "source": "question_only",
                    "answer": pre_attempt.answer,
                    "nvc": pre_attempt.nvc,
                    "sc_conf": pre_attempt.sc_conf,
                    "dinco_conf": pre_attempt.dinco_conf,
                    "rewritten_from_dependencies": bool(dependency_passages),
                    "passed": False,
                }
            )
            return self._decompose_node(
                state=state,
                question=question,
                node=node,
                running_id_counter=running_id_counter,
            )

        initial_hits = all_hits[: self.retrieval_top_k]
        retrieval_context_passages: Sequence[hotpot_utils.Passage] = dependency_passages
        stage1 = self._retrieval_attempt(
            subquestion=execution_subquestion,
            dependency_passages=retrieval_context_passages,
            hits=initial_hits,
            fallback_answer=pre_attempt.answer,
        )

        self._set_runtime(
            node,
            retrieved_chunk_ids_stage1=[str(hit.row.get("chunk_id") or "") for hit in initial_hits],
            retrieved_titles_stage1=[str(hit.row.get("title") or "") for hit in initial_hits],
            retrieved_scores_stage1=[float(hit.score) for hit in initial_hits],
        )

        selected, retry_used, strict_retry = self._maybe_run_strict_grounding_retry(
            subquestion=execution_subquestion,
            dependency_passages=retrieval_context_passages,
            hits=initial_hits,
            fallback_answer=pre_attempt.answer,
            first_attempt=stage1,
        )
        if strict_retry is not None:
            strict_attempt = strict_retry["attempt"]
            strict_support = strict_retry["support"]
            strict_supported = is_supported(strict_support, self.support_mean_threshold, self.support_min_threshold)
            self._set_runtime(
                node,
                strict_grounding_retry_used=True,
                strict_grounding_retry_triggered_by_weak_claim=True,
                strict_retry_answer=strict_attempt.answer,
                strict_retry_claims=list(strict_attempt.support_claims),
                strict_retry_g_mean=float(strict_support["g_mean"]),
                strict_retry_g_min=float(strict_support["g_min"]),
                strict_retry_claim_supports=list(strict_support["claim_supports"]),
            )
            state.planning_trace.append(
                {
                    "event": "retrieval_strict_grounding_retry",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "trigger": "weak_hop_claim",
                    "previous_answer": stage1["attempt"].answer,
                    "previous_claims": list(stage1["attempt"].support_claims),
                    "previous_claim_supports": list(stage1["support"].get("claim_supports", [])),
                    "answer": strict_attempt.answer,
                    "g_mean": float(strict_support["g_mean"]),
                    "g_min": float(strict_support["g_min"]),
                    "claim_supports": list(strict_support["claim_supports"]),
                    "passed": bool(
                        strict_attempt.available
                        and strict_supported
                        and hotpot_utils.normalize_answer(strict_attempt.answer)
                        != hotpot_utils.normalize_answer("insufficient evidence")
                    ),
                }
            )

        selected_attempt = selected["attempt"]
        selected_support = selected["support"]
        online_supported = is_supported(selected_support, self.support_mean_threshold, self.support_min_threshold)
        passed = bool(
            selected_attempt.available
            and online_supported
            and hotpot_utils.normalize_answer(selected_attempt.answer)
            != hotpot_utils.normalize_answer("insufficient evidence")
        )
        runtime_g_mean = float(selected_support["g_mean"])
        runtime_g_min = float(selected_support["g_min"])
        runtime_claim_supports = list(selected_support["claim_supports"])
        grounding_skipped = False
        if retry_used:
            grounding_mode = (
                "retrieval_strict_grounding_retry_with_dependency_memory"
                if dependency_passages
                else "retrieval_strict_grounding_retry"
            )
        else:
            grounding_mode = "retrieval_with_dependency_memory" if dependency_passages else "retrieval_only_gate"

        self._set_runtime(
            node,
            retry_used=bool(retry_used),
            retrieved_chunk_ids_online=list(selected.get("evidence_chunk_ids", [])),
            retrieved_titles_online=list(selected.get("evidence_titles", [])),
            retrieved_scores_online=list(selected.get("evidence_scores", [])),
            online_claims=list(selected_attempt.support_claims),
            online_g_mean=runtime_g_mean,
            online_g_min=runtime_g_min,
            online_claim_supports=runtime_claim_supports,
            online_supported=bool(online_supported),
            grounding_skipped=grounding_skipped,
            grounding_mode=grounding_mode,
        )

        state.planning_trace.append(
            {
                "event": "subquestion_scored",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "source": selected_attempt.source,
                "answer": selected_attempt.answer,
                "nvc": selected_attempt.nvc,
                "sc_conf": selected_attempt.sc_conf,
                "dinco_conf": selected_attempt.dinco_conf,
                "g_mean": runtime_g_mean,
                "g_min": runtime_g_min,
                "grounding_mode": grounding_mode,
                "used_dinco": False,
                "retry_used": bool(retry_used),
                "passed": passed,
            }
        )

        if passed:
            repaired, running_id_counter = self._maybe_add_bridge_repair_child(
                state=state,
                question=question,
                node=node,
                execution_subquestion=execution_subquestion,
                attempt=selected_attempt,
                running_id_counter=running_id_counter,
            )
            if repaired:
                return running_id_counter
            self._commit_success(node=node, attempt=selected_attempt, retrieved_titles=selected_attempt.retrieved_titles)
            self._append_success_entry(state=state, question=question, node=node)
            return running_id_counter

        return self._decompose_node(
            state=state,
            question=question,
            node=node,
            running_id_counter=running_id_counter,
        )

    def _run_final_answer(self, state: planner_utils.PipelineState, question: str) -> Dict[str, Any]:
        memory_passages = self._entries_to_passages(state.appended_entries)
        memory_passages = reindex_passages(memory_passages)
        attempt = self._answer_with_passages(
            question=question,
            passages=memory_passages,
            source="memory_final",
            unavailable_explanation="No appended memory was available for the final answer.",
            use_dinco=False,
            reasoning_effort=self.final_answer_reasoning_effort,
        )
        pred_answer = attempt.answer if attempt.available else "insufficient evidence"
        pred_raw = attempt.raw_answer if attempt.available else pred_answer
        final_claims = select_claims_for_attempt(
            qwen_model=self.qwen_model,
            question=question,
            answer=pred_answer,
            support_claims=attempt.support_claims if attempt.available else [],
        )
        final_support = skipped_support_result("MiniCheck grounding is skipped for memory-only final answers.")
        final_supported = None

        state.planning_trace.append(
            {
                "event": "final_memory_answered",
                "answer": pred_answer if attempt.available else "",
                "nvc": attempt.nvc,
                "sc_conf": attempt.sc_conf,
                "dinco_conf": attempt.dinco_conf,
                "g_mean": None,
                "g_min": None,
                "supported": None,
                "grounding_skipped": True,
                "grounding_mode": "skipped_memory_only_final",
                "used_dinco": False,
                "memory_entry_count": len(state.appended_entries),
            }
        )

        return {
            "pred_answer": pred_answer or "insufficient evidence",
            "pred_answer_raw": pred_raw or pred_answer or "insufficient evidence",
            "nvc": float(attempt.nvc or 0.0),
            "sc_conf": float(attempt.sc_conf or 0.0),
            "dinco_conf": float(attempt.dinco_conf or 0.0),
            "dinco_candidates": list(attempt.dinco_candidates),
            "dinco_ptrues": list(attempt.dinco_ptrues),
            "final_answer_source": "memory_only_final_stage" if memory_passages else "insufficient_evidence",
            "final_grounding_skipped": True,
            "final_answer_debug": {
                "mode": "memory_only_final_stage",
                "memory_entry_count": len(state.appended_entries),
                "selected_answer": pred_answer,
                "selected_raw_answer": pred_raw,
                "selected_nvc": attempt.nvc,
                "selected_sc_conf": attempt.sc_conf,
                "selected_dinco_conf": attempt.dinco_conf,
                "selected_source": attempt.source if attempt.available else "insufficient_evidence",
                "selected_support_claims": list(final_claims),
                "final_g_mean": None,
                "final_g_min": None,
                "final_claim_supports": list(final_support["claim_supports"]),
                "final_supported": None,
                "grounding_skipped": True,
                "grounding_mode": "skipped_memory_only_final",
                "fallback_used": not bool(attempt.available),
            },
            "final_claims": list(final_claims),
            "final_support": final_support,
            "final_supported": None,
        }

    def run_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question_id = str(example.get("_id") or example.get("id") or example.get("question_id") or "")
        question = str(example.get("question", "")).strip()
        gold_answer = str(example.get("answer", "")).strip()
        passages = hotpot_utils.build_passages(example)
        full_context = hotpot_utils.format_evidence(passages)
        self._current_question_id = question_id
        max_example_attempts = 2 if self.retry_example_on_dinco_azure_failure else 1
        carryover_trace: List[Dict[str, Any]] = []
        last_restart_error: Optional[str] = None

        for example_attempt_index in range(max_example_attempts):
            self._current_example_attempt = example_attempt_index
            state = planner_utils.PipelineState(full_context=full_context)
            if carryover_trace:
                state.planning_trace.extend(carryover_trace)
            initial_nodes = self._build_plan_nodes(state=state, question=question)
            for node in initial_nodes:
                state.nodes[node.id] = node
            state.planning_trace.append(
                {
                    "event": "plan_initialized",
                    "node_ids": [node.id for node in initial_nodes],
                    "node_count": len(initial_nodes),
                    "example_attempt_index": example_attempt_index,
                }
            )

            try:
                running_id_counter = 0
                max_steps = max(4 * self.max_subquestion_nodes, 20)
                for _ in range(max_steps):
                    self._refresh_decomposed_parents(state=state)
                    next_node = self._choose_next_node(state=state)
                    if next_node is None:
                        break
                    running_id_counter = self._run_subquestion(
                        state=state,
                        question=question,
                        node=next_node,
                        running_id_counter=running_id_counter,
                        full_passages=passages,
                    )
            except _RetryExampleFromStart as exc:
                carryover_trace.append(dict(exc.trace_event))
                last_restart_error = str(exc)
                if example_attempt_index + 1 >= max_example_attempts:
                    raise
                continue

            for node in state.nodes.values():
                if node.status == "pending":
                    node.status = "failed_unresolved"
                    state.planning_trace.append(
                        {
                            "event": "node_blocked_unresolved",
                            "node_id": node.id,
                        }
                    )

            for _ in range(self.max_subquestion_depth + 2):
                self._refresh_decomposed_parents(state=state)

            final_result = self._run_final_answer(state=state, question=question)
            pred_answer = str(final_result["pred_answer"]).strip() or "insufficient evidence"
            nvc = float(final_result["nvc"])
            dinco_conf = float(final_result["dinco_conf"])
            em = hotpot_utils.exact_match(pred_answer, gold_answer)
            f1 = planner_utils.hotpot_answer_f1(pred_answer, gold_answer)
            policy_trace = self._build_policy_trace(
                planning_trace=state.planning_trace,
                pred_answer=pred_answer,
                nvc=nvc,
                dinco_conf=dinco_conf,
                appended_entry_count=len(state.appended_entries),
            )

            appended_context_text = "\n\n".join(
                str(entry.get("memory_text", "")).strip()
                for entry in state.appended_entries
                if str(entry.get("memory_text", "")).strip()
            )

            subquestion_graph = {}
            for node_id, node in state.nodes.items():
                runtime = self._get_runtime(node)
                subquestion_graph[node_id] = {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "subquestion": node.subquestion,
                    "resolved_subquestion": self._effective_subquestion(node),
                    "depends_on": list(node.depends_on),
                    "retrieve": bool(node.retrieve),
                    "purpose": node.purpose,
                    "status": node.status,
                    "answer": node.answer,
                    "raw_answer": node.raw_answer,
                    "explanation": node.explanation,
                    "support_claims": list(node.support_claims),
                    "facts": list(node.facts),
                    "summary": node.summary,
                    "answer_source": node.answer_source,
                    "selected_nvc": node.selected_nvc,
                    "selected_sc_conf": node.selected_sc_conf,
                    "selected_dinco_conf": node.selected_dinco_conf,
                    "dinco_candidates": list(node.dinco_candidates),
                    "dinco_ptrues": list(node.dinco_ptrues),
                    "retrieved_passage_indices": list(node.retrieved_passage_indices),
                    "retrieved_titles": list(node.retrieved_titles),
                    "resolution_note": node.resolution_note,
                    "calibrated_retrieval": runtime,
                }

            resolved_nodes = [node for node in state.nodes.values() if node.status in planner_utils.SUCCESS_STATUSES]
            decomposed_count = sum(
                1 for event in state.planning_trace if str(event.get("event", "")) == "decompose_success"
            )
            runtime_rows = [self._get_runtime(node) for node in state.nodes.values()]
            route_hist = summarize_counts(str(runtime.get("route_taken") or "unknown") for runtime in runtime_rows)
            answer_source_hist = summarize_counts(
                str(node.answer_source or "unresolved") for node in state.nodes.values()
            )
            root_closed_book_commit_count = sum(
                1 for runtime in runtime_rows if bool(runtime.get("root_closed_book_commit"))
            )
            retrieved_nodes = sum(1 for runtime in runtime_rows if str(runtime.get("route_taken") or "") == "retrieve")
            supported_nodes = sum(1 for runtime in runtime_rows if runtime.get("online_supported") is True)
            unsupported_nodes = sum(1 for runtime in runtime_rows if runtime.get("online_supported") is False)

            return {
                "id": question_id,
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "pred_answer_raw": final_result["pred_answer_raw"],
                "em": em,
                "f1": f1,
                "nvc": nvc,
                "sc_conf": float(final_result["sc_conf"]),
                "dinco_conf": dinco_conf,
                "dinco_candidates": list(final_result["dinco_candidates"]),
                "dinco_ptrues": list(final_result["dinco_ptrues"]),
                "final_conf": dinco_conf,
                "final_claims": list(final_result["final_claims"]),
                "final_g_mean": optional_float(final_result["final_support"]["g_mean"]),
                "final_g_min": optional_float(final_result["final_support"]["g_min"]),
                "final_claim_supports": list(final_result["final_support"]["claim_supports"]),
                "final_supported": final_result["final_supported"],
                "final_grounding_skipped": bool(final_result.get("final_grounding_skipped", False)),
                "subquestion_graph": subquestion_graph,
                "subgoal_graph": subquestion_graph,
                "planning_trace": state.planning_trace,
                "policy_trace": policy_trace,
                "execution_order": list(state.execution_order),
                "hops_used": len(state.execution_order),
                "appended_context_entries": state.appended_entries,
                "appended_context_text": appended_context_text,
                "final_answer_source": final_result["final_answer_source"],
                "final_answer_debug": final_result["final_answer_debug"],
                "subgoal_stats": {
                    "total_nodes": len(state.nodes),
                    "resolved_nodes": len(resolved_nodes),
                    "decomposed_nodes": decomposed_count,
                    "appended_entries": len(state.appended_entries),
                    "retrieved_nodes": retrieved_nodes,
                    "supported_nodes": supported_nodes,
                    "unsupported_nodes": unsupported_nodes,
                    "root_closed_book_commits": root_closed_book_commit_count,
                    "route_histogram": route_hist,
                    "answer_source_histogram": answer_source_hist,
                },
            }

        raise RuntimeError(
            f"Example retry loop exhausted for question_id={question_id}: {last_restart_error or 'unknown error'}"
        )


def build_summary(records: Sequence[Dict[str, Any]], args: argparse.Namespace, output_jsonl: Path) -> Dict[str, Any]:
    grounded_finals = [
        record
        for record in records
        if not bool(record.get("final_grounding_skipped", False)) and record.get("final_g_mean") is not None
    ]
    final_supported_values = [
        1.0 if record.get("final_supported") is True else 0.0
        for record in records
        if record.get("final_supported") is not None
    ]
    subgoal_stats = [record.get("subgoal_stats", {}) or {} for record in records]
    return {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "limit": normalize_limit(args.limit),
        "indexed_pool_limit": normalize_limit(getattr(args, "indexed_pool_limit", None)),
        "example_id": args.example_id,
        "index_dir": args.index_dir,
        "routing_mode": args.routing_mode,
        "gate_on": args.gate_on,
        "gate_threshold": args.gate_threshold,
        "retrieval_top_k": args.retrieval_top_k,
        "audit_top_k": args.audit_top_k,
        "retry_on_low_support": args.retry_on_low_support,
        "retry_extra_top_k": args.retry_extra_top_k,
        "root_subquestion_policy": args.root_subquestion_policy,
        "noground": bool(getattr(args, "noground", False)),
        "dry_run": args.dry_run,
        "num_examples": len(records),
        "mean_em": float(np.mean([float(record.get("em", 0.0)) for record in records])) if records else 0.0,
        "mean_f1": float(np.mean([float(record.get("f1", 0.0)) for record in records])) if records else 0.0,
        "mean_final_conf": float(np.mean([float(record.get("final_conf", 0.0)) for record in records])) if records else 0.0,
        "mean_final_g_mean": (
            float(np.mean([float(record.get("final_g_mean", 0.0)) for record in grounded_finals]))
            if grounded_finals
            else None
        ),
        "final_supported_rate": float(np.mean(final_supported_values)) if final_supported_values else None,
        "final_grounded_rate": (
            float(np.mean([0.0 if bool(record.get("final_grounding_skipped", False)) else 1.0 for record in records]))
            if records
            else 0.0
        ),
        "avg_total_nodes": float(np.mean([float(stats.get("total_nodes", 0.0)) for stats in subgoal_stats]))
        if subgoal_stats
        else 0.0,
        "avg_resolved_nodes": float(np.mean([float(stats.get("resolved_nodes", 0.0)) for stats in subgoal_stats]))
        if subgoal_stats
        else 0.0,
        "avg_appended_entries": float(np.mean([float(stats.get("appended_entries", 0.0)) for stats in subgoal_stats]))
        if subgoal_stats
        else 0.0,
        "avg_retrieved_nodes": float(np.mean([float(stats.get("retrieved_nodes", 0.0)) for stats in subgoal_stats]))
        if subgoal_stats
        else 0.0,
        "avg_root_closed_book_commits": float(
            np.mean([float(stats.get("root_closed_book_commits", 0.0)) for stats in subgoal_stats])
        )
        if subgoal_stats
        else 0.0,
        "output_jsonl": str(output_jsonl),
    }


def main() -> None:
    args = parse_args()
    hotpot_utils.seed_everything(args.seed)

    limit = normalize_limit(args.limit)
    output_jsonl, summary_json = default_output_paths(args, limit=limit)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args, limit=limit)
    index = BM25Index.load(Path(args.index_dir))

    if args.dry_run:
        qwen_model = hotpot_utils.MockQwenModel()
        planner = planner_utils.MockQwenPlannerModel()
        dinco = hotpot_utils.MockDincoCalibrator()
        grounder = hotpot_utils.MockMiniCheckGrounder()
    else:
        qwen_model = hotpot_utils.QwenDincoModel(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            device_map=args.generator_device_map,
            dtype=args.generator_dtype,
        )
        planner = planner_utils.QwenPlannerModel(
            qwen_model=qwen_model,
            max_new_tokens=args.planner_max_new_tokens,
            max_retries=args.max_retries,
        )
        dinco = hotpot_utils.DincoCalibrator(
            qwen_model=qwen_model,
            cache_dir=args.cache_dir,
            n_sc_samples=args.n_sc_samples,
            sc_match_threshold=args.sc_match_threshold,
        )
        grounder = hotpot_utils.MiniCheckGrounder(
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.minicheck_tensor_parallel_size,
            max_model_len=args.minicheck_max_model_len,
            model_name=args.minicheck_model_name,
            allow_cpu_fallback=args.allow_minicheck_cpu_fallback,
            cpu_fallback_model_name=args.minicheck_cpu_fallback_model_name,
            gpu_memory_utilization=args.minicheck_gpu_memory_utilization,
        )

    runner = CalibratedPlannerMemoryRunner(
        planner=planner,
        qwen_model=qwen_model,
        dinco=dinco,
        grounder=grounder,
        index=index,
        gate_on=args.gate_on,
        gate_threshold=args.gate_threshold,
        support_mean_threshold=args.support_mean_threshold,
        support_min_threshold=args.support_min_threshold,
        routing_mode=args.routing_mode,
        retry_on_low_support=args.retry_on_low_support,
        retry_extra_top_k=args.retry_extra_top_k,
        audit_top_k=args.audit_top_k,
        root_subquestion_policy=args.root_subquestion_policy,
        max_initial_subquestions=args.max_initial_subquestions,
        max_subquestion_depth=args.max_subquestion_depth,
        max_subquestion_nodes=args.max_subquestion_nodes,
        retrieval_top_k=args.retrieval_top_k,
        n_distractors=args.n_distractors,
        printbad=args.printbad,
        noground=args.noground,
    )

    records: List[Dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as writer:
        for example in examples:
            row = runner.run_example(example)
            writer.write(json.dumps(row, ensure_ascii=True) + "\n")
            writer.flush()
            records.append(row)

    summary = build_summary(records, args=args, output_jsonl=output_jsonl)
    write_json(summary, summary_json)
    print(f"Wrote {len(records)} records to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
