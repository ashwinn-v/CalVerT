#!/usr/bin/env python3
"""
Qwen-only planner-memory multihop QA pipeline for HotpotQA.

High-level behavior:
- Qwen plans disambiguated subquestions and decides whether each step should retrieve.
- Successful subquestion answers are converted into atomic facts and appended to memory.
- The final answer is produced from the original question plus appended memory only.
- DINCO confidence is used to gate subquestion commits and to score the final answer.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from telemetry_agent.runners._hotpot_utils import (
    DincoCalibrator,
    DincoResult,
    MockDincoCalibrator,
    MockQwenModel,
    Passage,
    QwenDincoModel,
    build_passages,
    exact_match,
    extract_json_dict,
    format_evidence,
    normalize_answer,
    rank_passages,
    resolve_json_output_path,
    seed_everything,
)


PLAN_PROMPT = """
You are planning a memory-building multi-hop QA strategy for a Hotpot-style question.

Return STRICT JSON only with schema:
{{
  "subgoals": [
    {{
      "id": "sg1",
      "subquestion": "atomic subquestion written only from the original question text and/or dependency references",
      "depends_on": [],
      "retrieve": true,
      "purpose": "atomic_fact"
    }}
  ]
}}

Rules:
- Use 1 to {max_initial_subquestions} subgoals.
- First infer the question's latent reasoning family internally, then emit only the minimum subgoal graph needed for that family.
- Reasoning families include bridge lookup, attribute composition, boolean composition, comparison, temporal relation, ordinal/ranking, count/aggregation, set intersection/union, and multi-entity relation lookup.
- ids must be unique.
- depends_on must reference only ids in this output.
- Each subquestion must be atomic and should preserve only information explicitly present in the original question.
- Each subquestion must target exactly one latent slot: entity, attribute, relation, date, number, location, title, or boolean/comparison result.
- Prefer DINCO-friendly subquestions: short natural questions that ask for exactly one answer slot.
- Prefer canonical slot-filling forms such as "Who ...?", "What country ...?", "When ...?", "What year ...?", "Where ...?", "What city ...?", "What neighborhood ...?", "How many ...?", and "What position ...?" when they fit the question.
- Prefer direct slot lookup over verbose identification phrasing when both obey the evidence constraints.
- Avoid "Identify ...", "Name ...", long appositives, nested relative clauses, and multi-clause descriptions when a shorter slot question can express the same missing fact.
- Do not add names, occupations, dates, nationalities, aliases, locations, appositives, or other factual descriptors unless they are literally written in the question.
- Do not use outside knowledge to resolve a descriptive mention in the question to a real-world entity.
- If the question says "the woman who portrayed Corliss Archer in the film Kiss and Tell", do not rewrite it as "Shirley Temple".
- If the question says only "Scott Derrickson", do not rewrite it as "Scott Derrickson, the American film director born in 1966".
- Minimal restatement is better than enriched disambiguation.
- Avoid pronouns or vague references unless the referent is named in the same subquestion or explicitly referenced through depends_on.
- If a node has depends_on, it may refer abstractly to "the person identified in sg1" or "the value from sg1", but it must not inject new facts not present in the question. Keep such dependency-based wording short and ask only for the remaining slot.
- If the question needs combining multiple earlier answers, include a final composition subgoal that depends on the relevant factual subgoals and combines only their outputs.
- Composition-required cues include same/different, both/either/neither, yes/no comparison, older/younger, before/after/during, earlier/later, higher/lower, more/less, rank/order, highest/lowest, and count/how many when multiple earlier facts must be combined.
- A composition subgoal should usually have retrieve=false and purpose comparison_fact or reasoning when it can be answered from dependency memory.
- If the question is a bridge / fact-lookup question, avoid an unnecessary composition subgoal. After identifying the missing bridge entity or value, ask the final slot directly instead of adding an extra combine node.
- For count or set questions, decompose into the prerequisite membership/fact questions and then one short count/set-composition node.
- For temporal or ranking questions, decompose into the prerequisite date/number/rank questions and then one short temporal/ranking composition node.
- For hidden intermediate entities, first identify the missing entity, then ask the target slot directly instead of repeating the full descriptive chain.
- Use retrieve=true when the subquestion needs external evidence from the Hotpot context.
- Use retrieve=false only when the answer should be derivable from prerequisite memory produced by earlier subgoals.
- If a node has depends_on, assume those prerequisite answers will already be available as appended memory.
- Prefer planning prerequisite fact questions first, then bridge/composition questions later.
- Do not make a graph-internal final-answer node. The final answer is produced after memory is built.
- Subgoals must be non-overlapping and should not restate the same missing fact.
- purpose should be a short label such as atomic_fact, bridge_fact, disambiguation, comparison_fact, or reasoning.
- No markdown, no extra keys.

Lightweight reasoning-family hints (may be incomplete):
{reasoning_family_hints_json}

Operator hints (may be incomplete):
{operator_hints_json}

Expected final answer type hint:
{expected_answer_type_hint}

Latent slot hints (may be incomplete):
{slot_hints_json}

Question:
{question}
""".strip()


DECOMPOSE_PROMPT = """
A subquestion failed or remained low-confidence. Decompose it into smaller disambiguated questions.

Return STRICT JSON only with schema:
{{
  "replacement_subgoals": [
    {{
      "id": "{parent_id}_a",
      "subquestion": "smaller atomic subquestion written only from the original question text, failed subquestion text, resolved dependency context, and/or dependency references",
      "depends_on": [],
      "retrieve": true,
      "purpose": "atomic_fact"
    }}
  ]
}}

Rules:
- Return 1 to 3 replacement subgoals.
- First identify which latent slot failed: entity resolution, attribute lookup, relation lookup, temporal scope, rank/order input, aggregation input, or composition operator input.
- ids must be unique within this response.
- depends_on may reference ids in this response and already-resolved dependency ids listed below.
- Each replacement subquestion must isolate one missing fact, bridge fact, or comparison component without adding outside knowledge.
- Each replacement must target exactly one smaller unresolved slot instead of paraphrasing the whole failed question.
- Prefer DINCO-friendly replacements: short natural questions that ask for exactly one answer slot.
- Prefer canonical slot-filling forms such as "Who ...?", "What country ...?", "When ...?", "What year ...?", "Where ...?", "What city ...?", "How many ...?", and "What position ...?" when they fit the missing fact.
- Prefer direct slot lookup over verbose identification phrasing when both obey the evidence constraints.
- Avoid "Identify ...", "Name ...", long appositives, nested relative clauses, and multi-clause descriptions when a shorter slot question can express the same missing fact.
- Use only wording from the original question, the failed subquestion, resolved dependency context, and dependency references.
- Do not add names, occupations, dates, nationalities, aliases, locations, appositives, or other factual descriptors unless they are literally present in the original question or failed subquestion.
- Do not resolve a descriptive mention to a real-world entity using outside knowledge.
- If the failed subquestion says "the woman who portrayed Corliss Archer in the film Kiss and Tell", do not rewrite it as "Shirley Temple".
- Minimal restatement is better than enriched disambiguation.
- Resolved dependencies listed below are already solved and available as memory. Treat them as known facts.
- Never emit a replacement child that re-asks, paraphrases, or re-resolves a resolved dependency subgoal.
- Decompose only the still-missing part of the failed subquestion after using the resolved dependency answers.
- If a resolved dependency already identifies the needed bridge entity, keep that dependency and ask only for the remaining missing slot/value.
- Prefer one child over two when only one unresolved fact remains after using dependency memory.
- If the failed node is comparison / boolean composition / temporal relation / order / count / set composition, preserve or reintroduce a final composition child that depends on earlier factual children.
- If the failed node is bridge / fact lookup, decompose it into bridge-resolution and target-slot subquestions without adding an unnecessary composition child.
- Composition-required cues include same/different, both/either/neither, yes/no comparison, older/younger, before/after/during, earlier/later, higher/lower, more/less, rank/order, highest/lowest, and count/how many when multiple earlier facts must be combined.
- A composition child should usually have retrieve=false and purpose comparison_fact or reasoning when it can be answered from dependency memory.
- Use retrieve=true when that node should read Hotpot evidence.
- Use retrieve=false only when that node should be answerable from its dependency memory.
- Prefer prerequisite atomic facts first, then dependent bridge/composition questions.
- Do not merely paraphrase the failed subquestion.
- Do not repeat the failed subquestion verbatim.
- Example with resolved dependency:
  resolved dependency: sg1 -> "Who is the woman who portrayed Corliss Archer in the film Kiss and Tell?" -> "Shirley Temple"
  bad replacement: "Who portrayed Corliss Archer in the film Kiss and Tell?"
  good replacement: "What government position was held by the person identified in sg1?" with depends_on ["sg1"]
- Example for a bridge lookup:
  bad replacement: "Identify the director of Big Stone Gap and the New York city where that person is based."
  good replacements:
  1. "Who is the director of Big Stone Gap?"
  2. "What New York city is the person identified in sg1 based in?" with depends_on ["sg1"]
- Example for a comparison:
  bad replacement: "Who is older based on Annie Morton's birth date and Terry Richardson's birth date and all related evidence?"
  good replacement: "Who is older, Annie Morton or Terry Richardson, based on sg1 and sg2?" with depends_on ["sg1", "sg2"]
- Example for a count/set question:
  bad replacement: "How many of the earlier answers satisfy the original question?"
  good replacements:
  1. ask the prerequisite membership/fact slots
  2. "How many of sg1, sg2, and sg3 satisfy the question?" with depends_on ["sg1", "sg2", "sg3"]
- Keep every subquestion atomic and as self-contained as possible without injecting facts not stated in the question.
- No markdown, no extra keys.

Failed-node reasoning-family hints (may be incomplete):
{reasoning_family_hints_json}

Operator hints (may be incomplete):
{operator_hints_json}

Expected answer type hint:
{expected_answer_type_hint}

Latent slot hints (may be incomplete):
{slot_hints_json}

Original question:
{question}

Failed subquestion id: {parent_id}
Failed subquestion text: {failed_subquestion}
Failed rewritten subquestion used at execution time: {failed_resolved_subquestion}

Resolved dependency ids available for reuse:
{resolved_dependency_ids_json}

Resolved dependency context (already solved; do not re-ask):
{resolved_dependency_context_json}
""".strip()


FACT_PROMPT = """
Convert a solved subquestion into reusable appended-memory facts.

Return STRICT JSON only with schema:
{{
  "facts": [
    "atomic decontextualized fact"
  ]
}}

Rules:
- Use only the provided subquestion, answer, explanation, and support claims.
- Facts must be atomic, self-contained, and explicitly name entities.
- Do not use pronouns if the entity can be named directly.
- Preserve uncertainty only if the answer itself is uncertain.
- Return 1 to 4 facts.
- Do not add unsupported information.
- No markdown, no extra keys.

Original question:
{question}

Subquestion:
{subquestion}

Answer:
{answer}

Explanation:
{explanation}

Support claims:
{support_claims_json}
""".strip()


MEMORY_SUMMARY_PROMPT = """
Produce a concise reusable memory summary.

Return STRICT JSON only with schema:
{{
  "summary": "short factual summary"
}}

Rules:
- Use only the provided answer and facts.
- Keep the summary concise and factual.
- Do not add unsupported information.
- No markdown, no extra keys.

Original question:
{question}

Subquestion:
{subquestion}

Answer:
{answer}

Facts:
{facts_json}
""".strip()


DEPENDENCY_REWRITE_PROMPT = """
Rewrite a dependent subquestion by filling in resolved answers from earlier subquestions.

Return STRICT JSON only with schema:
{{
  "resolved_subquestion": "filled-in natural-language subquestion"
}}

Rules:
- Preserve the meaning and answer target of the original dependent subquestion.
- Preserve the original operator or reasoning relation when present: same/different, both/either/neither, older/younger, before/after/during, higher/lower, more/less, rank/order, and count/how many.
- Fill only the unresolved slot left after the dependencies are known; do not turn a composition question into a bridge lookup or vice versa.
- Use the resolved dependency answers directly when they can replace references like "the person identified in sg1" or "the value from sg1".
- Use only the original question, the dependent subquestion, and the resolved dependency context below.
- Do not add unsupported facts beyond the already-resolved dependency answers.
- If no rewrite is needed, return the original subquestion unchanged.
- Keep the rewritten subquestion concise, natural, and answer-slot targeted.
- If preserving the operator would make the rewrite awkward, prefer the original subquestion unchanged over a broadened paraphrase.
- No markdown, no extra keys.

Operator hints (may be incomplete):
{operator_hints_json}

Expected answer type hint:
{expected_answer_type_hint}

Latent slot hints (may be incomplete):
{slot_hints_json}

Original question:
{question}

Dependent subquestion:
{subquestion}

Resolved dependency ids:
{resolved_dependency_ids_json}

Resolved dependency context:
{resolved_dependency_context_json}
""".strip()


BRIDGE_REPAIR_PROMPT = """
Decide whether a supported subquestion answer should be accepted directly or used as a bridge entity for one more slot lookup.

Return STRICT JSON only with schema:
{{
  "action": "accept" | "bridge_repair",
  "replacement_subgoals": [
    {{
      "id": "sg_bridge",
      "subquestion": "one child subquestion with the supported answer inlined directly",
      "depends_on": [],
      "retrieve": true,
      "purpose": "bridge_fact"
    }}
  ]
}}

Rules:
- If the supported answer already directly answers the resolved subquestion, return "accept" and an empty replacement_subgoals list.
- Use "bridge_repair" only when the supported answer appears to identify an intermediate entity but the question still asks for a remaining slot about that entity.
- For "bridge_repair", return exactly one child subgoal.
- Inline the supported answer directly in the child subquestion. Do not use sg ids or placeholders.
- The child must ask only the remaining target slot, not restate the whole original question.
- Keep the child concise and natural.
- Do not add unsupported facts beyond the question, supported answer, explanation, and support claims below.
- No markdown, no extra keys.

Expected answer type hint:
{expected_answer_type_hint}

Latent slot hints:
{slot_hints_json}

Original question:
{question}

Current subquestion:
{subquestion}

Resolved subquestion:
{resolved_subquestion}

Supported answer:
{answer}

Explanation:
{explanation}

Support claims:
{support_claims_json}
""".strip()


SUCCESS_STATUSES = {"resolved", "resolved_from_children"}
TERMINAL_STATUSES = SUCCESS_STATUSES | {"failed_unresolved", "failed_invalid"}


@dataclass
class AttemptResult:
    answer: str
    raw_answer: str
    nvc: Optional[float]
    sc_conf: Optional[float]
    dinco_conf: Optional[float]
    source: str
    support_claims: List[str] = field(default_factory=list)
    explanation: str = ""
    dinco_candidates: List[str] = field(default_factory=list)
    dinco_ptrues: List[float] = field(default_factory=list)
    available: bool = True
    retrieved_passage_indices: List[int] = field(default_factory=list)
    retrieved_titles: List[str] = field(default_factory=list)
    # Post-retrieval sampling-DINCO telemetry. Populated only when
    # `enable_sampling_dinco_telemetry` is True on the runner. None for
    # closed-book attempts and any retrieve attempt where the signal was
    # not computed.
    sampling_dinco_conf: Optional[float] = None
    sampling_dinco_degenerate: Optional[bool] = None
    sampling_dinco_agreement_rate: Optional[float] = None
    sampling_dinco_n_unique: Optional[int] = None
    sampling_distractors: List[str] = field(default_factory=list)
    sampling_ptrues: List[float] = field(default_factory=list)


@dataclass
class SubquestionNode:
    id: str
    parent_id: Optional[str]
    depth: int
    subquestion: str
    depends_on: List[str]
    retrieve: bool
    purpose: str
    status: str = "pending"
    resolved_subquestion: str = ""
    answer: str = ""
    raw_answer: str = ""
    explanation: str = ""
    support_claims: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    summary: str = ""
    answer_source: str = ""
    selected_nvc: Optional[float] = None
    selected_sc_conf: Optional[float] = None
    selected_dinco_conf: Optional[float] = None
    dinco_candidates: List[str] = field(default_factory=list)
    dinco_ptrues: List[float] = field(default_factory=list)
    retrieved_passage_indices: List[int] = field(default_factory=list)
    retrieved_titles: List[str] = field(default_factory=list)
    resolution_note: str = ""


@dataclass
class PipelineState:
    full_context: str
    nodes: Dict[str, SubquestionNode] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    appended_entries: List[Dict[str, Any]] = field(default_factory=list)
    planning_trace: List[Dict[str, Any]] = field(default_factory=list)


def _safe_json_array_str(items: Sequence[Any]) -> str:
    try:
        return json.dumps(list(items), ensure_ascii=True)
    except Exception:  # noqa: BLE001
        return "[]"


def _clean_node_id(node_id: Any, fallback: str) -> str:
    text = str(node_id).strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9_\-]", "_", text)
    return text or fallback


def _normalize_depends(depends: Any) -> List[str]:
    if not isinstance(depends, list):
        return []
    out: List[str] = []
    for item in depends:
        dep = str(item).strip()
        if dep:
            out.append(dep)
    return out


def _normalize_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(default)


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        norm = normalize_answer(text)
        if not text or not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


_NODE_REF_RE = re.compile(r"\bsg[A-Za-z0-9_-]+\b", flags=re.IGNORECASE)


def _extract_node_refs(*texts: str) -> List[str]:
    refs: List[str] = []
    seen: Set[str] = set()
    for text in texts:
        for match in _NODE_REF_RE.finditer(str(text or "")):
            ref = str(match.group(0)).strip()
            if not ref:
                continue
            key = ref.lower()
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def _parse_string_list(raw: Any) -> List[str]:
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("claim", "") or item.get("fact", "")).strip()
            else:
                text = ""
            if text:
                out.append(text)
    return _unique_keep_order(out)


_QUESTION_SIGNATURE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


def _question_signature(text: str) -> Set[str]:
    tokens = normalize_answer(text).split()
    return {
        token
        for token in tokens
        if token and token not in _QUESTION_SIGNATURE_STOPWORDS and not token.isdigit()
    }


def _looks_like_reask(candidate_subquestion: str, prior_subquestion: str) -> bool:
    cand_sig = _question_signature(candidate_subquestion)
    prior_sig = _question_signature(prior_subquestion)
    if not cand_sig or not prior_sig:
        return False
    overlap = cand_sig & prior_sig
    if not overlap:
        return False
    smaller = min(len(cand_sig), len(prior_sig))
    if smaller <= 0:
        return False
    overlap_ratio = len(overlap) / float(smaller)
    if overlap_ratio < 0.8:
        return False
    return cand_sig.issubset(prior_sig) or prior_sig.issubset(cand_sig) or len(cand_sig ^ prior_sig) <= 1


_OPERATOR_FAMILY_CUES: Dict[str, Sequence[str]] = {
    "same_different": ("same", "different"),
    "both_either_neither": ("both", "either", "neither"),
    "older_younger": ("older", "younger"),
    "before_after": ("before", "after", "earlier", "later", "during"),
    "higher_lower": ("higher", "lower"),
    "more_less": ("more", "less", "fewer"),
    "rank_order": ("rank", "ranking", "order", "highest", "lowest", "first", "last"),
    "count": ("how many", "number of", "count"),
}


def _contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    norm = normalize_answer(text)
    for phrase in phrases:
        phrase_norm = normalize_answer(str(phrase))
        if not phrase_norm:
            continue
        if " " in phrase_norm:
            if phrase_norm in norm:
                return True
            continue
        if re.search(rf"\b{re.escape(phrase_norm)}\b", norm):
            return True
    return False


def _infer_operator_hints(text: str) -> List[str]:
    hints: List[str] = []
    for name, phrases in _OPERATOR_FAMILY_CUES.items():
        if _contains_any_phrase(text, phrases):
            hints.append(name)
    return hints


def _infer_reasoning_family_hints(text: str) -> List[str]:
    hints: List[str] = []
    operator_hints = set(_infer_operator_hints(text))
    if operator_hints & {"same_different", "older_younger", "before_after", "higher_lower", "more_less"}:
        hints.append("comparison")
    if operator_hints & {"both_either_neither"}:
        hints.append("boolean_composition")
        hints.append("set_composition")
    if operator_hints & {"before_after"}:
        hints.append("temporal_relation")
    if operator_hints & {"rank_order"}:
        hints.append("ordinal_ranking")
    if operator_hints & {"count"}:
        hints.append("count_aggregation")
    norm = normalize_answer(text)
    if not hints and (
        " of the " in norm
        or " by the " in norm
        or " based on " in norm
        or " person who " in norm
        or " woman who " in norm
        or " man who " in norm
        or " the one who " in norm
    ):
        hints.append("bridge_lookup")
    if not hints:
        hints.append("attribute_lookup")
    return _unique_keep_order(hints)


def _infer_expected_answer_type_hint(text: str) -> str:
    norm = normalize_answer(text)
    first = norm.split()[:3]
    first_text = " ".join(first)
    if first and first[0] in {"is", "are", "was", "were", "did", "do", "does", "can", "could", "would", "should", "will", "has", "have", "had"}:
        return "yes_no"
    if "how many" in norm or "number of" in norm or first_text == "what number":
        return "number"
    if first_text.startswith("when") or first_text.startswith("what year") or first_text.startswith("what date"):
        return "date"
    if first_text.startswith("where") or _contains_any_phrase(norm, ("what city", "what country", "what county", "what neighborhood", "what town", "what village", "what state")):
        return "location"
    if first_text.startswith("who"):
        return "person_or_entity"
    return "entity_or_attribute"


def _infer_slot_hints(text: str) -> List[str]:
    norm = normalize_answer(text)
    hints: List[str] = []
    if _contains_any_phrase(norm, ("who", "person", "woman", "man", "director", "author", "actor", "actress", "president", "composer")):
        hints.append("entity_slot")
    if _contains_any_phrase(norm, ("when", "what year", "what date", "before", "after", "earlier", "later", "during")):
        hints.append("date_or_time_slot")
    if _contains_any_phrase(norm, ("where", "what city", "what country", "what county", "what neighborhood", "what state")):
        hints.append("location_slot")
    if _contains_any_phrase(norm, ("how many", "number of", "count", "rank", "ranking", "highest", "lowest", "more", "less", "fewer")):
        hints.append("number_or_order_slot")
    if _contains_any_phrase(norm, ("same", "different", "both", "either", "neither", "older", "younger", "before", "after")):
        hints.append("composition_operator_slot")
    if _contains_any_phrase(norm, ("who is the", "what is the name of", "which person", "which film", "which book", "which team")):
        hints.append("hidden_bridge_entity_slot")
    return _unique_keep_order(hints)


def _requires_composition_node(text: str) -> bool:
    families = set(_infer_reasoning_family_hints(text))
    return bool(
        families
        & {"comparison", "boolean_composition", "temporal_relation", "ordinal_ranking", "count_aggregation", "set_composition"}
    )


def _is_composition_purpose(purpose: str) -> bool:
    norm = normalize_answer(purpose).replace(" ", "_")
    return norm in {"comparison_fact", "reasoning", "count_aggregation", "set_composition", "temporal_relation"}


def _default_composition_purpose(text: str) -> str:
    families = set(_infer_reasoning_family_hints(text))
    if families & {"count_aggregation", "set_composition"}:
        return "reasoning"
    return "comparison_fact"


def _rewrite_type_compatible(expected: str, actual: str) -> bool:
    if expected in {"entity_or_attribute", "person_or_entity"}:
        return True
    if expected == "yes_no":
        return actual == "yes_no"
    if expected == "number":
        return actual == "number"
    if expected == "date":
        return actual in {"date", "number"}
    if expected == "location":
        return actual in {"location", "entity_or_attribute", "person_or_entity"}
    return True


def _sanitize_filename_token(raw: Any) -> str:
    token = str(raw).strip().replace("/", "_")
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("._")
    return token or "na"


def _format_name_template(name_template: str, args: argparse.Namespace) -> str:
    example_tag = _sanitize_filename_token(args.example_id) if args.example_id else "none"
    slice_tag = f"id-{example_tag}" if args.example_id else f"s{int(args.start_idx)}-n{int(args.max_examples)}"
    fmt_vars = {
        "dataset_name": _sanitize_filename_token(args.dataset_name),
        "dataset_subset": _sanitize_filename_token(args.dataset_subset),
        "split": _sanitize_filename_token(args.split),
        "slice_tag": _sanitize_filename_token(slice_tag),
        "example_tag": example_tag,
        "model": _sanitize_filename_token(args.qwen_model_name),
        "mode": "dry_run" if bool(args.dry_run) else "run",
        "seed": str(int(args.seed)),
    }
    try:
        rendered = name_template.format(**fmt_vars)
    except KeyError as exc:
        known = ",".join(sorted(fmt_vars.keys()))
        raise ValueError(f"Unknown output filename placeholder '{exc.args[0]}'. Known placeholders: {known}") from exc
    return rendered


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_jsonl:
        output_name = _format_name_template(args.output_jsonl, args)
    else:
        output_name = _format_name_template(
            "results_hotpot_planner_memory_{dataset_name}_{dataset_subset}_{split}_{slice_tag}_{mode}.jsonl",
            args,
        )
    if not output_name.endswith(".jsonl"):
        output_name = f"{Path(output_name).stem}.jsonl"

    output_path = resolve_json_output_path(output_name)

    if args.summary_json:
        summary_name = _format_name_template(args.summary_json, args)
    else:
        summary_name = f"{output_path.stem}.summary.json"
    if not summary_name.endswith(".json"):
        summary_name = f"{Path(summary_name).stem}.json"

    summary_path = resolve_json_output_path(summary_name)
    return output_path, summary_path


def _slice_dataset(
    ds: Any,
    start_idx: int,
    max_examples: int,
    example_id: Optional[str],
) -> tuple[Any, Optional[int], Optional[int]]:
    selected_start_idx: Optional[int] = None
    selected_end_idx: Optional[int] = None
    if example_id:
        if "id" not in ds.column_names:
            raise ValueError("Dataset has no 'id' column; cannot use --example_id.")
        ids = ds["id"]
        matches = [i for i, ex_id in enumerate(ids) if str(ex_id) == str(example_id)]
        if not matches:
            raise ValueError(f"example_id='{example_id}' not found in dataset.")
        selected_start_idx = int(matches[0])
        selected_end_idx = selected_start_idx + 1
        ds = ds.select([selected_start_idx])
    else:
        selected_start_idx = int(start_idx)
        selected_end_idx = min(selected_start_idx + int(max_examples), len(ds))
        if selected_start_idx >= selected_end_idx:
            raise ValueError(f"Empty slice requested: start_idx={selected_start_idx}, end_idx={selected_end_idx}")
        ds = ds.select(range(selected_start_idx, selected_end_idx))
    return ds, selected_start_idx, selected_end_idx


def hotpot_answer_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    remaining = dict(common)
    for token in gold_tokens:
        count = remaining.get(token, 0)
        if count > 0:
            overlap += 1
            remaining[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return float((2 * precision * recall) / (precision + recall))


class QwenPlannerModel:
    def __init__(self, qwen_model: Any, max_new_tokens: int, max_retries: int) -> None:
        self.qwen_model = qwen_model
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.max_retries = max(0, int(max_retries))

    def call_json(self, mode: str, prompt: str) -> Dict[str, Any]:
        strict_prompt = (
            f"{prompt}\n\n"
            "Return valid JSON only. Do not add markdown fences or any non-JSON text."
        )
        thinking_modes = {"plan"}
        enable_thinking = mode in thinking_modes
        last_raw = ""
        for _ in range(self.max_retries + 1):
            raw = self.qwen_model.generate(
                strict_prompt,
                max_new_tokens=self.max_new_tokens,
                enable_thinking=enable_thinking,
            )
            last_raw = raw
            parsed = extract_json_dict(raw)
            if parsed is not None:
                return parsed
        raise ValueError(f"{mode} returned non-JSON output: {last_raw[:300]}")


class MockQwenPlannerModel:
    def call_json(self, mode: str, prompt: str) -> Dict[str, Any]:
        del prompt
        if mode == "plan":
            return {
                "subgoals": [
                    {
                        "id": "sg1",
                        "subquestion": "What fact should be retrieved first to answer the question?",
                        "depends_on": [],
                        "retrieve": True,
                        "purpose": "atomic_fact",
                    },
                    {
                        "id": "sg2",
                        "subquestion": "What bridge fact should be inferred from the first answer?",
                        "depends_on": ["sg1"],
                        "retrieve": False,
                        "purpose": "bridge_fact",
                    },
                ]
            }
        if mode == "decompose":
            return {
                "replacement_subgoals": [
                    {
                        "id": "sgx_a",
                        "subquestion": "What narrower fact can be retrieved for the failed question?",
                        "depends_on": [],
                        "retrieve": True,
                        "purpose": "atomic_fact",
                    },
                    {
                        "id": "sgx_b",
                        "subquestion": "What follow-up can be answered from the earlier memory?",
                        "depends_on": ["sgx_a"],
                        "retrieve": False,
                        "purpose": "bridge_fact",
                    },
                ]
            }
        if mode == "facts":
            return {
                "facts": [
                    "Mock fact extracted from the solved subquestion.",
                    "Mock bridge fact is stored in appended memory.",
                ]
            }
        if mode == "memory_summary":
            return {"summary": "Mock summary for appended memory."}
        if mode == "dependency_rewrite":
            return {"resolved_subquestion": "Mock rewritten dependent question using prior answers."}
        if mode == "bridge_repair":
            return {"action": "accept", "replacement_subgoals": []}
        return {}


class QwenAnswerEngine:
    def __init__(self, qwen_model: Any, dinco: Any, n_distractors: int) -> None:
        self.qwen_model = qwen_model
        self.dinco = dinco
        self.n_distractors = max(1, int(n_distractors))

    def _clean_answer(self, question: str, answer: str) -> str:
        text = str(answer or "").strip()
        if hasattr(self.qwen_model, "shorten_answer_for_hotpot"):
            try:
                text = self.qwen_model.shorten_answer_for_hotpot(question=question, answer=text)
            except Exception:  # noqa: BLE001
                pass
        text = text.strip()
        return text or "insufficient evidence"

    def answer_with_passages(
        self,
        question: str,
        passages: Sequence[Passage],
        source: str,
        unavailable_explanation: str,
        use_dinco: bool = True,
        reasoning_effort: Optional[str] = None,
        refinement: bool = False,
        previous_answer: str = "",
        previous_claims: Optional[Sequence[str]] = None,
    ) -> AttemptResult:
        if not passages:
            return AttemptResult(
                answer="",
                raw_answer="",
                nvc=None,
                sc_conf=None,
                dinco_conf=None,
                source=source,
                support_claims=[],
                explanation=unavailable_explanation,
                dinco_candidates=[],
                dinco_ptrues=[],
                available=False,
            )

        gen_kwargs: Dict[str, Any] = {
            "question": question,
            "passages": passages,
        }
        try:
            generation_parameters = inspect.signature(self.qwen_model.generate_answer_and_claims).parameters
        except (TypeError, ValueError):
            generation_parameters = {}
        if reasoning_effort is not None and "reasoning_effort" in generation_parameters:
            gen_kwargs["reasoning_effort"] = reasoning_effort
        if refinement and "refinement" in generation_parameters:
            gen_kwargs["refinement"] = True
        if previous_answer and "previous_answer" in generation_parameters:
            gen_kwargs["previous_answer"] = previous_answer
        if previous_claims is not None and "previous_claims" in generation_parameters:
            gen_kwargs["previous_claims"] = list(previous_claims)
        gen = self.qwen_model.generate_answer_and_claims(**gen_kwargs)
        raw_answer = self._clean_answer(question=question, answer=gen.answer)
        # Retrieval grounding should operate on the hop-wise chain the model used,
        # not on a compressed final-answer claim. Fall back only for legacy outputs.
        support_claims = _unique_keep_order(list(gen.support_claims or gen.answer_support_claims))
        explanation = " ".join(support_claims[:2]) if support_claims else f"{source} answer using provided evidence."

        if not use_dinco:
            return AttemptResult(
                answer=raw_answer,
                raw_answer=raw_answer,
                nvc=None,
                sc_conf=None,
                dinco_conf=None,
                source=source,
                support_claims=support_claims,
                explanation=explanation,
                dinco_candidates=[],
                dinco_ptrues=[],
                available=True,
            )

        dinco_result: DincoResult = self.dinco.compute(
            question=question,
            answer=raw_answer,
            passages=passages,
            n_distractors=self.n_distractors,
        )
        answer = raw_answer
        if dinco_result.candidates:
            answer = self._clean_answer(question=question, answer=dinco_result.candidates[0])

        return AttemptResult(
            answer=answer,
            raw_answer=raw_answer,
            nvc=float(dinco_result.nvc),
            sc_conf=float(dinco_result.sc_conf),
            dinco_conf=float(dinco_result.final_conf),
            source=source,
            support_claims=support_claims,
            explanation=explanation,
            dinco_candidates=list(dinco_result.candidates),
            dinco_ptrues=[float(x) for x in dinco_result.ptrues],
            available=True,
        )



class PlannerMemoryRunner:
    def __init__(
        self,
        planner: Any,
        qwen_model: Any,
        subquestion_qwen_model: Optional[Any],
        dinco: Any,
        max_initial_subquestions: int,
        max_subquestion_depth: int,
        max_subquestion_nodes: int,
        confidence_threshold: float,
        retrieval_top_k: int,
        n_distractors: int,
    ) -> None:
        self.planner = planner
        self.qwen_model = qwen_model
        self.subquestion_qwen_model = qwen_model if subquestion_qwen_model is None else subquestion_qwen_model
        self.dinco = dinco
        self.max_initial_subquestions = max(1, int(max_initial_subquestions))
        self.max_subquestion_depth = max(0, int(max_subquestion_depth))
        self.max_subquestion_nodes = max(1, int(max_subquestion_nodes))
        self.confidence_threshold = float(confidence_threshold)
        self.retrieval_top_k = max(1, int(retrieval_top_k))
        self.n_distractors = max(1, int(n_distractors))
        self.answer_engine = QwenAnswerEngine(
            qwen_model=self.subquestion_qwen_model,
            dinco=dinco,
            n_distractors=n_distractors,
        )

    @staticmethod
    def _normalized_question_key(text: str) -> str:
        return re.sub(r"\s+", " ", normalize_answer(text)).strip()

    def _build_plan_nodes(self, state: PipelineState, question: str) -> List[SubquestionNode]:
        prompt = PLAN_PROMPT.format(
            question=question,
            max_initial_subquestions=self.max_initial_subquestions,
            reasoning_family_hints_json=_safe_json_array_str(_infer_reasoning_family_hints(question)),
            operator_hints_json=_safe_json_array_str(_infer_operator_hints(question)),
            expected_answer_type_hint=_infer_expected_answer_type_hint(question),
            slot_hints_json=_safe_json_array_str(_infer_slot_hints(question)),
        )
        try:
            parsed = self.planner.call_json(mode="plan", prompt=prompt)
            raw_nodes = parsed.get("subgoals", [])
        except Exception as exc:  # noqa: BLE001
            raw_nodes = []
            state.planning_trace.append(
                {
                    "event": "plan_error",
                    "error": str(exc),
                }
            )

        nodes: List[SubquestionNode] = []
        seen: Set[str] = set()
        if isinstance(raw_nodes, list):
            for i, item in enumerate(raw_nodes):
                if not isinstance(item, dict):
                    continue
                node_id = _clean_node_id(item.get("id", f"sg{i+1}"), fallback=f"sg{i+1}")
                subquestion = str(item.get("subquestion", "") or item.get("subgoal", "")).strip()
                if not subquestion:
                    continue
                key = self._normalized_question_key(subquestion)
                if not key or key in seen:
                    state.planning_trace.append(
                        {
                            "event": "duplicate_filtered_exact",
                            "source_stage": "plan",
                            "candidate_id": node_id,
                            "candidate_subquestion": subquestion,
                        }
                    )
                    continue
                seen.add(key)
                node = SubquestionNode(
                    id=node_id,
                    parent_id=None,
                    depth=0,
                    subquestion=subquestion,
                    depends_on=_normalize_depends(item.get("depends_on", [])),
                    retrieve=_normalize_bool(item.get("retrieve", True), default=True),
                    purpose=str(item.get("purpose", "atomic_fact")).strip() or "atomic_fact",
                )
                nodes.append(node)
                if len(nodes) >= self.max_initial_subquestions:
                    break

        if not nodes:
            nodes = [
                SubquestionNode(
                    id="sg1",
                    parent_id=None,
                    depth=0,
                    subquestion=question,
                    depends_on=[],
                    retrieve=True,
                    purpose="fallback",
                )
            ]

        node_ids = {node.id for node in nodes}
        for node in nodes:
            node.depends_on = [dep for dep in node.depends_on if dep in node_ids and dep != node.id]
        return self._repair_plan_nodes(state=state, question=question, nodes=nodes)

    def _repair_plan_nodes(
        self,
        state: PipelineState,
        question: str,
        nodes: List[SubquestionNode],
    ) -> List[SubquestionNode]:
        if not nodes:
            return nodes

        original_key = self._normalized_question_key(question)
        question_requires_composition = _requires_composition_node(question)
        question_operator_hints = set(_infer_operator_hints(question))

        for node in nodes:
            if not node.depends_on:
                continue
            node_key = self._normalized_question_key(node.subquestion)
            node_operator_hints = set(_infer_operator_hints(node.subquestion))
            if (
                _is_composition_purpose(node.purpose)
                or (question_operator_hints and bool(question_operator_hints & node_operator_hints))
                or node_key == original_key
                or _looks_like_reask(node.subquestion, question)
            ):
                node.retrieve = False
                node.purpose = _default_composition_purpose(question)
                state.planning_trace.append(
                    {
                        "event": "plan_repair_memory_composition",
                        "node_id": node.id,
                        "subquestion": node.subquestion,
                    }
                )

        if (
            question_requires_composition
            and len(nodes) < self.max_initial_subquestions
            and not any(_is_composition_purpose(node.purpose) for node in nodes)
        ):
            parent_ids = {dep for node in nodes for dep in node.depends_on}
            compose_depends = [node.id for node in nodes if node.id not in parent_ids]
            if len(compose_depends) >= 2:
                next_idx = 1
                existing_ids = {node.id for node in nodes}
                while f"sg{next_idx}" in existing_ids:
                    next_idx += 1
                compose_id = f"sg{next_idx}"
                nodes.append(
                    SubquestionNode(
                        id=compose_id,
                        parent_id=None,
                        depth=0,
                        subquestion=question,
                        depends_on=compose_depends,
                        retrieve=False,
                        purpose=_default_composition_purpose(question),
                    )
                )
                state.planning_trace.append(
                    {
                        "event": "plan_repair_add_composition",
                        "node_id": compose_id,
                        "depends_on": list(compose_depends),
                    }
                )
        return nodes

    def _collect_children(self, state: PipelineState, parent_id: str) -> List[SubquestionNode]:
        return [node for node in state.nodes.values() if node.parent_id == parent_id]

    def _deps_successful(self, state: PipelineState, node: SubquestionNode) -> bool:
        for dep_id in node.depends_on:
            dep = state.nodes.get(dep_id)
            if dep is None or dep.status not in SUCCESS_STATUSES:
                return False
        return True

    def _dependency_memory_entries(self, state: PipelineState, node: SubquestionNode) -> List[Dict[str, Any]]:
        entry_by_node: Dict[str, Dict[str, Any]] = {}
        for entry in state.appended_entries:
            node_id = str(entry.get("node_id", "")).strip()
            if node_id:
                entry_by_node[node_id] = entry

        entries: List[Dict[str, Any]] = []
        seen_nodes: Set[str] = set()
        visited: Set[str] = set()
        stack = list(node.depends_on)
        while stack:
            dep_id = stack.pop()
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep = state.nodes.get(dep_id)
            if dep is None or dep.status not in SUCCESS_STATUSES:
                continue
            entry = entry_by_node.get(dep_id)
            if entry is not None and dep_id not in seen_nodes:
                seen_nodes.add(dep_id)
                entries.append(entry)
            if dep.status == "resolved_from_children":
                stack.extend(
                    child.id
                    for child in self._collect_children(state=state, parent_id=dep_id)
                    if child.status in SUCCESS_STATUSES
                )
            stack.extend(dep.depends_on)
        return entries

    def _resolved_dependency_prompt_records(
        self,
        state: PipelineState,
        node: SubquestionNode,
    ) -> List[Dict[str, Any]]:
        execution_rank = {node_id: idx for idx, node_id in enumerate(state.execution_order)}
        records: List[Dict[str, Any]] = []
        for entry in self._dependency_memory_entries(state=state, node=node):
            node_id = str(entry.get("node_id", "")).strip()
            if not node_id:
                continue
            dep_node = state.nodes.get(node_id)
            subquestion = str(entry.get("subquestion", "")).strip()
            if dep_node is not None and dep_node.subquestion:
                subquestion = dep_node.subquestion
            record = {
                "id": node_id,
                "subquestion": subquestion,
                "resolved_subquestion": (
                    str(getattr(dep_node, "resolved_subquestion", "")).strip()
                    if dep_node is not None
                    else str(entry.get("resolved_subquestion", "")).strip()
                ),
                "answer": str(entry.get("answer", "")).strip(),
                "summary": str(entry.get("summary", "")).strip(),
                "facts": _parse_string_list(entry.get("facts", []))[:3],
                "answer_source": str(entry.get("answer_source", "")).strip(),
                "_sort_key": execution_rank.get(node_id, len(execution_rank)),
            }
            records.append(record)
        records.sort(key=lambda item: (int(item.pop("_sort_key", 0)), str(item.get("id", ""))))
        return records

    @staticmethod
    def _entries_to_passages(entries: Sequence[Dict[str, Any]]) -> List[Passage]:
        passages: List[Passage] = []
        for i, entry in enumerate(entries):
            text = str(entry.get("memory_text", "")).strip()
            if not text:
                continue
            title = f"Memory-{entry.get('node_id', f'node{i+1}')}"
            passages.append(Passage(index=i, title=title, text=text))
        return passages

    @staticmethod
    def _effective_subquestion(node: SubquestionNode) -> str:
        text = str(node.resolved_subquestion or node.subquestion or "").strip()
        return text

    def _resolve_dependency_subquestion(
        self,
        state: PipelineState,
        question: str,
        node: SubquestionNode,
    ) -> str:
        original_subquestion = str(node.subquestion).strip()
        if not node.depends_on:
            node.resolved_subquestion = original_subquestion
            return node.resolved_subquestion

        dependency_records = self._resolved_dependency_prompt_records(state=state, node=node)
        if not dependency_records:
            node.resolved_subquestion = original_subquestion
            return node.resolved_subquestion

        resolved_dependency_ids_json = _safe_json_array_str(
            [
                str(record.get("id", "")).strip()
                for record in dependency_records
                if str(record.get("id", "")).strip()
            ]
        )
        resolved_dependency_context_json = _safe_json_array_str(dependency_records)
        prompt = DEPENDENCY_REWRITE_PROMPT.format(
            question=question,
            subquestion=original_subquestion,
            resolved_dependency_ids_json=resolved_dependency_ids_json,
            resolved_dependency_context_json=resolved_dependency_context_json,
            operator_hints_json=_safe_json_array_str(_infer_operator_hints(original_subquestion or question)),
            expected_answer_type_hint=_infer_expected_answer_type_hint(original_subquestion or question),
            slot_hints_json=_safe_json_array_str(_infer_slot_hints(original_subquestion or question)),
        )
        try:
            parsed = self.planner.call_json(mode="dependency_rewrite", prompt=prompt)
            rewritten = str(parsed.get("resolved_subquestion", "")).strip()
            validated_rewrite = rewritten or original_subquestion
            fallback_used = False
            fallback_reason = ""

            original_ops = set(_infer_operator_hints(original_subquestion or question))
            rewrite_ops = set(_infer_operator_hints(validated_rewrite))
            if original_ops and not (original_ops & rewrite_ops):
                validated_rewrite = original_subquestion
                fallback_used = True
                fallback_reason = "operator_dropped"

            expected_type = _infer_expected_answer_type_hint(original_subquestion or question)
            rewrite_type = _infer_expected_answer_type_hint(validated_rewrite)
            if not fallback_used and not _rewrite_type_compatible(expected_type, rewrite_type):
                validated_rewrite = original_subquestion
                fallback_used = True
                fallback_reason = f"answer_type_mismatch:{expected_type}->{rewrite_type}"

            if (
                not fallback_used
                and len(validated_rewrite.split()) > max(24, int(max(1, len(original_subquestion.split())) * 1.75))
                and len(validated_rewrite.split()) > len(original_subquestion.split()) + 8
            ):
                validated_rewrite = original_subquestion
                fallback_used = True
                fallback_reason = "rewrite_too_verbose"

            node.resolved_subquestion = validated_rewrite
            state.planning_trace.append(
                {
                    "event": "dependency_rewrite",
                    "node_id": node.id,
                    "original_subquestion": original_subquestion,
                    "resolved_subquestion": node.resolved_subquestion,
                    "dependency_ids": [
                        str(record.get("id", "")).strip()
                        for record in dependency_records
                        if str(record.get("id", "")).strip()
                    ],
                    "rewrite_changed": node.resolved_subquestion != original_subquestion,
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason or None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            node.resolved_subquestion = original_subquestion
            state.planning_trace.append(
                {
                    "event": "dependency_rewrite_error",
                    "node_id": node.id,
                    "original_subquestion": original_subquestion,
                    "error": str(exc),
                }
            )
        return node.resolved_subquestion

    def _select_retrieval_passages(
        self,
        question: str,
        passages: Sequence[Passage],
    ) -> Tuple[List[Passage], List[int], List[float]]:
        if not passages:
            return [], [], []
        order, scores = rank_passages(question=question, passages=passages)
        keep = order[: min(self.retrieval_top_k, len(order))]
        selected = [passages[i] for i in keep]
        return selected, list(keep), [float(scores[i]) for i in keep]

    def _build_facts(self, question: str, node: SubquestionNode) -> List[str]:
        effective_subquestion = self._effective_subquestion(node)
        prompt = FACT_PROMPT.format(
            question=question,
            subquestion=effective_subquestion,
            answer=node.answer,
            explanation=node.explanation,
            support_claims_json=_safe_json_array_str(node.support_claims),
        )
        try:
            parsed = self.planner.call_json(mode="facts", prompt=prompt)
            facts = _parse_string_list(parsed.get("facts", []))
            if facts:
                return facts
        except Exception:  # noqa: BLE001
            pass
        if node.support_claims:
            return list(node.support_claims[:3])
        if node.answer:
            return [f"For the question '{effective_subquestion}', the answer is {node.answer}."]
        return []

    def _build_summary(self, question: str, node: SubquestionNode) -> str:
        effective_subquestion = self._effective_subquestion(node)
        prompt = MEMORY_SUMMARY_PROMPT.format(
            question=question,
            subquestion=effective_subquestion,
            answer=node.answer,
            facts_json=_safe_json_array_str(node.facts),
        )
        try:
            parsed = self.planner.call_json(mode="memory_summary", prompt=prompt)
            summary = str(parsed.get("summary", "")).strip()
            if summary:
                return summary
        except Exception:  # noqa: BLE001
            pass
        if node.facts:
            return " ".join(node.facts[:2])
        if node.explanation:
            return node.explanation
        return node.answer or "insufficient evidence"

    def _append_success_entry(self, state: PipelineState, question: str, node: SubquestionNode) -> None:
        node.facts = self._build_facts(question=question, node=node)
        node.summary = self._build_summary(question=question, node=node)
        memory_text = str(node.summary or "").strip()
        entry = {
            "type": "resolved",
            "node_id": node.id,
            "subquestion": node.subquestion,
            "resolved_subquestion": self._effective_subquestion(node),
            "retrieve": bool(node.retrieve),
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
            "memory_text": memory_text,
        }
        state.appended_entries.append(entry)
        state.planning_trace.append(
            {
                "event": "append_success",
                "node_id": node.id,
                "source": node.answer_source,
                "nvc": node.selected_nvc,
                "sc_conf": node.selected_sc_conf,
                "dinco_conf": node.selected_dinco_conf,
                "fact_count": len(node.facts),
            }
        )

    def _commit_success(self, node: SubquestionNode, attempt: AttemptResult, retrieved_titles: Sequence[str]) -> None:
        node.answer = attempt.answer
        node.raw_answer = attempt.raw_answer
        node.explanation = attempt.explanation
        node.support_claims = list(attempt.support_claims)
        node.answer_source = attempt.source
        node.selected_nvc = attempt.nvc
        node.selected_sc_conf = attempt.sc_conf
        node.selected_dinco_conf = attempt.dinco_conf
        node.dinco_candidates = list(attempt.dinco_candidates)
        node.dinco_ptrues = list(attempt.dinco_ptrues)
        node.retrieved_passage_indices = list(attempt.retrieved_passage_indices)
        node.retrieved_titles = list(retrieved_titles)
        node.resolution_note = f"Resolved from {attempt.source} evidence."
        node.status = "resolved"

    def _is_dependency_reask(
        self,
        *,
        candidate_subquestion: str,
        candidate_depends_on: Sequence[str],
        reusable_dependency_ids: Set[str],
        dependency_records: Sequence[Dict[str, Any]],
    ) -> bool:
        if not reusable_dependency_ids:
            return False
        if any(dep in reusable_dependency_ids for dep in candidate_depends_on):
            return False
        candidate_key = self._normalized_question_key(candidate_subquestion)
        if not candidate_key:
            return False
        for record in dependency_records:
            dep_subquestion = str(record.get("subquestion", "")).strip()
            dep_key = self._normalized_question_key(dep_subquestion)
            if not dep_key:
                continue
            if candidate_key == dep_key:
                return True
            if _looks_like_reask(candidate_subquestion, dep_subquestion):
                return True
        return False

    def _child_reference_integrity_reasons(
        self,
        *,
        child: SubquestionNode,
        allowed_ids: Set[str],
    ) -> List[str]:
        reasons: List[str] = []
        allowed_norm = {dep.lower() for dep in allowed_ids}
        child_norm = str(child.id or "").strip().lower()

        for dep in child.depends_on:
            dep_norm = str(dep or "").strip().lower()
            if not dep_norm:
                continue
            if dep_norm == child_norm:
                reasons.append(f"self_dep:{dep}")
            elif dep_norm not in allowed_norm:
                reasons.append(f"missing_dep:{dep}")

        referenced_ids = _extract_node_refs(child.subquestion, child.resolved_subquestion)
        depends_norm = {str(dep).strip().lower() for dep in child.depends_on if str(dep).strip()}
        for ref in referenced_ids:
            ref_norm = ref.lower()
            if ref_norm == child_norm:
                reasons.append(f"self_ref:{ref}")
            if ref_norm not in allowed_norm:
                reasons.append(f"missing_text_ref:{ref}")
            if ref_norm not in depends_norm:
                reasons.append(f"text_ref_not_in_depends:{ref}")

        return _unique_keep_order(reasons)

    def _log_invalid_child_reference(
        self,
        *,
        state: PipelineState,
        node: SubquestionNode,
        child: SubquestionNode,
        source_stage: str,
        reasons: Sequence[str],
    ) -> None:
        state.planning_trace.append(
            {
                "event": "invalid_child_placeholder_refs",
                "source_stage": source_stage,
                "node_id": node.id,
                "child_id": child.id,
                "child_subquestion": child.subquestion,
                "child_depends_on": list(child.depends_on),
                "reasons": list(reasons),
            }
        )

    def _validate_child_reference_integrity(
        self,
        *,
        state: PipelineState,
        node: SubquestionNode,
        children: Sequence[SubquestionNode],
        reusable_dependency_ids: Set[str],
        source_stage: str,
    ) -> bool:
        allowed_ids = {child.id for child in children} | set(reusable_dependency_ids)
        invalid_found = False
        for child in children:
            reasons = self._child_reference_integrity_reasons(child=child, allowed_ids=allowed_ids)
            if reasons:
                self._log_invalid_child_reference(
                    state=state,
                    node=node,
                    child=child,
                    source_stage=source_stage,
                    reasons=reasons,
                )
                invalid_found = True
        return not invalid_found

    def _maybe_add_bridge_repair_child(
        self,
        *,
        state: PipelineState,
        question: str,
        node: SubquestionNode,
        execution_subquestion: str,
        attempt: AttemptResult,
        running_id_counter: int,
    ) -> tuple[bool, int]:
        if node.depth >= self.max_subquestion_depth:
            return False, running_id_counter
        if len(state.nodes) >= self.max_subquestion_nodes:
            return False, running_id_counter
        if not attempt.available or normalize_answer(attempt.answer) == normalize_answer("insufficient evidence"):
            return False, running_id_counter

        expected_answer_type_hint = _infer_expected_answer_type_hint(execution_subquestion or node.subquestion or question)
        slot_hints_json = _safe_json_array_str(_infer_slot_hints(execution_subquestion or node.subquestion or question))
        support_claims = _parse_string_list(attempt.support_claims)
        support_claims_json = _safe_json_array_str(support_claims)

        base_prompt = BRIDGE_REPAIR_PROMPT.format(
            question=question,
            subquestion=node.subquestion,
            resolved_subquestion=execution_subquestion,
            answer=attempt.answer,
            explanation=attempt.explanation,
            support_claims_json=support_claims_json,
            expected_answer_type_hint=expected_answer_type_hint,
            slot_hints_json=slot_hints_json,
        )

        def parse_bridge_child(parsed: Dict[str, Any], source_stage: str) -> Optional[SubquestionNode]:
            action = normalize_answer(str(parsed.get("action", "accept") or "accept"))
            if action != "bridge_repair":
                return None

            raw_children = parsed.get("replacement_subgoals", [])
            if not isinstance(raw_children, list) or len(raw_children) != 1 or not isinstance(raw_children[0], dict):
                state.planning_trace.append(
                    {
                        "event": "bridge_repair_invalid_shape",
                        "source_stage": source_stage,
                        "node_id": node.id,
                        "parsed_action": action,
                    }
                )
                return None

            item = raw_children[0]
            subquestion = str(item.get("subquestion", "") or item.get("subgoal", "")).strip()
            depends_on = _normalize_depends(item.get("depends_on", []))
            candidate_id = _clean_node_id(item.get("id", f"{node.id}_bridge"), fallback=f"{node.id}_bridge")
            if candidate_id in state.nodes:
                candidate_id = f"{candidate_id}_{max(1, running_id_counter + 1)}"
            child = SubquestionNode(
                id=candidate_id,
                parent_id=node.id,
                depth=node.depth + 1,
                subquestion=subquestion,
                depends_on=depends_on,
                retrieve=True,
                purpose=str(item.get("purpose", "bridge_fact")).strip() or "bridge_fact",
            )

            reasons: List[str] = []
            if not subquestion:
                reasons.append("empty_subquestion")
            if child.depends_on:
                reasons.append("bridge_repair_depends_not_allowed")
            if _extract_node_refs(subquestion):
                reasons.append("bridge_repair_contains_placeholder_ref")
            if self._normalized_question_key(subquestion) in {
                self._normalized_question_key(node.subquestion),
                self._normalized_question_key(execution_subquestion),
            }:
                reasons.append("bridge_repair_same_question")
            elif _looks_like_reask(subquestion, execution_subquestion):
                reasons.append("bridge_repair_paraphrase_reask")
            if reasons:
                state.planning_trace.append(
                    {
                        "event": "bridge_repair_invalid_child",
                        "source_stage": source_stage,
                        "node_id": node.id,
                        "child_id": child.id,
                        "child_subquestion": child.subquestion,
                        "reasons": list(reasons),
                    }
                )
                return None
            return child

        retry_prompt = (
            f"{base_prompt}\n\n"
            "Previous bridge-repair attempt was invalid. If you return bridge_repair, "
            "the child must inline the supported answer directly, must not mention any sg ids, "
            "must have depends_on as an empty list, and must ask only the remaining target slot."
        )

        bridge_child: Optional[SubquestionNode] = None
        for attempt_idx, prompt in enumerate((base_prompt, retry_prompt)):
            try:
                parsed = self.planner.call_json(mode="bridge_repair", prompt=prompt)
            except Exception as exc:  # noqa: BLE001
                state.planning_trace.append(
                    {
                        "event": "bridge_repair_error",
                        "node_id": node.id,
                        "attempt_index": attempt_idx,
                        "error": str(exc),
                    }
                )
                continue
            bridge_child = parse_bridge_child(parsed, "bridge_repair_retry" if attempt_idx else "bridge_repair")
            if bridge_child is not None or normalize_answer(str(parsed.get("action", "accept") or "accept")) != "bridge_repair":
                break

        if bridge_child is None:
            state.planning_trace.append(
                {
                    "event": "bridge_repair_attempted",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "answer": attempt.answer,
                    "accepted_direct_answer": True,
                }
            )
            return False, running_id_counter

        running_id_counter += 1
        if bridge_child.id in state.nodes:
            bridge_child.id = f"{node.id}_bridge_{running_id_counter}"
        state.nodes[bridge_child.id] = bridge_child
        node.status = "decomposed"
        state.planning_trace.append(
            {
                "event": "bridge_repair_child_added",
                "node_id": node.id,
                "child_id": bridge_child.id,
                "resolved_subquestion": execution_subquestion,
                "bridge_answer": attempt.answer,
                "child_subquestion": bridge_child.subquestion,
            }
        )
        return True, running_id_counter

    def _decompose_node(
        self,
        state: PipelineState,
        question: str,
        node: SubquestionNode,
        running_id_counter: int,
    ) -> int:
        if node.depth >= self.max_subquestion_depth:
            node.status = "failed_unresolved"
            state.planning_trace.append(
                {
                    "event": "decompose_blocked_depth",
                    "node_id": node.id,
                    "depth": node.depth,
                }
            )
            return running_id_counter

        if len(state.nodes) >= self.max_subquestion_nodes:
            node.status = "failed_unresolved"
            state.planning_trace.append(
                {
                    "event": "decompose_blocked_node_budget",
                    "node_id": node.id,
                    "node_count": len(state.nodes),
                }
            )
            return running_id_counter

        dependency_records = self._resolved_dependency_prompt_records(state=state, node=node)
        reusable_dependency_ids = {
            str(record.get("id", "")).strip()
            for record in dependency_records
            if str(record.get("id", "")).strip()
        }
        resolved_dependency_ids_json = _safe_json_array_str(sorted(reusable_dependency_ids))
        resolved_dependency_context_json = _safe_json_array_str(dependency_records)

        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            parent_id=node.id,
            failed_subquestion=node.subquestion,
            failed_resolved_subquestion=self._effective_subquestion(node),
            resolved_dependency_ids_json=resolved_dependency_ids_json,
            resolved_dependency_context_json=resolved_dependency_context_json,
            reasoning_family_hints_json=_safe_json_array_str(
                _infer_reasoning_family_hints(self._effective_subquestion(node) or node.subquestion or question)
            ),
            operator_hints_json=_safe_json_array_str(
                _infer_operator_hints(self._effective_subquestion(node) or node.subquestion or question)
            ),
            expected_answer_type_hint=_infer_expected_answer_type_hint(
                self._effective_subquestion(node) or node.subquestion or question
            ),
            slot_hints_json=_safe_json_array_str(
                _infer_slot_hints(self._effective_subquestion(node) or node.subquestion or question)
            ),
        )
        state.planning_trace.append(
            {
                "event": "decompose_prompt_context",
                "node_id": node.id,
                "failed_resolved_subquestion": self._effective_subquestion(node),
                "resolved_dependency_ids": sorted(reusable_dependency_ids),
                "resolved_dependency_context": dependency_records,
            }
        )
        failed_text = self._effective_subquestion(node) or node.subquestion or question

        def build_children(raw_children: Any, source_stage: str) -> List[SubquestionNode]:
            nonlocal running_id_counter
            children: List[SubquestionNode] = []
            seen_keys = {self._normalized_question_key(existing.subquestion) for existing in state.nodes.values()}
            if isinstance(raw_children, list):
                for item in raw_children:
                    if not isinstance(item, dict):
                        continue
                    running_id_counter += 1
                    candidate_id = _clean_node_id(
                        item.get("id", f"{node.id}_d{running_id_counter}"),
                        fallback=f"{node.id}_d{running_id_counter}",
                    )
                    if candidate_id in state.nodes:
                        candidate_id = f"{candidate_id}_{running_id_counter}"
                    subquestion = str(item.get("subquestion", "") or item.get("subgoal", "")).strip()
                    if not subquestion:
                        continue
                    candidate_depends = _normalize_depends(item.get("depends_on", []))
                    key = self._normalized_question_key(subquestion)
                    if not key or key in seen_keys:
                        state.planning_trace.append(
                            {
                                "event": "duplicate_filtered_exact",
                                "source_stage": source_stage,
                                "node_id": node.id,
                                "candidate_id": candidate_id,
                                "candidate_subquestion": subquestion,
                            }
                        )
                        continue
                    if self._is_dependency_reask(
                        candidate_subquestion=subquestion,
                        candidate_depends_on=candidate_depends,
                        reusable_dependency_ids=reusable_dependency_ids,
                        dependency_records=dependency_records,
                    ):
                        state.planning_trace.append(
                            {
                                "event": "duplicate_filtered_dependency_reask",
                                "source_stage": source_stage,
                                "node_id": node.id,
                                "candidate_id": candidate_id,
                                "candidate_subquestion": subquestion,
                                "resolved_dependency_ids": sorted(reusable_dependency_ids),
                            }
                        )
                        continue
                    seen_keys.add(key)
                    child = SubquestionNode(
                        id=candidate_id,
                        parent_id=node.id,
                        depth=node.depth + 1,
                        subquestion=subquestion,
                        depends_on=candidate_depends,
                        retrieve=_normalize_bool(item.get("retrieve", True), default=True),
                        purpose=str(item.get("purpose", "atomic_fact")).strip() or "atomic_fact",
                    )
                    children.append(child)
                    if len(state.nodes) + len(children) >= self.max_subquestion_nodes:
                        break

            filtered_children: List[SubquestionNode] = []
            for child in children:
                if self._normalized_question_key(child.subquestion) in {
                    self._normalized_question_key(node.subquestion),
                    self._normalized_question_key(failed_text),
                }:
                    state.planning_trace.append(
                        {
                            "event": "duplicate_filtered_failed_reask",
                            "source_stage": source_stage,
                            "node_id": node.id,
                            "candidate_id": child.id,
                            "candidate_subquestion": child.subquestion,
                        }
                    )
                    continue
                if not child.depends_on and _looks_like_reask(child.subquestion, failed_text):
                    state.planning_trace.append(
                        {
                            "event": "duplicate_filtered_failed_paraphrase",
                            "source_stage": source_stage,
                            "node_id": node.id,
                            "candidate_id": child.id,
                            "candidate_subquestion": child.subquestion,
                        }
                    )
                    continue
                filtered_children.append(child)
            children = filtered_children

            if _requires_composition_node(failed_text) and not any(_is_composition_purpose(child.purpose) for child in children):
                compose_depends = [child.id for child in children]
                compose_depends.extend(dep for dep in node.depends_on if dep in reusable_dependency_ids and dep not in compose_depends)
                if len(compose_depends) >= 2 and len(state.nodes) + len(children) < self.max_subquestion_nodes:
                    running_id_counter += 1
                    compose_id = f"{node.id}_d{running_id_counter}"
                    children.append(
                        SubquestionNode(
                            id=compose_id,
                            parent_id=node.id,
                            depth=node.depth + 1,
                            subquestion=failed_text,
                            depends_on=compose_depends,
                            retrieve=False,
                            purpose=_default_composition_purpose(failed_text),
                        )
                    )
                    state.planning_trace.append(
                        {
                            "event": "decompose_repair_add_composition",
                            "node_id": node.id,
                            "child_id": compose_id,
                            "depends_on": list(compose_depends),
                            "source_stage": source_stage,
                        }
                    )
            return children

        def call_decompose(prompt_text: str, source_stage: str) -> List[SubquestionNode]:
            try:
                parsed = self.planner.call_json(mode="decompose", prompt=prompt_text)
                raw_children = parsed.get("replacement_subgoals", [])
            except Exception as exc:  # noqa: BLE001
                state.planning_trace.append(
                    {
                        "event": "decompose_error",
                        "source_stage": source_stage,
                        "node_id": node.id,
                        "error": str(exc),
                    }
                )
                raw_children = []
            return build_children(raw_children, source_stage)

        children = call_decompose(prompt, "decompose")
        if children and not self._validate_child_reference_integrity(
            state=state,
            node=node,
            children=children,
            reusable_dependency_ids=reusable_dependency_ids,
            source_stage="decompose",
        ):
            state.planning_trace.append(
                {
                    "event": "decompose_retry_for_invalid_children",
                    "node_id": node.id,
                }
            )
            retry_prompt = (
                f"{prompt}\n\n"
                "Previous decomposition attempt was invalid. Every sg... id mentioned in any replacement_subgoal "
                "must exist either in this response or in the resolved dependency ids above, and every textual "
                "sg... reference must also appear in depends_on. Do not return any child that references a missing sg id."
            )
            children = call_decompose(retry_prompt, "decompose_retry")
            if children and not self._validate_child_reference_integrity(
                state=state,
                node=node,
                children=children,
                reusable_dependency_ids=reusable_dependency_ids,
                source_stage="decompose_retry",
            ):
                state.planning_trace.append(
                    {
                        "event": "decompose_retry_failed",
                        "node_id": node.id,
                    }
                )
                children = []

        if not children:
            # Patch: decompose-empty fallback. Instead of marking the node as
            # `failed_unresolved` (which terminates the subgoal with no evidence and
            # no answer), mark it back to `pending` so the agent re-enters this node
            # on the next planner sweep. Combined with the role prompt's hard rule
            # against first-turn decompose without evidence, the agent will pick
            # `retrieve` on the retry. Track the attempts to avoid infinite loops.
            node.decompose_empty_attempts = getattr(node, "decompose_empty_attempts", 0) + 1
            if node.decompose_empty_attempts >= 2:
                # Already retried — give up to avoid loops.
                node.status = "failed_unresolved"
                state.planning_trace.append(
                    {
                        "event": "decompose_empty",
                        "node_id": node.id,
                        "attempts": node.decompose_empty_attempts,
                        "note": "exhausted retries; marked failed_unresolved",
                    }
                )
            else:
                node.status = "pending"
                state.planning_trace.append(
                    {
                        "event": "decompose_empty_fallback_to_pending",
                        "node_id": node.id,
                        "attempts": node.decompose_empty_attempts,
                    }
                )
            return running_id_counter

        child_ids = {child.id for child in children}
        for child in children:
            child.depends_on = [
                dep
                for dep in child.depends_on
                if dep != child.id and (dep in child_ids or dep in reusable_dependency_ids)
            ]
            state.nodes[child.id] = child

        node.status = "decomposed"
        state.planning_trace.append(
            {
                "event": "decompose_success",
                "node_id": node.id,
                "child_ids": [child.id for child in children],
            }
        )
        return running_id_counter

    def _refresh_decomposed_parents(self, state: PipelineState) -> None:
        for node in list(state.nodes.values()):
            if node.status != "decomposed":
                continue
            children = self._collect_children(state=state, parent_id=node.id)
            if not children:
                node.status = "failed_unresolved"
                continue
            if any(child.status not in TERMINAL_STATUSES for child in children):
                continue
            successful_children = [child for child in children if child.status in SUCCESS_STATUSES]
            if not successful_children:
                node.status = "failed_unresolved"
                state.planning_trace.append(
                    {
                        "event": "decompose_children_unresolved",
                        "node_id": node.id,
                    }
                )
                continue
            best_child = max(
                successful_children,
                key=lambda child: (
                    float(child.selected_dinco_conf or 0.0),
                    float(child.selected_nvc or 0.0),
                    float(child.selected_sc_conf or 0.0),
                ),
            )
            node.answer = best_child.answer
            node.raw_answer = best_child.raw_answer
            node.explanation = "Resolved via successful decomposed children."
            node.support_claims = _unique_keep_order(
                [claim for child in successful_children for claim in (child.facts or child.support_claims)]
            )
            node.answer_source = "children"
            nvc_values = [float(child.selected_nvc) for child in successful_children if child.selected_nvc is not None]
            sc_values = [float(child.selected_sc_conf) for child in successful_children if child.selected_sc_conf is not None]
            dinco_values = [
                float(child.selected_dinco_conf)
                for child in successful_children
                if child.selected_dinco_conf is not None
            ]
            node.selected_nvc = max(nvc_values) if nvc_values else None
            node.selected_sc_conf = max(sc_values) if sc_values else None
            node.selected_dinco_conf = max(dinco_values) if dinco_values else None
            node.status = "resolved_from_children"
            node.resolution_note = "Resolved from decomposed child subquestions."
            state.planning_trace.append(
                {
                    "event": "parent_resolved_from_children",
                    "node_id": node.id,
                    "child_ids": [child.id for child in successful_children],
                }
            )

    def _choose_next_node(self, state: PipelineState) -> Optional[SubquestionNode]:
        ready: List[SubquestionNode] = []
        for node in state.nodes.values():
            if node.status != "pending":
                continue
            if self._deps_successful(state=state, node=node):
                ready.append(node)
        if not ready:
            return None
        ready.sort(key=lambda node: (node.depth, node.id))
        return ready[0]

    def _run_subquestion(
        self,
        state: PipelineState,
        question: str,
        node: SubquestionNode,
        running_id_counter: int,
        full_passages: Sequence[Passage],
    ) -> int:
        node.status = "running"
        execution_subquestion = self._resolve_dependency_subquestion(state=state, question=question, node=node)
        dependency_entries = self._dependency_memory_entries(state=state, node=node)
        dependency_passages = self._entries_to_passages(dependency_entries)
        state.execution_order.append(node.id)

        retrieved_passages: List[Passage] = []
        retrieved_indices: List[int] = []
        retrieved_scores: List[float] = []
        answer_passages: List[Passage] = []
        if node.retrieve:
            retrieved_passages, retrieved_indices, retrieved_scores = self._select_retrieval_passages(
                question=execution_subquestion,
                passages=full_passages,
            )
            answer_passages = list(dependency_passages) + list(retrieved_passages)
        else:
            answer_passages = list(dependency_passages)

        state.planning_trace.append(
            {
                "event": "execute_subquestion",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "depends_on": list(node.depends_on),
                "retrieve": bool(node.retrieve),
                "purpose": node.purpose,
                "dependency_memory_count": len(dependency_passages),
                "retrieved_count": len(retrieved_passages),
                "retrieved_passage_indices": list(retrieved_indices),
                "retrieved_titles": [p.title for p in retrieved_passages],
                "retrieval_scores": list(retrieved_scores),
            }
        )

        if not node.retrieve and not dependency_passages:
            node.status = "failed_unresolved"
            state.planning_trace.append(
                {
                    "event": "memory_unavailable",
                    "node_id": node.id,
                    "reason": "retrieve_false_without_dependency_memory",
                }
            )
            return self._decompose_node(
                state=state,
                question=question,
                node=node,
                running_id_counter=running_id_counter,
            )

        source = "retrieval" if node.retrieve else "memory"
        attempt = self.answer_engine.answer_with_passages(
            question=execution_subquestion,
            passages=answer_passages,
            source=source,
            unavailable_explanation="No passages were available for this subquestion.",
        )
        attempt.retrieved_passage_indices = list(retrieved_indices)
        attempt.retrieved_titles = [p.title for p in retrieved_passages]

        node.answer = attempt.answer
        node.raw_answer = attempt.raw_answer
        node.explanation = attempt.explanation
        node.support_claims = list(attempt.support_claims)
        node.answer_source = attempt.source
        node.selected_nvc = attempt.nvc
        node.selected_sc_conf = attempt.sc_conf
        node.selected_dinco_conf = attempt.dinco_conf
        node.dinco_candidates = list(attempt.dinco_candidates)
        node.dinco_ptrues = list(attempt.dinco_ptrues)
        node.retrieved_passage_indices = list(retrieved_indices)
        node.retrieved_titles = [p.title for p in retrieved_passages]

        passed = bool(
            attempt.available
            and attempt.dinco_conf is not None
            and attempt.dinco_conf >= self.confidence_threshold
            and normalize_answer(attempt.answer) != normalize_answer("insufficient evidence")
        )
        state.planning_trace.append(
            {
                "event": "subquestion_scored",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "source": attempt.source,
                "answer": attempt.answer,
                "nvc": attempt.nvc,
                "sc_conf": attempt.sc_conf,
                "dinco_conf": attempt.dinco_conf,
                "passed": passed,
            }
        )

        if passed:
            self._commit_success(node=node, attempt=attempt, retrieved_titles=[p.title for p in retrieved_passages])
            self._append_success_entry(state=state, question=question, node=node)
            return running_id_counter

        return self._decompose_node(
            state=state,
            question=question,
            node=node,
            running_id_counter=running_id_counter,
        )

    def _run_final_answer(self, state: PipelineState, question: str) -> Dict[str, Any]:
        memory_passages = self._entries_to_passages(state.appended_entries)
        attempt = self.answer_engine.answer_with_passages(
            question=question,
            passages=memory_passages,
            source="memory_final",
            unavailable_explanation="No appended memory was available for the final answer.",
        )
        state.planning_trace.append(
            {
                "event": "final_memory_answered",
                "answer": attempt.answer if attempt.available else "",
                "nvc": attempt.nvc,
                "sc_conf": attempt.sc_conf,
                "dinco_conf": attempt.dinco_conf,
                "used_dinco": bool(attempt.available),
                "memory_entry_count": len(state.appended_entries),
            }
        )

        pred_answer = attempt.answer if attempt.available else "insufficient evidence"
        pred_raw = attempt.raw_answer if attempt.available else pred_answer
        return {
            "pred_answer": pred_answer or "insufficient evidence",
            "pred_answer_raw": pred_raw or pred_answer or "insufficient evidence",
            "nvc": float(attempt.nvc or 0.0),
            "sc_conf": float(attempt.sc_conf or 0.0),
            "dinco_conf": float(attempt.dinco_conf or 0.0),
            "dinco_candidates": list(attempt.dinco_candidates),
            "dinco_ptrues": list(attempt.dinco_ptrues),
            "final_answer_source": "memory_only_final_stage" if memory_passages else "insufficient_evidence",
            "final_answer_debug": {
                "mode": "memory_only_final_stage",
                "memory_entry_count": len(state.appended_entries),
                "selected_answer": pred_answer,
                "selected_raw_answer": pred_raw,
                "selected_nvc": attempt.nvc,
                "selected_sc_conf": attempt.sc_conf,
                "selected_dinco_conf": attempt.dinco_conf,
                "selected_source": attempt.source if attempt.available else "insufficient_evidence",
                "selected_support_claims": list(attempt.support_claims),
                "fallback_used": not bool(attempt.available),
            },
        }

    def _build_policy_trace(
        self,
        planning_trace: Sequence[Dict[str, Any]],
        pred_answer: str,
        nvc: float,
        dinco_conf: float,
        appended_entry_count: int,
    ) -> List[Dict[str, Any]]:
        policy_trace: List[Dict[str, Any]] = []
        step = 0
        for event in planning_trace:
            step += 1
            action = str(event.get("event", "")).strip() or "event"
            policy_trace.append(
                {
                    "step": step,
                    "action": action,
                    "node_id": event.get("node_id"),
                    "details": event,
                }
            )
        step += 1
        policy_trace.append(
            {
                "step": step,
                "action": "finalize",
                "node_id": None,
                "details": {
                    "pred_answer": pred_answer,
                    "nvc": float(nvc),
                    "dinco_conf": float(dinco_conf),
                    "appended_entry_count": int(appended_entry_count),
                },
            }
        )
        return policy_trace

    def run_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = str(example.get("question", "")).strip()
        gold_answer = str(example.get("answer", "")).strip()
        passages = build_passages(example)
        full_context = format_evidence(passages)

        state = PipelineState(full_context=full_context)
        initial_nodes = self._build_plan_nodes(state=state, question=question)
        for node in initial_nodes:
            state.nodes[node.id] = node
        state.planning_trace.append(
            {
                "event": "plan_initialized",
                "node_ids": [node.id for node in initial_nodes],
                "node_count": len(initial_nodes),
            }
        )

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
        em = exact_match(pred_answer, gold_answer)
        f1 = hotpot_answer_f1(pred_answer, gold_answer)
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

        subquestion_graph = {
            node_id: {
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
            }
            for node_id, node in state.nodes.items()
        }

        resolved_nodes = [node for node in state.nodes.values() if node.status in SUCCESS_STATUSES]
        decomposed_count = sum(
            1 for event in state.planning_trace if str(event.get("event", "")) == "decompose_success"
        )

        return {
            "id": example.get("id"),
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
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen planner-memory multihop QA pipeline on HotpotQA"
    )

    parser.add_argument("--dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument("--dataset_subset", type=str, default="distractor")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument(
        "--example_id",
        type=str,
        default=None,
        help="Run one specific example id (overrides --start_idx/--max_examples).",
    )

    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--qwen_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--planner_max_new_tokens", type=int, default=800)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--n_distractors", type=int, default=5)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--max_initial_subquestions", type=int, default=4)
    parser.add_argument("--max_subquestion_depth", type=int, default=2)
    parser.add_argument("--max_subquestion_nodes", type=int, default=12)
    parser.add_argument("--confidence_threshold", type=float, default=0.80)
    parser.add_argument("--retrieval_top_k", type=int, default=4)

    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help=(
            "Output JSONL filename (relative paths are written under results). "
            "Supports {dataset_name},{dataset_subset},{split},{slice_tag},{example_tag},{model},{mode},{seed}. "
            "If omitted, a name is auto-generated from CLI params."
        ),
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help=(
            "Summary JSON filename (relative paths are written under results). "
            "Supports the same placeholders as --output_jsonl. "
            "If omitted, defaults to <output>.summary.json."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_path, summary_path = _resolve_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        planner = MockQwenPlannerModel()
        qwen_model = MockQwenModel()
        dinco = MockDincoCalibrator()
    else:
        qwen_model = QwenDincoModel(
            model_name=args.qwen_model_name,
            cache_dir=args.cache_dir,
            dtype=args.qwen_dtype,
        )
        planner = QwenPlannerModel(
            qwen_model=qwen_model,
            max_new_tokens=args.planner_max_new_tokens,
            max_retries=args.max_retries,
        )
        dinco = DincoCalibrator(qwen_model=qwen_model, cache_dir=args.cache_dir)

    runner = PlannerMemoryRunner(
        planner=planner,
        qwen_model=qwen_model,
        subquestion_qwen_model=None,
        dinco=dinco,
        max_initial_subquestions=args.max_initial_subquestions,
        max_subquestion_depth=args.max_subquestion_depth,
        max_subquestion_nodes=args.max_subquestion_nodes,
        confidence_threshold=args.confidence_threshold,
        retrieval_top_k=args.retrieval_top_k,
        n_distractors=args.n_distractors,
    )

    ds = load_dataset(args.dataset_name, args.dataset_subset, split=args.split)
    ds, selected_start_idx, selected_end_idx = _slice_dataset(
        ds=ds,
        start_idx=args.start_idx,
        max_examples=args.max_examples,
        example_id=args.example_id,
    )

    metrics = {
        "count": 0,
        "em_sum": 0.0,
        "f1_sum": 0.0,
        "nvc": [],
        "subquestions": [],
        "resolved_subquestions": [],
        "decompositions": [],
        "appended_entries": [],
    }

    with output_path.open("w", encoding="utf-8") as writer:
        desc = "Running Qwen planner-memory pipeline"
        for ex in tqdm(ds, desc=desc):
            rec = runner.run_example(ex)
            writer.write(json.dumps(rec, ensure_ascii=True) + "\n")
            writer.flush()

            metrics["count"] += 1
            metrics["em_sum"] += float(rec["em"])
            metrics["f1_sum"] += float(rec.get("f1", 0.0))
            metrics["nvc"].append(float(rec["nvc"]))
            stats = rec.get("subgoal_stats", {})
            metrics["subquestions"].append(float(stats.get("total_nodes", 0)))
            metrics["resolved_subquestions"].append(float(stats.get("resolved_nodes", 0)))
            metrics["decompositions"].append(float(stats.get("decomposed_nodes", 0)))
            metrics["appended_entries"].append(float(stats.get("appended_entries", 0)))

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "split": args.split,
        "start_idx": selected_start_idx,
        "end_idx": selected_end_idx,
        "example_id": args.example_id,
        "count": metrics["count"],
        "em": (metrics["em_sum"] / metrics["count"]) if metrics["count"] else math.nan,
        "f1": (metrics["f1_sum"] / metrics["count"]) if metrics["count"] else math.nan,
        "avg_nvc": float(np.mean(metrics["nvc"])) if metrics["nvc"] else math.nan,
        "avg_total_nodes": float(np.mean(metrics["subquestions"])) if metrics["subquestions"] else math.nan,
        "avg_resolved_nodes": float(np.mean(metrics["resolved_subquestions"])) if metrics["resolved_subquestions"] else math.nan,
        "avg_decomposed_nodes": float(np.mean(metrics["decompositions"])) if metrics["decompositions"] else math.nan,
        "avg_appended_entries": float(np.mean(metrics["appended_entries"])) if metrics["appended_entries"] else math.nan,
        "config": vars(args),
    }

    with summary_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, ensure_ascii=True, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
