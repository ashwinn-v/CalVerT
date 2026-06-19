#!/usr/bin/env python3
"""
Agent-gated calibrated retrieval for HotpotQA.

Replaces the hardcoded threshold gates in the origbeam pipeline with an LLM
agent that receives DINCO confidence and MiniCheck grounding telemetry and
decides the next action (commit, retrieve, refine, or decompose).

Research question: Can an LLM effectively use numerical confidence/grounding
telemetry to decide when to retrieve, rather than relying on hardcoded
thresholds?
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    BASE_DIR,
    normalize_limit,
    slugify_name,
    write_json,
)
from retrieval_index import BM25Index, SearchHit

LLMRECOURSE_DIR = Path(__file__).resolve().parents[1]
if str(LLMRECOURSE_DIR) not in sys.path:
    sys.path.insert(0, str(LLMRECOURSE_DIR))

DINCO_DIR = LLMRECOURSE_DIR / "dinco"
if str(DINCO_DIR) not in sys.path:
    sys.path.insert(0, str(DINCO_DIR))

from telemetry_agent.dinco import triviaqa as dinco_base  # noqa: E402
from telemetry_agent.runners import _hotpot_utils as hotpot_utils  # noqa: E402


def _hybrid_or_bm25_search(
    index,
    dense_bundle,
    question_id,
    query,
    top_k,
    dense_top_k=25,
    rrf_k=60,
):
    """Route a retrieval call to hybrid (BM25 + dense, RRF-fused) when a
    dense_bundle is attached to the runner, else fall back to plain BM25
    via base_runner.search_passages. Drop-in for the existing retrieval call
    pattern in this runner.

    Used in 3 call sites in AgentGatedPlannerRunner._run_subquestion (retrieve,
    refine-fallback, react-retrieve). When --use_hybrid_retrieval is off,
    behaviour is identical to the legacy BM25-only path.
    """
    if dense_bundle is None:
        return base_runner.search_passages(index, question_id, query, top_k)
    from telemetry_agent.retrieval.hybrid_retriever import hybrid_search  # lazy import
    return hybrid_search(
        bm25_index=index,
        dense_bundle=dense_bundle,
        question_id=question_id,
        query=query,
        bm25_top_k=top_k,
        dense_top_k=int(dense_top_k),
        fused_top_k=top_k,
        rrf_k=int(rrf_k),
        use_per_qid_first=True,
    )
from telemetry_agent.planner import qwen_planner as planner_utils  # noqa: E402
from telemetry_agent.runners import _base_runner as base_runner  # noqa: E402
from telemetry_agent.runners import _origbeam as origbeam  # noqa: E402


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


HOTPOT_AGENT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["commit", "retrieve", "refine", "decompose"]},
        "query": {"type": "string"},
        "analysis": {"type": "string"},
        "reason": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["action"],
}


@dataclass
class AgentAction:
    """A single action chosen by the agent in one turn of the loop."""

    type: str  # "commit" | "retrieve" | "refine" | "decompose"
    query: str = ""  # search query (for retrieve action)
    reason: str = ""  # agent's stated reasoning
    analysis: str = ""  # agent's analysis of the telemetry
    raw_text: str = ""  # raw LLM output for debugging
    commit_answer: str = ""  # ReAct Finish[answer] — proposed answer override


@dataclass
class AgentTurnRecord:
    """Snapshot of state and action for one agent turn."""

    turn: int
    answer: str
    nvc: Optional[float]
    sc_conf: Optional[float]
    dinco_conf: Optional[float]
    g_mean: Optional[float]
    g_min: Optional[float]
    claim_supports: List[float]
    claims: List[str]
    action: AgentAction
    # Post-retrieval sampling-DINCO telemetry (None on closed-book turns or
    # when --enable_sampling_dinco_telemetry is off).
    sampling_dinco_conf: Optional[float] = None
    sampling_dinco_degenerate: Optional[bool] = None
    sampling_dinco_agreement_rate: Optional[float] = None
    sampling_dinco_n_unique: Optional[int] = None


# ---------------------------------------------------------------------------
# Agent prompt templates
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    'You are a retrieval policy controller for multi-hop question answering.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question. '
    'After each action you take, the system computes numerical telemetry from confidence and grounding '
    'models and shows it to you. You must decide what to do next.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Return STRICT JSON with exactly one action:\n'
    '\n'
    'When you commit, you MUST include an "answer" field with your final answer string. '
    'Commits without an answer field are invalid.\n'
    '{"action": "commit", "answer": "final answer string", "analysis": "your analysis of the telemetry", '
    '"reason": "why you are committing"}\n'
    '  Accept the current answer. Use when confidence is high AND the answer is well-grounded in evidence '
    '(or when you are confident the answer is correct from parametric knowledge for simple factoid questions).\n'
    '\n'
    '{"action": "retrieve", "query": "optional search query", "analysis": "your analysis of the telemetry", '
    '"reason": "why you need more evidence"}\n'
    '  Search for more evidence passages. The query defaults to the subquestion text if omitted. '
    'Use when confidence is low or grounding is insufficient.\n'
    '\n'
    '{"action": "refine", "analysis": "your analysis of the telemetry", "reason": "why refinement would help"}\n'
    '  Re-generate an answer using the same evidence with a refinement prompt. Use when grounding scores '
    'suggest the evidence is adequate but the answer or its claims are poorly formulated.\n'
    '\n'
    '{"action": "decompose", "analysis": "your analysis of the telemetry", "reason": "why this question needs decomposition"}\n'
    '  Give up on this subquestion and break it into smaller pieces. Use as a last resort when retrieval '
    'and refinement have not helped.\n'
    '\n'
    '## Telemetry Signals\n'
    '\n'
    'You will receive these numerical signals. Use them to inform your decision:\n'
    '\n'
    '- DINCO confidence (final_conf): Combined NVC + self-consistency score in [0,1]. Higher means the model '
    'is more internally consistent about the answer.\n'
    '- NVC (normalized verbal confidence): Verbal confidence normalized by how contradictory the alternative '
    'beam candidates are.\n'
    '- SC confidence: Self-consistency -- what fraction of independent samples agree with the main answer.\n'
    '- MiniCheck grounding (g_mean, g_min): How well the answer\'s support claims are grounded in retrieved '
    'evidence. Only available after retrieval.\n'
    '- Per-claim support scores: Individual grounding probability for each support claim.\n'
    '\n'
    '## Decision Principles\n'
    '\n'
    '1. Think before acting. Explain your reasoning in the "analysis" field before choosing an action.\n'
    '2. An answer without supporting evidence is risky for non-trivial questions.\n'
    '3. Diminishing returns. If multiple retrievals haven\'t helped, decomposition may be the right move.\n'
    '4. Budget awareness. You have a limited number of turns.\n'
    '\n'
    'Return STRICT JSON only. No markdown, no extra text outside the JSON object.'
)

AGENT_TURN_PROMPT = """## Turn {turn}/{max_turns} -- Subquestion: "{subquestion}"

### Original Multi-Hop Question
{original_question}

### Current State
- Current answer: "{answer}"
- Answer source: {source}
- Has dependency memory from earlier subquestions: {has_dependency_memory}
- Evidence passages retrieved so far: {n_passages}

### Closed-Book Confidence Telemetry
{closed_book_confidence_block}

### Grounding Telemetry (MiniCheck)
{grounding_block}

### Action History
{action_history_block}

### Budget
{budget_remaining} turns remaining. Choose your next action wisely.

Return STRICT JSON only: {{"action": "...", "analysis": "...", "reason": "..."}}"""

AGENT_TURN_DELTA_PROMPT = """## Turn {turn}/{max_turns} — Update

### Action Executed
Your previous action **{prev_action}** has been executed.{action_outcome}

### Current Answer
"{answer}"

### Closed-Book Confidence Telemetry (Beam-DINCO, computed before retrieval)
- **DINCO final confidence:** {dinco_conf}{dinco_delta}
- **NVC:** {nvc}
- **SC confidence:** {sc_conf}

### Grounding Telemetry (MiniCheck)
{grounding_block}

### Post-Retrieval Sampling-DINCO Telemetry
{sampling_dinco_block}

### Budget
{budget_remaining} turns remaining.

Return STRICT JSON only: {{"action": "...", "analysis": "...", "reason": "..."}}"""

NO_GROUNDING_BLOCK = "Not yet available — no retrieval has been performed. Grounding scores require evidence passages."

GROUNDING_BLOCK_TEMPLATE = """- **g_mean (average claim support):** {g_mean:.3f}
- **g_min (worst claim support):** {g_min:.3f}
- **Per-claim scores:**
{per_claim_lines}"""

NO_SAMPLING_DINCO_BLOCK = (
    "Not yet available — no retrieval has been performed. "
    "Post-retrieval sampling-DINCO is only computed once retrieval has run."
)

SAMPLING_DINCO_DISABLED_BLOCK = (
    "Not enabled for this run — `--enable_sampling_dinco_telemetry` is off."
)

SAMPLING_DINCO_BLOCK_TEMPLATE = """- **Sampling-DINCO confidence (post-retrieval):** {sampling_dinco_conf:.3f}
- **Degenerate flag:** {degenerate}{degenerate_note}
- **Agreement rate (samples agreeing with current answer):** {agreement_rate:.3f}
- **Unique distractors:** {n_unique}
- **Sampled distractor candidates (with P(true)):**
{sampling_distractor_lines}"""

# ---------------------------------------------------------------------------
# Prompt variants for telemetry-framing ablation
# (experiment: agent-telemetry-prompt-framing)
#
# Four variants share the same "Available Actions" and "Telemetry Signals"
# scaffolding. They differ only in the Decision Principles section and (for
# `role`) an extra Signal Roles section. The `narrative` variant additionally
# runs all numeric scores through a qualitative-bin formatter at the call site.
# ---------------------------------------------------------------------------

# Variant A baseline: AGENT_SYSTEM_PROMPT (defined above) is used directly.

_PROMPT_INFO = (
    'You are a retrieval policy controller for multi-hop question answering.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question. '
    'After each action you take, the system computes numerical telemetry from confidence and grounding '
    'models and shows it to you. You must decide what to do next.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Return STRICT JSON with exactly one action:\n'
    '\n'
    '{"action": "commit", "analysis": "your analysis of the telemetry", "reason": "why you are committing"}\n'
    '  Accept the current answer. Use when your analysis of the available signals supports that the answer '
    'is correct and well-supported.\n'
    '\n'
    '{"action": "retrieve", "query": "optional search query", "analysis": "your analysis of the telemetry", '
    '"reason": "why you need more evidence"}\n'
    '  Search for more evidence passages. The query defaults to the subquestion text if omitted. '
    'Use when the signals suggest the answer would benefit from additional evidence.\n'
    '\n'
    '{"action": "refine", "analysis": "your analysis of the telemetry", "reason": "why refinement would help"}\n'
    '  Re-generate an answer using the same evidence with a refinement prompt. Use when the evidence appears '
    'relevant but the answer or its claims are poorly formulated against that evidence.\n'
    '\n'
    '{"action": "decompose", "analysis": "your analysis of the telemetry", "reason": "why this question needs decomposition"}\n'
    '  Give up on this subquestion and break it into smaller pieces. Use as a last resort when retrieval '
    'and refinement have not helped.\n'
    '\n'
    '## Telemetry Signals\n'
    '\n'
    'You will receive these numerical signals as inputs to your reasoning. They are not commands; they are '
    'evidence about the current state of your answer:\n'
    '\n'
    '- DINCO confidence (final_conf): Combined NVC + self-consistency score in [0,1]. Higher means the model '
    'is more internally consistent about the answer. Measures how much the model "agrees with itself" across '
    'multiple generation attempts.\n'
    '- NVC (normalized verbal confidence): Verbal confidence normalized by how contradictory the alternative '
    'beam candidates are. A high NVC with contradictory alternatives is more meaningful than a high NVC with '
    'similar alternatives.\n'
    '- SC confidence: Self-consistency — what fraction of independent samples agree with the main answer.\n'
    '- Beam candidates: Alternative answers the model considered, with P(true) scores. The relative spread '
    'between the top candidate and its alternatives carries information about how concentrated the model\'s '
    'belief is.\n'
    '- MiniCheck grounding (g_mean, g_min): How well the answer\'s support claims are grounded in retrieved '
    'evidence. g_mean is average grounding across claims, g_min is the worst-grounded claim. These are only '
    'available after retrieval.\n'
    '- Per-claim support scores: Individual grounding probability for each support claim. Examine the '
    'distribution; a single very weak claim can pull g_min down even when other claims are well-grounded.\n'
    '- Post-retrieval sampling-DINCO confidence: NVC computed over N stochastic samples conditioned on the '
    'retrieved passages. This is DIFFERENT from the closed-book DINCO above (which is computed before any '
    'retrieval); this signal reflects how much the model agrees with itself given the retrieved context. '
    'A large drop from closed-book DINCO to post-retrieval DINCO means the retrieval changed the model\'s '
    'belief, which is itself useful information.\n'
    '- Degenerate flag: TRUE means the model produced 0 unique distractors after dedupe (samples all '
    'collapsed onto the current answer). This is information, not a verdict — the same flag can arise either '
    'when the answer is trivially extractable from the passages OR when the model is locked onto a single '
    'wrong answer regardless of evidence. Cross-reference with grounding signals to interpret it.\n'
    '- Agreement rate: fraction of stochastic samples that match the current answer after normalization. '
    'Mechanically near 1.0 when degenerate=TRUE; useful when degenerate=FALSE as a self-consistency proxy.\n'
    '\n'
    '## Decision Principles\n'
    '\n'
    '1. Reason before acting. Examine the telemetry signals and explain in the "analysis" field what they '
    'tell you about the current state of the answer before choosing an action.\n'
    '2. The signals describe different aspects of the answer. Weigh them together rather than reading any '
    'one signal as a binary trigger; an answer can be self-consistent yet unsupported, or supported yet '
    'self-inconsistent, and the appropriate action depends on the joint state.\n'
    '3. When grounding scores after retrieval are low, the evidence does not yet support the answer. Either '
    'the evidence is irrelevant (try a different retrieval query), or the answer is poorly stated against '
    'relevant evidence (refine).\n'
    '4. Diminishing returns. If multiple retrievals have not improved grounding, the subquestion may be too '
    'broad or compositional, and decomposition may be the right move.\n'
    '5. Budget awareness. You have a limited number of turns. Spend them on actions whose expected '
    'information gain is positive given the current signals — repeating an action that has not changed the '
    'signals is unlikely to help.\n'
    '6. The signals are inputs to your judgement, not gates. Two signals can disagree (high DINCO + low '
    'g_min, or low DINCO + high g_min). Such disagreement is itself information about the answer and '
    'often points to which action is most useful next.\n'
    '\n'
    'Return STRICT JSON only. No markdown, no extra text outside the JSON object.'
)

_PROMPT_ROLE = (
    'You are a retrieval policy controller for multi-hop question answering.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question. '
    'After each action you take, the system computes numerical telemetry from two distinct models — a '
    'self-confidence model and a grounding model — and shows it to you. You must decide what to do next.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Return STRICT JSON with exactly one action:\n'
    '\n'
    '{"action": "commit", "analysis": "your analysis of the telemetry", "reason": "why you are committing"}\n'
    '  Accept the current answer. Use when your analysis of the signals supports that the answer is correct '
    'and well-supported.\n'
    '\n'
    '{"action": "retrieve", "query": "optional search query", "analysis": "your analysis of the telemetry", '
    '"reason": "why you need more evidence"}\n'
    '  Search for more evidence passages. The query defaults to the subquestion text if omitted.\n'
    '\n'
    '{"action": "refine", "analysis": "your analysis of the telemetry", "reason": "why refinement would help"}\n'
    '  Re-generate an answer using the same evidence with a refinement prompt.\n'
    '\n'
    '{"action": "decompose", "analysis": "your analysis of the telemetry", "reason": "why this question needs decomposition"}\n'
    '  Give up on this subquestion and break it into smaller pieces. Use as a last resort.\n'
    '\n'
    '## Signal Roles (read this carefully)\n'
    '\n'
    'You will see two families of signals. They measure different things and should be reasoned about '
    'separately:\n'
    '\n'
    '- **DINCO family (DINCO confidence, NVC, SC, sampling-DINCO):** these come from the GENERATOR model '
    'reasoning about its own answer. They measure SELF-CONFIDENCE — how much the model agrees with itself. '
    'A high DINCO does NOT mean the answer is correct; it means the model has a stable internal belief. '
    'Models can be confidently wrong.\n'
    '- **MiniCheck family (g_mean, g_min, per-claim support):** these come from a SEPARATE GROUNDING MODEL '
    'that asks: do the retrieved passages support the claims in the current answer? MiniCheck is a '
    'GROUNDING signal, not a reasoning-quality score. A high g_min means the retrieved evidence supports '
    'the claims; a low g_min means the evidence does not support the claims (regardless of whether the '
    'answer is actually correct).\n'
    '- **These two families are orthogonal.** All four combinations occur (high/low DINCO × high/low '
    'g_min) and each carries different information about whether the answer is well-supported.\n'
    '\n'
    '## Telemetry Signals\n'
    '\n'
    '- DINCO confidence (final_conf): Combined NVC + self-consistency score in [0,1]. Higher means the '
    'model agrees with itself more consistently across attempts.\n'
    '- NVC (normalized verbal confidence): Verbal confidence normalized by how contradictory the alternative '
    'beam candidates are. A high NVC with contradictory alternatives is more meaningful than a high NVC with '
    'similar alternatives.\n'
    '- SC confidence: Self-consistency — what fraction of independent samples agree with the main answer.\n'
    '- Beam candidates: Alternative answers the model considered, with P(true) scores. The spread between '
    'top candidate and alternatives reflects how concentrated the model\'s belief is.\n'
    '- MiniCheck g_mean, g_min, per-claim support: grounding scores in [0,1] from the separate grounding '
    'model. Higher means the retrieved passages better support the claim. Available only after retrieval. '
    'A single very weak claim can pull g_min down even when other claims are well-grounded.\n'
    '- Post-retrieval sampling-DINCO: self-consistency computed over N stochastic samples conditioned on '
    'retrieved passages. A drop from closed-book DINCO to post-retrieval DINCO means retrieval changed the '
    'model\'s belief; that change is itself information about the answer.\n'
    '- Degenerate flag: TRUE means samples collapsed onto the current answer. The flag does not by itself '
    'imply correctness — interpret it together with grounding scores.\n'
    '- Agreement rate: fraction of stochastic samples that match the current answer after normalization.\n'
    '\n'
    '## Decision Principles\n'
    '\n'
    '1. Reason about both signal families before acting. In the "analysis" field, mention what each family '
    '(self-confidence vs grounding) is telling you about the current state of the answer.\n'
    '2. Treat agreement and disagreement between the two families as informative — disagreement is often '
    'where the most useful action is decided. Confidently unsupported answers warrant more retrieval; '
    'uncertain but supported answers may be ready to commit.\n'
    '3. When grounding is low after retrieval, the evidence does not yet support the answer. Either retrieve '
    'with a different query (if evidence seems irrelevant) or refine (if evidence seems relevant but the '
    'answer is poorly stated against it).\n'
    '4. Diminishing returns. If multiple retrievals have not improved grounding, the subquestion may be too '
    'broad and decomposition may help.\n'
    '5. Budget awareness. You have a limited number of turns. Spend them on actions whose expected '
    'information gain is positive — repeating an action that has not changed the signals is unlikely to help.\n'
    '\n'
    'Return STRICT JSON only. No markdown, no extra text outside the JSON object.'
)

_PROMPT_NARRATIVE = (
    'You are a retrieval policy controller for multi-hop question answering.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question. '
    'After each action you take, the system computes telemetry from confidence and grounding models and '
    'shows it to you in qualitative form. You must decide what to do next.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Return STRICT JSON with exactly one action:\n'
    '\n'
    '{"action": "commit", "analysis": "your analysis of the telemetry", "reason": "why you are committing"}\n'
    '  Accept the current answer. Use when your analysis of the signals supports that the answer is correct.\n'
    '\n'
    '{"action": "retrieve", "query": "optional search query", "analysis": "your analysis of the telemetry", '
    '"reason": "why you need more evidence"}\n'
    '  Search for more evidence passages. The query defaults to the subquestion text if omitted.\n'
    '\n'
    '{"action": "refine", "analysis": "your analysis of the telemetry", "reason": "why refinement would help"}\n'
    '  Re-generate an answer using the same evidence with a refinement prompt.\n'
    '\n'
    '{"action": "decompose", "analysis": "your analysis of the telemetry", "reason": "why this question needs decomposition"}\n'
    '  Give up on this subquestion and break it into smaller pieces. Use as a last resort.\n'
    '\n'
    '## Telemetry Signals (qualitative)\n'
    '\n'
    'All scores below are presented as qualitative bins: VERY LOW / LOW / MEDIUM / HIGH / VERY HIGH. '
    'These reflect five graded levels of strength on a [0,1] underlying scale; reason about the relative '
    'strength rather than expecting precise numbers. Two signals at "high" are not necessarily equally '
    'strong — they fall in the same bin but the relevant comparison is across signals at the same time.\n'
    '\n'
    '- DINCO confidence (final_conf): how much the model agrees with itself across multiple generation '
    'attempts. Higher means the model is more internally consistent about the answer; it does not directly '
    'measure correctness.\n'
    '- NVC (normalized verbal confidence): verbal confidence normalized by how contradictory the alternative '
    'beam candidates are. A high NVC with contradictory alternatives carries more weight than a high NVC '
    'with similar alternatives.\n'
    '- SC confidence: self-consistency — what fraction of independent samples agree with the main answer.\n'
    '- Beam candidates: alternative answers the model considered, each with a strength label. The spread '
    'between top candidate and alternatives reflects how concentrated the model\'s belief is.\n'
    '- MiniCheck grounding (g_mean, g_min): grounding strength of the answer\'s support claims against '
    'retrieved evidence. g_mean is average across claims; g_min is the worst-grounded claim. Available only '
    'after retrieval. A single very weak claim can pull g_min down even when other claims are well-grounded.\n'
    '- Per-claim support: grounding strength of each individual support claim against the retrieved evidence.\n'
    '- Post-retrieval sampling-DINCO: model self-agreement computed over stochastic samples conditioned on '
    'retrieved passages. This is DIFFERENT from the closed-book DINCO above (which is computed before any '
    'retrieval); a drop from closed-book to post-retrieval DINCO means retrieval changed the model\'s belief, '
    'and that change is itself useful information.\n'
    '- Degenerate flag: TRUE if samples collapsed onto the current answer. This is information, not a verdict '
    '— it can arise either when the answer is trivially extractable from the passages, or when the model is '
    'locked onto a single wrong answer regardless of evidence. Cross-reference with grounding to interpret.\n'
    '- Agreement rate: how strongly stochastic samples agreed with the current answer after normalization. '
    'Mechanically near "very high" when degenerate=TRUE; useful when degenerate=FALSE as a self-consistency '
    'proxy.\n'
    '\n'
    '## Decision Principles\n'
    '\n'
    '1. Reason before acting. Examine the qualitative signals and explain in "analysis" what each signal '
    'says about the current state of the answer.\n'
    '2. The signals describe different aspects of the answer. Weigh them together; do not read any one '
    'signal as a binary trigger. An answer can be self-consistent yet unsupported, or supported yet '
    'self-inconsistent, and the appropriate action depends on the joint state.\n'
    '3. When grounding is weak after retrieval, the evidence does not yet support the answer. Either '
    'retrieve again with a different query (if evidence seems irrelevant), or refine (if evidence seems '
    'relevant but the answer is poorly stated against it).\n'
    '4. Diminishing returns. If multiple retrievals have not improved grounding, the subquestion may be too '
    'broad and decomposition may help.\n'
    '5. Budget awareness. You have a limited number of turns. Spend them where expected information gain '
    'is positive — repeating an action that has not changed the signals is unlikely to help.\n'
    '6. The signals are inputs to your judgement, not gates. Disagreement between signals (high DINCO + '
    'low g_min, or vice versa) is itself information about the answer and often points to which action is '
    'most useful next.\n'
    '\n'
    'Return STRICT JSON only. No markdown, no extra text outside the JSON object.'
)


# Score-bin helper for `narrative` mode. Five evenly spaced bins on [0,1].
_NARRATIVE_BIN_LABELS = ("very low", "low", "medium", "high", "very high")


def _score_to_bin(x: Optional[float]) -> str:
    """Return a qualitative bin label for a [0,1] score. None → 'n/a'."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v < 0.2:
        return _NARRATIVE_BIN_LABELS[0]
    if v < 0.4:
        return _NARRATIVE_BIN_LABELS[1]
    if v < 0.6:
        return _NARRATIVE_BIN_LABELS[2]
    if v < 0.8:
        return _NARRATIVE_BIN_LABELS[3]
    return _NARRATIVE_BIN_LABELS[4]


def _get_system_prompt(mode: str) -> str:
    """Dispatch the system prompt for an `agent_telemetry_mode`."""
    if mode == "no_telemetry":
        return AGENT_SYSTEM_PROMPT_NO_TELEMETRY
    if mode == "info":
        return _PROMPT_INFO
    if mode == "role":
        return _PROMPT_ROLE
    if mode == "narrative":
        return _PROMPT_NARRATIVE
    return AGENT_SYSTEM_PROMPT  # "full" / default


# ---------------------------------------------------------------------------
# No-telemetry prompt variants (for ablation)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT_NO_TELEMETRY = (
    'You are a retrieval policy controller for multi-hop question answering.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question. '
    'After each action you take, the system updates the state and you must decide what to do next.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Return STRICT JSON with exactly one action:\n'
    '\n'
    '{"action": "commit", "analysis": "your reasoning", "reason": "why you are committing"}\n'
    '  Accept the current answer. Use when you believe the answer is correct.\n'
    '\n'
    '{"action": "retrieve", "query": "optional search query", "analysis": "your reasoning", '
    '"reason": "why you need more evidence"}\n'
    '  Search for more evidence passages. The query defaults to the subquestion text if omitted.\n'
    '\n'
    '{"action": "refine", "analysis": "your reasoning", "reason": "why refinement would help"}\n'
    '  Re-generate an answer using the same evidence with a refinement prompt.\n'
    '\n'
    '{"action": "decompose", "analysis": "your reasoning", "reason": "why this question needs decomposition"}\n'
    '  Give up on this subquestion and break it into smaller pieces. Use as a last resort.\n'
    '\n'
    '## Decision Principles\n'
    '\n'
    '1. Think before acting. Explain your reasoning in the "analysis" field before choosing an action.\n'
    '2. An answer without supporting evidence is risky for non-trivial questions.\n'
    '3. Diminishing returns. If multiple retrievals haven\'t helped, decomposition may be the right move.\n'
    '4. Budget awareness. You have a limited number of turns.\n'
    '\n'
    'Return STRICT JSON only. No markdown, no extra text outside the JSON object.'
)

AGENT_TURN_PROMPT_NO_TELEMETRY = """## Turn {turn}/{max_turns} — Subquestion: "{subquestion}"

### Original Multi-Hop Question
{original_question}

### Current State
- **Current answer:** "{answer}"
- **Answer source:** {source}
- **Has dependency memory from earlier subquestions:** {has_dependency_memory}
- **Evidence passages retrieved so far:** {n_passages}

### Action History
{action_history_block}

### Budget
{budget_remaining} turns remaining. Choose your next action wisely.

Return STRICT JSON only: {{"action": "...", "analysis": "...", "reason": "..."}}"""

AGENT_TURN_DELTA_PROMPT_NO_TELEMETRY = """## Turn {turn}/{max_turns} — Update

### Action Executed
Your previous action **{prev_action}** has been executed.{action_outcome}

### Current Answer
"{answer}"

### Evidence passages retrieved so far: {n_passages}

### Budget
{budget_remaining} turns remaining.

Return STRICT JSON only: {{"action": "...", "analysis": "...", "reason": "..."}}"""

# ---------------------------------------------------------------------------
# ReAct prompt templates
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = (
    'Solve a multi-hop question by interleaving Thought, Action, and Observation steps.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Action: Search[query] — Search for evidence passages related to the query.\n'
    'Action: Finish[answer] — Submit your final answer for this subquestion.\n'
    '\n'
    '## Format\n'
    '\n'
    'Always respond in this exact format:\n'
    'Thought: <your reasoning about what to do next>\n'
    'Action: <Search[query] or Finish[answer]>\n'
    '\n'
    'Do NOT include anything after the Action line. The system will provide Observation after Search.'
)

REACT_SYSTEM_PROMPT_TELEMETRY = (
    'Solve a multi-hop question by interleaving Thought, Action, and Observation steps.\n'
    '\n'
    'You are processing one subquestion at a time as part of answering a larger multi-hop question.\n'
    '\n'
    '## Available Actions\n'
    '\n'
    'Action: Search[query] — Search for evidence passages related to the query.\n'
    'Action: Finish[answer] — Submit your final answer for this subquestion.\n'
    '\n'
    '## Format\n'
    '\n'
    'Always respond in this exact format:\n'
    'Thought: <your reasoning about what to do next, referencing the telemetry signals>\n'
    'Action: <Search[query] or Finish[answer]>\n'
    '\n'
    'Do NOT include anything after the Action line. The system will provide Observation after Search.\n'
    '\n'
    '## Telemetry Signals\n'
    '\n'
    'You will receive numerical signals alongside passage text. Use them to inform your decision:\n'
    '\n'
    '- DINCO confidence (final_conf): Combined NVC + self-consistency score in [0,1]. Higher means the model '
    'is more internally consistent about the answer.\n'
    '- NVC (normalized verbal confidence): Verbal confidence normalized by how contradictory the alternative '
    'beam candidates are.\n'
    '- SC confidence: Self-consistency — what fraction of independent samples agree with the main answer.\n'
    '- MiniCheck grounding (g_mean, g_min): How well the answer\'s claims are grounded in retrieved '
    'evidence. Only available after Search. g_mean is average grounding, g_min is the worst-grounded claim.\n'
    '- Per-claim support scores: Individual grounding probability for each claim.\n'
    '\n'
    '## Decision Principles\n'
    '\n'
    '1. Use both the passage text AND the numerical telemetry to decide. High confidence + strong grounding = '
    'safe to Finish. Low grounding or contradictory evidence = Search more.\n'
    '2. High DINCO confidence without evidence is risky for non-trivial questions.\n'
    '3. Low grounding after retrieval means the evidence doesn\'t support the answer — Search with a '
    'different query.\n'
    '4. Budget awareness. Don\'t waste turns on searches unlikely to help.'
)

REACT_TURN1_PROMPT = """## Subquestion: "{subquestion}"

### Original Multi-Hop Question
{original_question}

### Current Closed-Book Answer
The model's initial answer (without evidence): "{answer}"

### Context
- **Has dependency memory from earlier subquestions:** {has_dependency_memory}
- **Turns remaining:** {budget_remaining}

Decide: is this answer reliable enough to submit, or should you search for supporting evidence?

Thought:"""

REACT_TURN1_PROMPT_TELEMETRY = """## Subquestion: "{subquestion}"

### Original Multi-Hop Question
{original_question}

### Current Closed-Book Answer
The model's initial answer (without evidence): "{answer}"

### Confidence Telemetry (DINCO)
- **DINCO final confidence:** {dinco_conf}
- **NVC (verbal confidence):** {nvc}
- **Self-consistency confidence:** {sc_conf}
- **Beam candidates (answer, P(true)):**
{beam_candidates_block}

### Context
- **Has dependency memory from earlier subquestions:** {has_dependency_memory}
- **Turns remaining:** {budget_remaining}

Decide: given the confidence scores and your assessment, is this answer reliable enough to submit, or should you search for supporting evidence?

Thought:"""

REACT_OBSERVATION_PROMPT = """Observation: Retrieved {n_new} passage(s). {n_total} total passages available.

{passages_text}

Thought:"""

REACT_OBSERVATION_PROMPT_TELEMETRY = """Observation: Retrieved {n_new} passage(s). {n_total} total passages available.

{passages_text}

### Grounding Telemetry (MiniCheck)
{grounding_block}

Thought:"""

REACT_FOLLOWUP_PROMPT = """The answer has been updated based on the retrieved evidence.

### Current Answer
"{answer}"

### Turns remaining: {budget_remaining}

Decide: is this answer well-supported by the evidence, or do you need to search for more?

Thought:"""

REACT_FOLLOWUP_PROMPT_TELEMETRY = """The answer has been updated based on the retrieved evidence.

### Current Answer
"{answer}"

### Confidence Telemetry (DINCO)
- **DINCO final confidence:** {dinco_conf}
- **NVC:** {nvc}
- **SC confidence:** {sc_conf}

### Grounding Telemetry (MiniCheck)
{grounding_block}

### Turns remaining: {budget_remaining}

Decide: given the confidence and grounding scores, is this answer well-supported, or do you need to search for more?

Thought:"""


def _format_passages_for_react(hits: Sequence["SearchHit"], max_passages: int = 5, max_chars: int = 500) -> str:
    """Format retrieved passages as readable text for ReAct observations."""
    if not hits:
        return "(no passages retrieved)"
    lines = []
    for i, hit in enumerate(hits[:max_passages]):
        title = str(hit.row.get("title") or "Untitled")
        text = str(hit.row.get("chunk_body_text") or hit.row.get("chunk_text") or "")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        lines.append(f"[{i+1}] **{title}**: {text}")
    if len(hits) > max_passages:
        lines.append(f"... and {len(hits) - max_passages} more passage(s)")
    return "\n\n".join(lines)


def parse_react_action(raw_text: str) -> AgentAction:
    """Parse ReAct-style 'Thought: ... Action: ...' output into an AgentAction."""
    import re

    # Extract thought
    thought = ""
    thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\Z)', raw_text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action
    action_match = re.search(r'Action:\s*(Search|Finish)\[(.+?)\]\s*$', raw_text, re.MULTILINE | re.IGNORECASE)
    if action_match:
        action_type_raw = action_match.group(1).strip().lower()
        action_arg = action_match.group(2).strip()
        if action_type_raw == "search":
            return AgentAction(
                type="retrieve",
                query=action_arg,
                reason=thought,
                analysis=thought,
                raw_text=raw_text,
            )
        elif action_type_raw == "finish":
            return AgentAction(
                type="commit",
                query="",
                reason=thought,
                analysis=thought,
                raw_text=raw_text,
                # Store the proposed answer in query field for extraction
                commit_answer=action_arg,
            )

    # Fallback: if no valid action parsed, default to retrieve with subquestion
    return AgentAction(type="retrieve", reason=f"parse_failure: {thought}", raw_text=raw_text)


# ---------------------------------------------------------------------------
# Agent action parsing
# ---------------------------------------------------------------------------

_VALID_ACTIONS = frozenset({"commit", "retrieve", "refine", "decompose"})


_THINK_BLOCK_RE_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE_THINKING = re.compile(r"<think>.*$", re.DOTALL)


def parse_agent_action(raw_text: str) -> AgentAction:
    """Parse agent JSON output into an AgentAction. Falls back to 'retrieve' on failure.

    THINKING-VARIANT: strip <think>...</think> blocks before JSON extraction.
    Without this, curly braces inside the model's thinking content would corrupt
    the brace-scanner in extract_json_dict.
    """
    stripped = _THINK_BLOCK_RE_THINKING.sub("", raw_text)
    stripped = _UNCLOSED_THINK_RE_THINKING.sub("", stripped)
    parsed = hotpot_utils.extract_json_dict(stripped)
    if parsed is None:
        return AgentAction(type="retrieve", reason="parse_failure", raw_text=raw_text)

    action_type = str(parsed.get("action", "")).strip().lower()
    if action_type not in _VALID_ACTIONS:
        action_type = "retrieve"

    return AgentAction(
        type=action_type,
        query=str(parsed.get("query", "")).strip(),
        reason=str(parsed.get("reason", "")).strip(),
        analysis=str(parsed.get("analysis", "")).strip(),
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------


def _format_beam_candidates(
    attempt: planner_utils.AttemptResult,
    max_candidates: int = 5,
    narrative: bool = False,
) -> str:
    candidates = list(attempt.dinco_candidates or [])[:max_candidates]
    ptrues = list(attempt.dinco_ptrues or [])
    if not candidates:
        return "  (no beam candidates available)"
    lines = []
    for i, cand in enumerate(candidates):
        ptrue = ptrues[i] if i < len(ptrues) else 0.0
        if narrative:
            lines.append(f'  {i+1}. "{cand}" — strength={_score_to_bin(ptrue)}')
        else:
            lines.append(f'  {i+1}. "{cand}" — P(true)={ptrue:.3f}')
    return "\n".join(lines)


def _format_grounding_block(
    support: Optional[Dict[str, Any]],
    claims: Sequence[str],
    narrative: bool = False,
) -> str:
    if support is None:
        return NO_GROUNDING_BLOCK

    g_mean = float(support.get("g_mean", 0.0))
    g_min = float(support.get("g_min", 0.0))
    claim_supports = list(support.get("claim_supports", []))

    per_claim_lines = []
    for i, claim in enumerate(claims):
        score = claim_supports[i] if i < len(claim_supports) else 0.0
        if narrative:
            per_claim_lines.append(f'  [{_score_to_bin(score)}] "{claim}"')
        else:
            per_claim_lines.append(f'  [{score:.3f}] "{claim}"')
    if not per_claim_lines:
        per_claim_lines = ["  (no claims generated)"]

    if narrative:
        return (
            f"- **g_mean (average claim support):** {_score_to_bin(g_mean)}\n"
            f"- **g_min (worst claim support):** {_score_to_bin(g_min)}\n"
            f"- **Per-claim scores:**\n" + "\n".join(per_claim_lines)
        )
    return GROUNDING_BLOCK_TEMPLATE.format(
        g_mean=g_mean,
        g_min=g_min,
        per_claim_lines="\n".join(per_claim_lines),
    )


def _format_sampling_dinco_block(
    attempt: planner_utils.AttemptResult,
    enabled: bool = True,
    n_passages: int = 0,
    max_distractors: int = 5,
    narrative: bool = False,
) -> str:
    """Render the post-retrieval sampling-DINCO telemetry block.

    Three states:
    - Disabled (feature flag off): explicit message so the agent doesn't expect a value.
    - Not yet available (no retrieval, or attempt didn't compute it): "Not yet available".
    - Available: filled template with degenerate/agreement/distractor lines.
    """
    if not enabled:
        return SAMPLING_DINCO_DISABLED_BLOCK
    if attempt.sampling_dinco_conf is None:
        # Either retrieval hasn't happened (n_passages == 0) or the runner
        # didn't compute the signal for this attempt. Either way, surface a
        # clear message rather than rendering with zeros.
        return NO_SAMPLING_DINCO_BLOCK

    distractors = list(attempt.sampling_distractors or [])[:max_distractors]
    ptrues = list(attempt.sampling_ptrues or [])
    if distractors:
        distractor_lines = []
        for i, cand in enumerate(distractors):
            ptrue = ptrues[i] if i < len(ptrues) else 0.0
            if narrative:
                distractor_lines.append(f'  {i+1}. "{cand}" — strength={_score_to_bin(ptrue)}')
            else:
                distractor_lines.append(f'  {i+1}. "{cand}" — P(true)={ptrue:.3f}')
        sampling_distractor_lines = "\n".join(distractor_lines)
    else:
        sampling_distractor_lines = "  (no unique distractors — samples collapsed onto the current answer)"

    degenerate = bool(attempt.sampling_dinco_degenerate)
    if degenerate:
        degenerate_note = (
            " — samples collapsed onto the current answer. "
            "Confidence falls back to raw P(true) on the answer; cross-check with MiniCheck g_min "
            "to disambiguate confidently-correct vs confidently-wrong."
        )
    else:
        degenerate_note = ""

    sampling_dinco_conf = float(attempt.sampling_dinco_conf)
    agreement_rate = float(attempt.sampling_dinco_agreement_rate or 0.0)

    if narrative:
        return (
            f"- **Sampling-DINCO confidence (post-retrieval):** {_score_to_bin(sampling_dinco_conf)}\n"
            f"- **Degenerate flag:** {'TRUE' if degenerate else 'FALSE'}{degenerate_note}\n"
            f"- **Agreement rate (samples agreeing with current answer):** {_score_to_bin(agreement_rate)}\n"
            f"- **Unique distractors:** {int(attempt.sampling_dinco_n_unique or 0)}\n"
            f"- **Sampled distractor candidates (with strength):**\n{sampling_distractor_lines}"
        )

    return SAMPLING_DINCO_BLOCK_TEMPLATE.format(
        sampling_dinco_conf=sampling_dinco_conf,
        degenerate="TRUE" if degenerate else "FALSE",
        degenerate_note=degenerate_note,
        agreement_rate=agreement_rate,
        n_unique=int(attempt.sampling_dinco_n_unique or 0),
        sampling_distractor_lines=sampling_distractor_lines,
    )


def _format_action_history(
    history: Sequence[AgentTurnRecord],
    max_entries: int = 8,
    include_telemetry: bool = True,
) -> str:
    if not history:
        return "(first turn — no previous actions)"
    entries = list(history)[-max_entries:]
    lines = []
    for record in entries:
        action = record.action
        parts = [
            f"- Turn {record.turn}: action={action.type}",
        ]
        if action.query:
            parts.append(f'  query="{action.query}"')
        if include_telemetry:
            telemetry_parts = []
            if record.dinco_conf is not None:
                telemetry_parts.append(f"dinco={record.dinco_conf:.3f}")
            if record.g_mean is not None:
                telemetry_parts.append(f"g_mean={record.g_mean:.3f}")
            if record.g_min is not None:
                telemetry_parts.append(f"g_min={record.g_min:.3f}")
            telemetry_str = ", ".join(telemetry_parts) if telemetry_parts else "n/a"
            parts.append(f"  telemetry: {telemetry_str}")
        parts.append(f'  answer: "{record.answer}"')
        if action.analysis:
            parts.append(f"  analysis: {action.analysis}")
        parts.append(f"  reason: {action.reason}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def format_agent_prompt(
    *,
    turn: int,
    max_turns: int,
    original_question: str,
    subquestion: str,
    attempt: planner_utils.AttemptResult,
    support: Optional[Dict[str, Any]],
    n_passages: int,
    has_dependency_memory: bool,
    history: Sequence[AgentTurnRecord],
    pre_attempt: Optional[planner_utils.AttemptResult] = None,
    sampling_dinco_enabled: bool = False,
    narrative: bool = False,
) -> str:
    """Build the per-turn user prompt for the agent.

    If *pre_attempt* is provided and the current *attempt* has no beam
    candidates (e.g. after retrieval replaced the closed-book attempt),
    the closed-book beam candidates from *pre_attempt* are shown instead
    so the agent retains visibility into the initial confidence landscape.

    When *narrative* is True, all numeric scores are rendered through
    `_score_to_bin` as qualitative bins.
    """
    if narrative:
        dinco_conf_str = _score_to_bin(attempt.dinco_conf)
        nvc_str = _score_to_bin(attempt.nvc)
        sc_conf_str = _score_to_bin(attempt.sc_conf)
    else:
        dinco_conf_str = f"{attempt.dinco_conf:.3f}" if attempt.dinco_conf is not None else "n/a"
        nvc_str = f"{attempt.nvc:.3f}" if attempt.nvc is not None else "n/a"
        sc_conf_str = f"{attempt.sc_conf:.3f}" if attempt.sc_conf is not None else "n/a"

    closed_book_confidence_block = (
        f"- DINCO final confidence: {dinco_conf_str}\n"
        f"- NVC: {nvc_str}\n"
        f"- SC confidence: {sc_conf_str}"
    )

    return AGENT_TURN_PROMPT.format(
        turn=turn,
        max_turns=max_turns,
        subquestion=subquestion,
        original_question=original_question,
        answer=attempt.answer or "(no answer yet)",
        source=attempt.source or "unknown",
        has_dependency_memory="yes" if has_dependency_memory else "no",
        n_passages=n_passages,
        closed_book_confidence_block=closed_book_confidence_block,
        grounding_block=_format_grounding_block(support, attempt.support_claims, narrative=narrative),
        action_history_block=_format_action_history(history),
        budget_remaining=max(0, max_turns - turn),
    )


def format_agent_delta_prompt(
    *,
    turn: int,
    max_turns: int,
    attempt: planner_utils.AttemptResult,
    support: Optional[Dict[str, Any]],
    prev_action: str,
    prev_attempt: planner_utils.AttemptResult,
    prev_support: Optional[Dict[str, Any]],
    n_passages: int,
    prev_n_passages: int,
    sampling_dinco_enabled: bool = False,
    narrative: bool = False,
) -> str:
    """Build a delta user prompt for turns 2+ in multi-turn mode."""
    # --- Action outcome summary ---
    outcome_parts: list[str] = []
    if prev_action == "retrieve":
        new_passages = n_passages - prev_n_passages
        outcome_parts.append(f" Retrieved {new_passages} new passage(s) ({n_passages} total).")
    elif prev_action == "refine":
        if attempt.answer != prev_attempt.answer:
            outcome_parts.append(f' Answer changed from "{prev_attempt.answer}" to "{attempt.answer}".')
        else:
            outcome_parts.append(" Answer unchanged after refinement.")
    elif prev_action == "commit":
        outcome_parts.append(" Answer committed.")
    action_outcome = "".join(outcome_parts)

    # --- Directional annotations for DINCO ---
    dinco_delta = ""
    if attempt.dinco_conf is not None and prev_attempt.dinco_conf is not None:
        diff = attempt.dinco_conf - prev_attempt.dinco_conf
        if abs(diff) > 0.005:
            direction = "IMPROVED" if diff > 0 else "DECREASED"
            if narrative:
                dinco_delta = f" ({direction} from {_score_to_bin(prev_attempt.dinco_conf)})"
            else:
                dinco_delta = f" ({direction} {diff:+.3f} from {prev_attempt.dinco_conf:.3f})"

    # --- Grounding block with directional annotations ---
    grounding_block = _format_grounding_block(support, attempt.support_claims, narrative=narrative)
    prev_g_mean = float(prev_support["g_mean"]) if prev_support and "g_mean" in prev_support else None
    curr_g_mean = float(support["g_mean"]) if support and "g_mean" in support else None
    if prev_g_mean is not None and curr_g_mean is not None:
        g_diff = curr_g_mean - prev_g_mean
        if abs(g_diff) > 0.005:
            direction = "IMPROVED" if g_diff > 0 else "DECREASED"
            if narrative:
                grounding_block += f"\n- **g_mean trend:** {direction} from {_score_to_bin(prev_g_mean)}"
            else:
                grounding_block += f"\n- **g_mean trend:** {direction} {g_diff:+.3f} from {prev_g_mean:.3f}"

    if narrative:
        dinco_conf_str = _score_to_bin(attempt.dinco_conf)
        nvc_str = _score_to_bin(attempt.nvc)
        sc_conf_str = _score_to_bin(attempt.sc_conf)
    else:
        dinco_conf_str = f"{attempt.dinco_conf:.3f}" if attempt.dinco_conf is not None else "n/a"
        nvc_str = f"{attempt.nvc:.3f}" if attempt.nvc is not None else "n/a"
        sc_conf_str = f"{attempt.sc_conf:.3f}" if attempt.sc_conf is not None else "n/a"

    return AGENT_TURN_DELTA_PROMPT.format(
        turn=turn,
        max_turns=max_turns,
        prev_action=prev_action,
        action_outcome=action_outcome,
        answer=attempt.answer or "(no answer yet)",
        dinco_conf=dinco_conf_str,
        dinco_delta=dinco_delta,
        nvc=nvc_str,
        sc_conf=sc_conf_str,
        grounding_block=grounding_block,
        sampling_dinco_block=_format_sampling_dinco_block(
            attempt, enabled=sampling_dinco_enabled, n_passages=n_passages, narrative=narrative,
        ),
        budget_remaining=max(0, max_turns - turn),
    )


def format_agent_prompt_no_telemetry(
    *,
    turn: int,
    max_turns: int,
    original_question: str,
    subquestion: str,
    attempt: planner_utils.AttemptResult,
    n_passages: int,
    has_dependency_memory: bool,
    history: Sequence[AgentTurnRecord],
) -> str:
    """Build the per-turn user prompt WITHOUT telemetry signals (ablation)."""
    return AGENT_TURN_PROMPT_NO_TELEMETRY.format(
        turn=turn,
        max_turns=max_turns,
        subquestion=subquestion,
        original_question=original_question,
        answer=attempt.answer or "(no answer yet)",
        source=attempt.source or "unknown",
        has_dependency_memory="yes" if has_dependency_memory else "no",
        n_passages=n_passages,
        action_history_block=_format_action_history(history, include_telemetry=False),
        budget_remaining=max(0, max_turns - turn),
    )


def format_agent_delta_prompt_no_telemetry(
    *,
    turn: int,
    max_turns: int,
    attempt: planner_utils.AttemptResult,
    prev_action: str,
    prev_attempt: planner_utils.AttemptResult,
    n_passages: int,
    prev_n_passages: int,
) -> str:
    """Build a delta user prompt WITHOUT telemetry signals (ablation)."""
    # --- Action outcome summary (same logic as full version) ---
    outcome_parts: list[str] = []
    if prev_action == "retrieve":
        new_passages = n_passages - prev_n_passages
        outcome_parts.append(f" Retrieved {new_passages} new passage(s) ({n_passages} total).")
    elif prev_action == "refine":
        if attempt.answer != prev_attempt.answer:
            outcome_parts.append(f' Answer changed from "{prev_attempt.answer}" to "{attempt.answer}".')
        else:
            outcome_parts.append(" Answer unchanged after refinement.")
    elif prev_action == "commit":
        outcome_parts.append(" Answer committed.")
    action_outcome = "".join(outcome_parts)

    return AGENT_TURN_DELTA_PROMPT_NO_TELEMETRY.format(
        turn=turn,
        max_turns=max_turns,
        prev_action=prev_action,
        action_outcome=action_outcome,
        answer=attempt.answer or "(no answer yet)",
        n_passages=n_passages,
        budget_remaining=max(0, max_turns - turn),
    )


# ---------------------------------------------------------------------------
# AgentGatedPlannerRunner
# ---------------------------------------------------------------------------


class AgentGatedPlannerRunner(origbeam.GPT54ParityQwenPlannerRunner):
    """
    Replaces threshold-based routing with an agent loop.

    For each subquestion:
    1. Compute closed-book DINCO (same as origbeam)
    2. Present telemetry to agent
    3. Agent decides: commit, retrieve, refine, or decompose
    4. Execute action, update telemetry, repeat until commit/decompose/budget
    """

    def __init__(
        self,
        *,
        agent_max_turns: int = 6,
        agent_max_new_tokens: int = 512,
        agent_prompt_mode: str = "stateless",
        agent_max_context_tokens: int = 16384,
        agent_telemetry_mode: str = "full",
        use_guided_json: bool = True,
        agent_action_model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_guided_json = bool(use_guided_json)
        # Per-role agent model: when set, _generate_agent_action uses this
        # instead of self.qwen_model. Lets us run cell C ablations where the
        # agent action role is a different LLM (e.g., Search-R1-14B) than
        # the planner+composer (Qwen2.5-14B-Instruct). If None, falls back
        # to qwen_model (cell D / single-model behavior).
        self.agent_action_model = agent_action_model
        self.agent_max_turns = max(1, int(agent_max_turns))
        self.agent_max_new_tokens = max(64, int(agent_max_new_tokens))
        if agent_prompt_mode not in ("stateless", "multi_turn", "react"):
            raise ValueError(f"agent_prompt_mode must be 'stateless', 'multi_turn', or 'react', got {agent_prompt_mode!r}")
        self.agent_prompt_mode = agent_prompt_mode
        self.agent_max_context_tokens = int(agent_max_context_tokens)
        valid_modes = ("full", "info", "role", "narrative", "no_telemetry")
        if agent_telemetry_mode not in valid_modes:
            raise ValueError(f"agent_telemetry_mode must be one of {valid_modes}, got {agent_telemetry_mode!r}")
        self.agent_telemetry_mode = agent_telemetry_mode
        # Sampling-DINCO telemetry flags are inherited from
        # CalibratedPlannerMemoryRunner via super().__init__(**kwargs). Validate
        # that we don't have a meaningless combination.
        if (
            getattr(self, "enable_sampling_dinco_telemetry", False)
            and self.agent_telemetry_mode == "no_telemetry"
        ):
            raise ValueError(
                "enable_sampling_dinco_telemetry=True is meaningless with "
                "agent_telemetry_mode='no_telemetry' — the signal would be "
                "computed but never shown to the agent."
            )

    def _generate_agent_action(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, int]:
        """Call the generator model with a message list for agent action.

        Backend-agnostic: routes through ``qwen_model.generate_chat`` so it
        works on either the HF ``QwenDincoModel`` or the vLLM ``QwenVLLMDincoModel``.

        Parameters
        ----------
        messages : list of dicts
            Full conversation so far. In stateless mode this is
            ``[system, user]``.

        Returns
        -------
        text : str
        input_tokens : int
        """
        schema = HOTPOT_AGENT_JSON_SCHEMA if getattr(self, "use_guided_json", True) else None
        # Route to per-role agent model when configured (cell C ablation).
        # Falls back to main qwen_model for backward compat (cell D / single-model).
        action_model = self.agent_action_model if self.agent_action_model is not None else self.qwen_model
        text, n_input_tokens = action_model.generate_chat(
            list(messages),
            max_new_tokens=self.agent_max_new_tokens,
            enable_thinking=True,
            json_schema=schema,
        )
        return text, int(n_input_tokens)

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

        # Phase 0: Resolve dependencies (unchanged from origbeam)
        execution_subquestion = self._resolve_dependency_subquestion(
            state=state, question=question, node=node
        )
        dependency_entries = self._dependency_memory_entries(state=state, node=node)
        dependency_passages = self._entries_to_passages(dependency_entries)
        has_dependency_memory = bool(dependency_passages)
        state.execution_order.append(node.id)

        # Phase 1: Compute closed-book DINCO (always, same as origbeam)
        pre_attempt: planner_utils.AttemptResult
        forced_retrieval_due_to_dinco_error = False
        dinco_failure_error: Optional[str] = None
        try:
            pre_attempt = self._question_only_attempt(execution_subquestion)
        except Exception as exc:  # noqa: BLE001
            pre_route = "dependency_rewrite_question_only" if has_dependency_memory else "question_only"
            self._emit_printbad(
                question=question,
                node=node,
                execution_subquestion=execution_subquestion,
                dependency_entries=dependency_entries,
                pre_route=pre_route,
                exc=exc,
            )
            # Public release: no remote-DINCO fallback; surface the error.
            raise

        # Phase 2: Agent loop
        current_attempt = pre_attempt
        current_support: Optional[Dict[str, Any]] = None
        all_hits: List[SearchHit] = []
        history: List[AgentTurnRecord] = []
        committed = False
        decomposed = False
        retrieve_count = 0
        refine_count = 0

        self._set_runtime(
            node,
            execution_subquestion=execution_subquestion,
            pre_route="agent_gated",
            route_taken="agent_gated",
            pre_answer=pre_attempt.answer,
            pre_nvc=pre_attempt.nvc,
            pre_sc_conf=pre_attempt.sc_conf,
            pre_dinco_conf=pre_attempt.dinco_conf,
            dependency_memory_count=len(dependency_passages),
            dependency_memory_titles=[p.title for p in dependency_passages],
            dependency_memory_dinco_used=False,
            example_attempt_index=self._current_example_attempt,
            decision_mode="agent",
            agent_max_turns=self.agent_max_turns,
        )

        state.planning_trace.append(
            {
                "event": "agent_loop_start",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "pre_nvc": pre_attempt.nvc,
                "pre_sc_conf": pre_attempt.sc_conf,
                "pre_dinco_conf": pre_attempt.dinco_conf,
                "pre_answer": pre_attempt.answer,
                "max_turns": self.agent_max_turns,
                "has_dependency_memory": has_dependency_memory,
            }
        )

        # Multi-turn message accumulation
        if self.agent_prompt_mode == "react":
            _system_prompt = (REACT_SYSTEM_PROMPT_TELEMETRY
                              if self.agent_telemetry_mode != "no_telemetry"
                              else REACT_SYSTEM_PROMPT)
        else:
            _system_prompt = _get_system_prompt(self.agent_telemetry_mode)
        _is_narrative = self.agent_telemetry_mode == "narrative"
        agent_messages: List[Dict[str, str]] = [
            {"role": "system", "content": _system_prompt},
        ]
        prev_n_passages = 0
        prev_attempt_snapshot_attempt = current_attempt
        prev_support_snapshot: Optional[Dict[str, Any]] = dict(current_support) if current_support else None

        # ===================================================================
        # ReAct loop — Thought / Action / Observation
        # ===================================================================
        if self.agent_prompt_mode == "react":
            _react_telemetry = self.agent_telemetry_mode != "no_telemetry"
            for turn in range(1, self.agent_max_turns + 1):
                # Build user prompt
                if turn == 1:
                    if _react_telemetry:
                        beam_source = current_attempt
                        if not current_attempt.dinco_candidates and pre_attempt and pre_attempt.dinco_candidates:
                            beam_source = pre_attempt
                        user_prompt = REACT_TURN1_PROMPT_TELEMETRY.format(
                            subquestion=execution_subquestion,
                            original_question=question,
                            answer=current_attempt.answer,
                            dinco_conf=f"{current_attempt.dinco_conf:.3f}" if current_attempt.dinco_conf is not None else "n/a",
                            nvc=f"{current_attempt.nvc:.3f}" if current_attempt.nvc is not None else "n/a",
                            sc_conf=f"{current_attempt.sc_conf:.3f}" if current_attempt.sc_conf is not None else "n/a",
                            beam_candidates_block=_format_beam_candidates(beam_source),
                            has_dependency_memory="yes" if has_dependency_memory else "no",
                            budget_remaining=self.agent_max_turns - turn,
                        )
                    else:
                        user_prompt = REACT_TURN1_PROMPT.format(
                            subquestion=execution_subquestion,
                            original_question=question,
                            answer=current_attempt.answer,
                            has_dependency_memory="yes" if has_dependency_memory else "no",
                            budget_remaining=self.agent_max_turns - turn,
                        )
                    agent_messages.append({"role": "user", "content": user_prompt})
                # else: user prompt was already appended after action execution below

                # Get agent response (Thought: ... Action: ...)
                raw_response, input_tokens = self._generate_agent_action(agent_messages)
                agent_messages.append({"role": "assistant", "content": raw_response})

                action = parse_react_action(raw_response)

                # Record this turn
                turn_record = AgentTurnRecord(
                    turn=turn,
                    answer=current_attempt.answer,
                    nvc=current_attempt.nvc,
                    sc_conf=current_attempt.sc_conf,
                    dinco_conf=current_attempt.dinco_conf,
                    g_mean=float(current_support["g_mean"]) if current_support else None,
                    g_min=float(current_support["g_min"]) if current_support else None,
                    claim_supports=list(current_support.get("claim_supports", [])) if current_support else [],
                    claims=list(current_attempt.support_claims),
                    action=action,
                    sampling_dinco_conf=current_attempt.sampling_dinco_conf,
                    sampling_dinco_degenerate=current_attempt.sampling_dinco_degenerate,
                    sampling_dinco_agreement_rate=current_attempt.sampling_dinco_agreement_rate,
                    sampling_dinco_n_unique=current_attempt.sampling_dinco_n_unique,
                )
                history.append(turn_record)

                state.planning_trace.append(
                    {
                        "event": "agent_turn",
                        "node_id": node.id,
                        "turn": turn,
                        "max_turns": self.agent_max_turns,
                        "action": action.type,
                        "action_query": action.query,
                        "action_reason": action.reason,
                        "action_analysis": action.analysis,
                        "agent_prompt_mode": self.agent_prompt_mode,
                        "agent_telemetry_mode": self.agent_telemetry_mode,
                        "agent_input_tokens": input_tokens,
                        "agent_messages_count": len(agent_messages),
                        "telemetry_snapshot": {
                            "answer": current_attempt.answer,
                            "nvc": current_attempt.nvc,
                            "sc_conf": current_attempt.sc_conf,
                            "dinco_conf": current_attempt.dinco_conf,
                            "g_mean": turn_record.g_mean,
                            "g_min": turn_record.g_min,
                            "claim_supports": turn_record.claim_supports,
                            "sampling_dinco_conf": current_attempt.sampling_dinco_conf,
                            "sampling_dinco_degenerate": current_attempt.sampling_dinco_degenerate,
                            "sampling_dinco_agreement_rate": current_attempt.sampling_dinco_agreement_rate,
                            "sampling_dinco_n_unique": current_attempt.sampling_dinco_n_unique,
                            "sampling_distractors": list(current_attempt.sampling_distractors or []),
                            "sampling_ptrues": list(current_attempt.sampling_ptrues or []),
                        },
                    }
                )

                # Execute action
                if action.type == "commit":
                    # ReAct Finish[answer] — use the proposed answer if provided
                    if action.commit_answer:
                        # Override the current attempt answer with what the agent proposed
                        current_attempt.answer = action.commit_answer
                        current_attempt.source = "agent_react_finish"
                    commit_answer_valid = bool(
                        current_attempt.available
                        and current_attempt.answer
                        and hotpot_utils.normalize_answer(current_attempt.answer)
                        != hotpot_utils.normalize_answer("insufficient evidence")
                    )
                    if not commit_answer_valid and turn < self.agent_max_turns:
                        state.planning_trace.append(
                            {
                                "event": "agent_commit_overridden",
                                "node_id": node.id,
                                "turn": turn,
                                "reason": "answer is empty or insufficient evidence",
                                "answer": current_attempt.answer,
                            }
                        )
                        action = AgentAction(
                            type="retrieve",
                            query=execution_subquestion,
                            reason="commit_overridden_invalid_answer",
                            raw_text=action.raw_text,
                        )
                        # Fall through to retrieve logic below
                    else:
                        committed = True
                        break

                if action.type == "retrieve":
                    retrieve_count += 1
                    query = action.query or execution_subquestion
                    prev_n_passages = len(all_hits)
                    new_hits = _hybrid_or_bm25_search(
                        self.index,
                        getattr(self, "dense_bundle", None),
                        self._current_question_id,
                        query,
                        top_k=self.retrieval_top_k,
                        dense_top_k=getattr(self, "dense_top_k", 25),
                        rrf_k=getattr(self, "rrf_k", 60),
                    )
                    # Merge new hits, avoiding duplicates by chunk_id
                    existing_chunk_ids = {
                        str(h.row.get("chunk_id") or "") for h in all_hits
                    }
                    n_new_hits = 0
                    for hit in new_hits:
                        chunk_id = str(hit.row.get("chunk_id") or "")
                        if chunk_id not in existing_chunk_ids:
                            all_hits.append(hit)
                            existing_chunk_ids.add(chunk_id)
                            n_new_hits += 1

                    # Generate answer with evidence + compute grounding
                    scored = self._score_retrieval_attempt(
                        subquestion=execution_subquestion,
                        dependency_passages=dependency_passages,
                        retrieved_passages=[base_runner.hit_to_passage(h) for h in all_hits],
                        fallback_answer=pre_attempt.answer,
                        evidence_chunk_ids=[str(h.row.get("chunk_id") or "") for h in all_hits],
                        evidence_titles=[str(h.row.get("title") or "") for h in all_hits],
                        evidence_scores=[float(h.score) for h in all_hits],
                        source="agent_react_retrieve",
                        compute_sampling_dinco=getattr(self, "enable_sampling_dinco_telemetry", False),
                    )
                    current_attempt = scored["attempt"]
                    current_support = scored["support"]

                    # Build observation + followup as next user message
                    passages_text = _format_passages_for_react(
                        all_hits[prev_n_passages:],  # only show newly retrieved
                        max_passages=5,
                        max_chars=500,
                    )
                    if _react_telemetry:
                        obs_prompt = REACT_OBSERVATION_PROMPT_TELEMETRY.format(
                            n_new=n_new_hits,
                            n_total=len(all_hits),
                            passages_text=passages_text,
                            grounding_block=_format_grounding_block(
                                current_support, current_attempt.support_claims,
                            ),
                        )
                        followup_prompt = REACT_FOLLOWUP_PROMPT_TELEMETRY.format(
                            answer=current_attempt.answer,
                            dinco_conf=f"{current_attempt.dinco_conf:.3f}" if current_attempt.dinco_conf is not None else "n/a",
                            nvc=f"{current_attempt.nvc:.3f}" if current_attempt.nvc is not None else "n/a",
                            sc_conf=f"{current_attempt.sc_conf:.3f}" if current_attempt.sc_conf is not None else "n/a",
                            grounding_block=_format_grounding_block(
                                current_support, current_attempt.support_claims,
                            ),
                            budget_remaining=self.agent_max_turns - turn,
                        )
                    else:
                        obs_prompt = REACT_OBSERVATION_PROMPT.format(
                            n_new=n_new_hits,
                            n_total=len(all_hits),
                            passages_text=passages_text,
                        )
                        followup_prompt = REACT_FOLLOWUP_PROMPT.format(
                            answer=current_attempt.answer,
                            budget_remaining=self.agent_max_turns - turn,
                        )
                    # Combine observation and followup into one user message
                    next_user_msg = obs_prompt + "\n\n" + followup_prompt
                    agent_messages.append({"role": "user", "content": next_user_msg})

                    # Token budget guard
                    est_tokens = sum(len(m["content"]) // 4 for m in agent_messages)
                    if est_tokens + self.agent_max_new_tokens > self.agent_max_context_tokens:
                        state.planning_trace.append({
                            "event": "agent_context_fallback",
                            "node_id": node.id,
                            "turn": turn,
                            "estimated_tokens": est_tokens,
                            "max_context_tokens": self.agent_max_context_tokens,
                        })
                        # Force commit on next iteration
                        committed = True
                        break

            # End of ReAct loop — skip the JSON agent loop below
        else:
            # ===============================================================
            # JSON agent loop (stateless / multi_turn)
            # ===============================================================
            for turn in range(1, self.agent_max_turns + 1):
                # --- Build user prompt ---
                if turn == 1 or self.agent_prompt_mode == "stateless":
                    # Turn 1 (both modes) or every turn (stateless): full state
                    if self.agent_telemetry_mode != "no_telemetry":
                        user_prompt = format_agent_prompt(
                            turn=turn,
                            max_turns=self.agent_max_turns,
                            original_question=question,
                            subquestion=execution_subquestion,
                            attempt=current_attempt,
                            support=current_support,
                            n_passages=len(all_hits),
                            has_dependency_memory=has_dependency_memory,
                            history=history,
                            pre_attempt=pre_attempt,
                            sampling_dinco_enabled=getattr(self, "enable_sampling_dinco_telemetry", False),
                            narrative=_is_narrative,
                        )
                    else:
                        user_prompt = format_agent_prompt_no_telemetry(
                            turn=turn,
                            max_turns=self.agent_max_turns,
                            original_question=question,
                            subquestion=execution_subquestion,
                            attempt=current_attempt,
                            n_passages=len(all_hits),
                            has_dependency_memory=has_dependency_memory,
                            history=history,
                        )
                    if self.agent_prompt_mode == "stateless":
                        # Stateless: reset to [system, user] each turn
                        agent_messages = [
                            {"role": "system", "content": _system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    else:
                        agent_messages.append({"role": "user", "content": user_prompt})
                else:
                    # Multi-turn, turn 2+: delta prompt only
                    prev_action_type = history[-1].action.type if history else "unknown"
                    if self.agent_telemetry_mode != "no_telemetry":
                        user_prompt = format_agent_delta_prompt(
                            turn=turn,
                            max_turns=self.agent_max_turns,
                            attempt=current_attempt,
                            support=current_support,
                            prev_action=prev_action_type,
                            prev_attempt=prev_attempt_snapshot_attempt,
                            prev_support=prev_support_snapshot,
                            n_passages=len(all_hits),
                            prev_n_passages=prev_n_passages,
                            sampling_dinco_enabled=getattr(self, "enable_sampling_dinco_telemetry", False),
                            narrative=_is_narrative,
                        )
                    else:
                        user_prompt = format_agent_delta_prompt_no_telemetry(
                            turn=turn,
                            max_turns=self.agent_max_turns,
                            attempt=current_attempt,
                            prev_action=prev_action_type,
                            prev_attempt=prev_attempt_snapshot_attempt,
                            n_passages=len(all_hits),
                            prev_n_passages=prev_n_passages,
                        )
                    agent_messages.append({"role": "user", "content": user_prompt})

                # Token budget guard: if multi-turn context exceeds limit, fall back
                # to stateless for this turn to avoid OOM or degraded quality.
                if self.agent_prompt_mode == "multi_turn" and turn > 1:
                    est_tokens = sum(len(m["content"]) // 4 for m in agent_messages)
                    if est_tokens + self.agent_max_new_tokens > self.agent_max_context_tokens:
                        state.planning_trace.append({
                            "event": "agent_context_fallback",
                            "node_id": node.id,
                            "turn": turn,
                            "estimated_tokens": est_tokens,
                            "max_context_tokens": self.agent_max_context_tokens,
                        })
                        # Rebuild as stateless single-shot
                        if self.agent_telemetry_mode != "no_telemetry":
                            fallback_prompt = format_agent_prompt(
                                turn=turn,
                                max_turns=self.agent_max_turns,
                                original_question=question,
                                subquestion=execution_subquestion,
                                attempt=current_attempt,
                                support=current_support,
                                n_passages=len(all_hits),
                                has_dependency_memory=has_dependency_memory,
                                history=history,
                                pre_attempt=pre_attempt,
                                sampling_dinco_enabled=getattr(self, "enable_sampling_dinco_telemetry", False),
                                narrative=_is_narrative,
                            )
                        else:
                            fallback_prompt = format_agent_prompt_no_telemetry(
                                turn=turn,
                                max_turns=self.agent_max_turns,
                                original_question=question,
                                subquestion=execution_subquestion,
                                attempt=current_attempt,
                                n_passages=len(all_hits),
                                has_dependency_memory=has_dependency_memory,
                                history=history,
                            )
                        agent_messages = [
                            {"role": "system", "content": _system_prompt},
                            {"role": "user", "content": fallback_prompt},
                        ]

                # Get agent decision
                raw_response, input_tokens = self._generate_agent_action(agent_messages)

                # In multi-turn mode, append assistant response to conversation
                if self.agent_prompt_mode == "multi_turn":
                    agent_messages.append({"role": "assistant", "content": raw_response})

                action = parse_agent_action(raw_response)

                # Snapshot state for delta computation on next turn
                prev_n_passages = len(all_hits)
                prev_attempt_snapshot_attempt = current_attempt
                prev_support_snapshot = dict(current_support) if current_support else None

                # Record this turn
                turn_record = AgentTurnRecord(
                    turn=turn,
                    answer=current_attempt.answer,
                    nvc=current_attempt.nvc,
                    sc_conf=current_attempt.sc_conf,
                    dinco_conf=current_attempt.dinco_conf,
                    g_mean=float(current_support["g_mean"]) if current_support else None,
                    g_min=float(current_support["g_min"]) if current_support else None,
                    claim_supports=list(current_support.get("claim_supports", [])) if current_support else [],
                    claims=list(current_attempt.support_claims),
                    action=action,
                    sampling_dinco_conf=current_attempt.sampling_dinco_conf,
                    sampling_dinco_degenerate=current_attempt.sampling_dinco_degenerate,
                    sampling_dinco_agreement_rate=current_attempt.sampling_dinco_agreement_rate,
                    sampling_dinco_n_unique=current_attempt.sampling_dinco_n_unique,
                )
                history.append(turn_record)

                state.planning_trace.append(
                    {
                        "event": "agent_turn",
                        "node_id": node.id,
                        "turn": turn,
                        "max_turns": self.agent_max_turns,
                        "action": action.type,
                        "action_query": action.query,
                        "action_reason": action.reason,
                        "action_analysis": action.analysis,
                        "agent_prompt_mode": self.agent_prompt_mode,
                        "agent_telemetry_mode": self.agent_telemetry_mode,
                        "agent_input_tokens": input_tokens,
                        "agent_messages_count": len(agent_messages),
                        "telemetry_snapshot": {
                            "answer": current_attempt.answer,
                            "nvc": current_attempt.nvc,
                            "sc_conf": current_attempt.sc_conf,
                            "dinco_conf": current_attempt.dinco_conf,
                            "g_mean": turn_record.g_mean,
                            "g_min": turn_record.g_min,
                            "claim_supports": turn_record.claim_supports,
                            "sampling_dinco_conf": current_attempt.sampling_dinco_conf,
                            "sampling_dinco_degenerate": current_attempt.sampling_dinco_degenerate,
                            "sampling_dinco_agreement_rate": current_attempt.sampling_dinco_agreement_rate,
                            "sampling_dinco_n_unique": current_attempt.sampling_dinco_n_unique,
                            "sampling_distractors": list(current_attempt.sampling_distractors or []),
                            "sampling_ptrues": list(current_attempt.sampling_ptrues or []),
                        },
                    }
                )

                # Execute action
                if action.type == "commit":
                    # Guard: don't commit empty or "insufficient evidence" answers
                    commit_answer_valid = bool(
                        current_attempt.available
                        and current_attempt.answer
                        and hotpot_utils.normalize_answer(current_attempt.answer)
                        != hotpot_utils.normalize_answer("insufficient evidence")
                    )
                    if not commit_answer_valid and turn < self.agent_max_turns:
                        # Override: force retrieve instead of committing garbage
                        state.planning_trace.append(
                            {
                                "event": "agent_commit_overridden",
                                "node_id": node.id,
                                "turn": turn,
                                "reason": "answer is empty or insufficient evidence",
                                "answer": current_attempt.answer,
                            }
                        )
                        action = AgentAction(
                            type="retrieve",
                            query=execution_subquestion,
                            reason="commit_overridden_invalid_answer",
                            raw_text=action.raw_text,
                        )
                        # Fall through to retrieve logic below
                    else:
                        committed = True
                        break

                if action.type == "retrieve":
                    retrieve_count += 1
                    query = action.query or execution_subquestion
                    new_hits = _hybrid_or_bm25_search(
                        self.index,
                        getattr(self, "dense_bundle", None),
                        self._current_question_id,
                        query,
                        top_k=self.retrieval_top_k,
                        dense_top_k=getattr(self, "dense_top_k", 25),
                        rrf_k=getattr(self, "rrf_k", 60),
                    )
                    # Merge new hits, avoiding duplicates by chunk_id
                    existing_chunk_ids = {
                        str(h.row.get("chunk_id") or "") for h in all_hits
                    }
                    for hit in new_hits:
                        chunk_id = str(hit.row.get("chunk_id") or "")
                        if chunk_id not in existing_chunk_ids:
                            all_hits.append(hit)
                            existing_chunk_ids.add(chunk_id)

                    # Generate answer with evidence + compute grounding
                    scored = self._score_retrieval_attempt(
                        subquestion=execution_subquestion,
                        dependency_passages=dependency_passages,
                        retrieved_passages=[base_runner.hit_to_passage(h) for h in all_hits],
                        fallback_answer=pre_attempt.answer,
                        evidence_chunk_ids=[str(h.row.get("chunk_id") or "") for h in all_hits],
                        evidence_titles=[str(h.row.get("title") or "") for h in all_hits],
                        evidence_scores=[float(h.score) for h in all_hits],
                        source="agent_retrieve",
                        compute_sampling_dinco=getattr(self, "enable_sampling_dinco_telemetry", False),
                    )
                    current_attempt = scored["attempt"]
                    current_support = scored["support"]

                elif action.type == "refine":
                    refine_count += 1
                    if not all_hits:
                        # Can't refine without passages; treat as retrieve instead
                        state.planning_trace.append(
                            {
                                "event": "agent_refine_fallback_to_retrieve",
                                "node_id": node.id,
                                "turn": turn,
                                "reason": "no passages available for refinement",
                            }
                        )
                        new_hits = _hybrid_or_bm25_search(
                            self.index,
                            getattr(self, "dense_bundle", None),
                            self._current_question_id,
                            execution_subquestion,
                            top_k=self.retrieval_top_k,
                            dense_top_k=getattr(self, "dense_top_k", 25),
                            rrf_k=getattr(self, "rrf_k", 60),
                        )
                        existing_chunk_ids = {
                            str(h.row.get("chunk_id") or "") for h in all_hits
                        }
                        for hit in new_hits:
                            chunk_id = str(hit.row.get("chunk_id") or "")
                            if chunk_id not in existing_chunk_ids:
                                all_hits.append(hit)
                                existing_chunk_ids.add(chunk_id)

                        scored = self._score_retrieval_attempt(
                            subquestion=execution_subquestion,
                            dependency_passages=dependency_passages,
                            retrieved_passages=[base_runner.hit_to_passage(h) for h in all_hits],
                            fallback_answer=pre_attempt.answer,
                            evidence_chunk_ids=[str(h.row.get("chunk_id") or "") for h in all_hits],
                            evidence_titles=[str(h.row.get("title") or "") for h in all_hits],
                            evidence_scores=[float(h.score) for h in all_hits],
                            source="agent_refine_fallback_retrieve",
                            compute_sampling_dinco=getattr(self, "enable_sampling_dinco_telemetry", False),
                        )
                    else:
                        scored = self._score_retrieval_attempt(
                            subquestion=execution_subquestion,
                            dependency_passages=dependency_passages,
                            retrieved_passages=[base_runner.hit_to_passage(h) for h in all_hits],
                            fallback_answer=pre_attempt.answer,
                            evidence_chunk_ids=[str(h.row.get("chunk_id") or "") for h in all_hits],
                            evidence_titles=[str(h.row.get("title") or "") for h in all_hits],
                            evidence_scores=[float(h.score) for h in all_hits],
                            source="agent_refine",
                            refinement=True,
                            previous_answer=current_attempt.answer,
                            previous_claims=list(current_attempt.support_claims),
                            compute_sampling_dinco=getattr(self, "enable_sampling_dinco_telemetry", False),
                        )
                    current_attempt = scored["attempt"]
                    current_support = scored["support"]

                elif action.type == "decompose":
                    decomposed = True
                    break

        # Phase 3: Finalize
        self._set_runtime(
            node,
            agent_turns_used=len(history),
            agent_action_history=[
                {
                    "turn": r.turn,
                    "action": r.action.type,
                    "query": r.action.query,
                    "reason": r.action.reason,
                    "analysis": r.action.analysis,
                }
                for r in history
            ],
            agent_retrieve_count=retrieve_count,
            agent_refine_count=refine_count,
            agent_committed=committed,
            agent_decomposed=decomposed,
            agent_budget_exhausted=not committed and not decomposed,
            retrieved_chunk_ids_online=[str(h.row.get("chunk_id") or "") for h in all_hits],
            retrieved_titles_online=[str(h.row.get("title") or "") for h in all_hits],
            retrieved_scores_online=[float(h.score) for h in all_hits],
            online_claims=list(current_attempt.support_claims),
            online_g_mean=float(current_support["g_mean"]) if current_support else None,
            online_g_min=float(current_support["g_min"]) if current_support else None,
            online_claim_supports=list(current_support.get("claim_supports", [])) if current_support else [],
            online_supported=(
                base_runner.is_supported(
                    current_support,
                    self.support_mean_threshold,
                    self.support_min_threshold,
                )
                if current_support
                else None
            ),
            grounding_skipped=current_support is None,
            grounding_mode="agent_gated",
        )

        if decomposed:
            state.planning_trace.append(
                {
                    "event": "agent_decompose",
                    "node_id": node.id,
                    "turns_used": len(history),
                }
            )
            return self._decompose_node(
                state=state,
                question=question,
                node=node,
                running_id_counter=running_id_counter,
            )

        if committed:
            # Agent explicitly chose to commit
            retrieved_titles = [str(h.row.get("title") or "") for h in all_hits]
            state.planning_trace.append(
                {
                    "event": "subquestion_scored",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "source": current_attempt.source,
                    "answer": current_attempt.answer,
                    "nvc": current_attempt.nvc,
                    "sc_conf": current_attempt.sc_conf,
                    "dinco_conf": current_attempt.dinco_conf,
                    "g_mean": float(current_support["g_mean"]) if current_support else None,
                    "g_min": float(current_support["g_min"]) if current_support else None,
                    "grounding_mode": "agent_gated",
                    "decision_mode": "agent",
                    "turns_used": len(history),
                    "passed": True,
                }
            )
            self._commit_success(
                node=node,
                attempt=current_attempt,
                retrieved_titles=retrieved_titles,
            )
            self._append_success_entry(state=state, question=question, node=node)
            return running_id_counter

        # Budget exhausted — force commit if grounding looks OK, else decompose
        state.planning_trace.append(
            {
                "event": "agent_budget_exhausted",
                "node_id": node.id,
                "turns_used": len(history),
                "final_answer": current_attempt.answer,
                "final_g_mean": float(current_support["g_mean"]) if current_support else None,
                "final_g_min": float(current_support["g_min"]) if current_support else None,
            }
        )

        grounding_ok = (
            current_support is not None
            and base_runner.is_supported(
                current_support,
                self.support_mean_threshold,
                self.support_min_threshold,
            )
        )
        answer_valid = bool(
            current_attempt.available
            and hotpot_utils.normalize_answer(current_attempt.answer)
            != hotpot_utils.normalize_answer("insufficient evidence")
        )

        if grounding_ok and answer_valid:
            retrieved_titles = [str(h.row.get("title") or "") for h in all_hits]
            state.planning_trace.append(
                {
                    "event": "subquestion_scored",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "source": current_attempt.source,
                    "answer": current_attempt.answer,
                    "grounding_mode": "agent_gated_budget_exhausted_commit",
                    "passed": True,
                }
            )
            self._commit_success(
                node=node,
                attempt=current_attempt,
                retrieved_titles=retrieved_titles,
            )
            self._append_success_entry(state=state, question=question, node=node)
            return running_id_counter

        # Even without grounding, if DINCO is very high, commit anyway
        if (
            answer_valid
            and current_attempt.dinco_conf is not None
            and current_attempt.dinco_conf >= 0.90
        ):
            retrieved_titles = [str(h.row.get("title") or "") for h in all_hits]
            state.planning_trace.append(
                {
                    "event": "subquestion_scored",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "source": current_attempt.source,
                    "answer": current_attempt.answer,
                    "grounding_mode": "agent_gated_budget_exhausted_high_dinco_commit",
                    "passed": True,
                }
            )
            self._commit_success(
                node=node,
                attempt=current_attempt,
                retrieved_titles=retrieved_titles,
            )
            self._append_success_entry(state=state, question=question, node=node)
            return running_id_counter

        return self._decompose_node(
            state=state,
            question=question,
            node=node,
            running_id_counter=running_id_counter,
        )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run agent-gated multihop calibrated retrieval on HotpotQA. "
            "Replaces hardcoded threshold gates with an LLM agent that receives "
            "DINCO/MiniCheck telemetry and decides actions."
        )
    )
    # Dataset args (same as origbeam)
    parser.add_argument("--dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument(
        "--dataset_config", "--dataset_subset",
        dest="dataset_config", type=str, default="distractor",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=2000, help="<= 0 means full split")
    parser.add_argument(
        "--indexed_pool_limit", type=int, default=None,
        help="If set, restrict dataset to first N examples before shuffle/limit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="Skip the first N examples after shuffle (for splitting a run across multiple jobs).",
    )
    parser.add_argument("--example_id", type=str, default=None)
    parser.add_argument(
        "--index_dir", type=str,
        default=str(BASE_DIR / "data" / "hotpotqa_distractor_validation_s0_n2000_chunks_bm25_index"),
    )
    # Retrieval args
    parser.add_argument("--retrieval_top_k", type=int, default=8)
    parser.add_argument("--audit_top_k", type=int, default=8)
    parser.add_argument("--retry_on_low_support", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry_extra_top_k", type=int, default=4)
    # Routing — agent replaces threshold routing, but we keep these for
    # the base class init and budget-exhaustion fallback logic
    parser.add_argument("--gate_on", type=str, default="dinco", choices=["dinco", "nvc"])
    parser.add_argument("--gate_threshold", type=float, default=0.80)
    parser.add_argument("--support_mean_threshold", type=float, default=0.70)
    parser.add_argument("--support_min_threshold", type=float, default=0.50)
    parser.add_argument(
        "--root_subquestion_policy", type=str, default="allow_closed_book_commit",
        choices=["always_retrieve", "allow_closed_book_commit", "skip_without_commit"],
    )
    parser.add_argument(
        "--routing_mode", type=str, default="agent_gated",
        help="Routing mode label for summary. Always 'agent_gated' for this script.",
    )
    # Planner args
    parser.add_argument("--max_initial_subquestions", type=int, default=4)
    parser.add_argument("--max_subquestion_depth", type=int, default=2)
    parser.add_argument("--max_subquestion_nodes", type=int, default=12)
    parser.add_argument("--planner_max_new_tokens", type=int, default=800)
    parser.add_argument("--max_retries", type=int, default=5)
    # DINCO args
    parser.add_argument("--n_distractors", type=int, default=5)
    parser.add_argument("--n_sc_samples", type=int, default=5)
    parser.add_argument("--sc_match_threshold", type=float, default=0.90)
    parser.add_argument("--dinco_nli_model_name", type=str, default=origbeam.DEFAULT_DINCO_NLI_MODEL_NAME)
    parser.add_argument("--dinco_beam_max_new_tokens", type=int, default=100)
    parser.add_argument("--dinco_beam_length_penalty", type=float, default=0.0)
    # Model args
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    # Per-role agent action model (cell C ablation): if set, the agent's per-turn
    # action decision uses this model. Planner, composer, and other roles still
    # use --model_name. If None, agent action also uses --model_name (cell D).
    parser.add_argument("--agent_action_model_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--generator_device_map", type=str, default="auto")
    parser.add_argument(
        "--generator_dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    # MiniCheck args
    parser.add_argument("--minicheck_model_name", type=str, default="Bespoke-MiniCheck-7B")
    parser.add_argument("--allow_minicheck_cpu_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minicheck_cpu_fallback_model_name", type=str, default="roberta-large")
    parser.add_argument("--minicheck_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--minicheck_max_model_len", type=int, default=None)
    parser.add_argument("--minicheck_gpu_memory_gb", type=float, default=25.0)
    parser.add_argument("--minicheck_gpu_memory_utilization", type=float, default=None)
    parser.add_argument(
        "--minicheck_cuda_visible_devices", type=str, default=None,
        help="If set, override CUDA_VISIBLE_DEVICES for MiniCheck's vLLM init only "
             "(restored after). Useful when Qwen TP=N occupies GPUs 0..N-1 and you want "
             "MiniCheck on a separate idle GPU on multi-GPU nodes (e.g. set to '2' on a "
             "3-GPU node when Qwen TP=2 is on GPUs 0,1).",
    )
    parser.add_argument("--noground", action=argparse.BooleanOptionalAction, default=False)
    # Agent-specific args
    parser.add_argument(
        "--agent_max_turns", type=int, default=6,
        help="Maximum agent loop iterations per subquestion.",
    )
    parser.add_argument(
        "--agent_max_new_tokens", type=int, default=512,
        help="Token budget for agent reasoning per turn.",
    )
    parser.add_argument(
        "--agent_prompt_mode", type=str, default="stateless",
        choices=["stateless", "multi_turn", "react"],
        help="Agent prompting strategy: 'stateless' reconstructs prompt each turn, "
             "'multi_turn' accumulates conversation history, "
             "'react' uses Thought/Action/Observation ReAct loop.",
    )
    parser.add_argument(
        "--agent_max_context_tokens", type=int, default=16384,
        help="Max input tokens for agent in multi_turn mode. Falls back to stateless "
             "for a turn if exceeded.",
    )
    parser.add_argument(
        "--agent_telemetry_mode", type=str, default="full",
        choices=["full", "info", "role", "narrative", "no_telemetry"],
        help="Prompt-framing variant. 'full' = baseline gating-language prompt; "
             "'info' = drop threshold language, frame signals as reasoning input; "
             "'role' = info + explicit DINCO-vs-MiniCheck role separation; "
             "'narrative' = info + numeric scores rendered as qualitative bins; "
             "'no_telemetry' = strip all numerical signals (ablation control).",
    )
    parser.add_argument(
        "--use_guided_json", action=argparse.BooleanOptionalAction, default=True,
        help="When the qwen_backend is vLLM, constrain the agent's JSON action "
             "output via vLLM StructuredOutputsParams. Required for Mistral-Small "
             "which otherwise emits ~100% parse-failures under long agent prompts. "
             "Use --no-use_guided_json to disable.",
    )
    parser.add_argument(
        "--qwen_backend", type=str, default="hf", choices=["hf", "vllm"],
        help="Backend for the Qwen3-32B generator. 'hf' = HF transformers (default, "
             "matches existing experiments). 'vllm' = vLLM (5-10x faster; both pre and "
             "post-retrieval DINCO go through the same vLLM engine). With 'vllm', "
             "MiniCheck's gpu_memory_utilization defaults to 0.15 to leave room for Qwen.",
    )
    parser.add_argument(
        "--qwen_gpu_memory_utilization", type=float, default=0.78,
        help="(vLLM only) GPU memory fraction reserved for the Qwen vLLM engine. "
             "Default 0.78 leaves ~0.15 for MiniCheck-vLLM and ~0.07 headroom on a 96 GB your GPU.",
    )
    parser.add_argument(
        "--qwen_max_model_len", type=int, default=4096,
        help="(vLLM only) Max sequence length for the Qwen vLLM engine.",
    )
    parser.add_argument(
        "--qwen_tensor_parallel_size", type=int, default=1,
        help="(vLLM only) Tensor parallel size for the Qwen vLLM engine. Default 1 "
             "(single GPU). Use 2+ when the bf16 weights don't fit a single card "
             "(e.g. Qwen-32B on A100-40GB needs TP=2 across 2 GPUs).",
    )
    parser.add_argument(
        "--enable_sampling_dinco_telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, after retrieval also compute and display sampling-DINCO with "
             "graceful-degenerate (degenerate flag + agreement_rate). The post-retrieval "
             "block is added alongside the existing closed-book DINCO + MiniCheck. Default off "
             "for backwards-compat with prior agent_gated experiments.",
    )
    parser.add_argument(
        "--sampling_dinco_n_samples", type=int, default=10,
        help="Number of stochastic samples per post-retrieval sampling-DINCO call. "
             "Matches the dinco-beam-vs-sampling-hotpotqa canary (n=10). Min 2.",
    )
    # Output args
    parser.add_argument(
        "--question_ids_file", type=str, default=None,
        help="Optional path to a newline-delimited list of HotpotQA question_ids. "
             "When provided, filters loaded examples to exactly these IDs (preserving "
             "the input ordering of the file). Used for paired comparison against a "
             "fixed baseline subset.",
    )
    # --- Hybrid retrieval (BM25 + dense, RRF-fused; from grpo/hybrid_retriever.py) ---
    parser.add_argument(
        "--use_hybrid_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, replace BM25-only retrieval with BM25 + dense RRF-fused via "
             "grpo.hybrid_retriever.hybrid_search. Requires --dense_index_dir.",
    )
    parser.add_argument(
        "--dense_index_dir", type=str, default=None,
        help="Directory with index.faiss + rows.jsonl from grpo/build_dense_index.py. "
             "Required when --use_hybrid_retrieval is on.",
    )
    parser.add_argument(
        "--dense_encoder_model", type=str, default="BAAI/bge-base-en-v1.5",
        help="HF encoder model id matching the dense index build (must match the "
             "model that produced the FAISS vectors).",
    )
    parser.add_argument(
        "--hybrid_dense_top_k", type=int, default=25,
        help="dense_top_k passed to hybrid_search.",
    )
    parser.add_argument(
        "--hybrid_rrf_k", type=int, default=60,
        help="rrf_k damping constant for RRF fusion (standard: 60).",
    )

    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--printbad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.n_distractors < 1:
        raise ValueError("--n_distractors must be >= 1.")
    if args.n_sc_samples < 1:
        raise ValueError("--n_sc_samples must be >= 1.")
    if args.dinco_beam_max_new_tokens < 1:
        raise ValueError("--dinco_beam_max_new_tokens must be >= 1.")
    return args


def default_output_paths(args: argparse.Namespace, limit: Optional[int]) -> Tuple[Path, Path]:
    subset_stem = base_runner.file_stem_for_subset(args.dataset_config, args.split, args.seed, limit)
    mode_tag = f"_{args.agent_prompt_mode}" if args.agent_prompt_mode != "stateless" else ""
    if args.agent_telemetry_mode == "no_telemetry":
        telemetry_tag = "_notelemetry"
    elif args.agent_telemetry_mode in ("info", "role", "narrative"):
        telemetry_tag = f"_{args.agent_telemetry_mode}"
    else:
        telemetry_tag = ""
    sampling_tag = "_samplingdinco" if getattr(args, "enable_sampling_dinco_telemetry", False) else ""
    backend_tag = "_vllm" if getattr(args, "qwen_backend", "hf") == "vllm" else ""
    run_name = (
        f"{subset_stem}_multihop_agent_gated{mode_tag}{telemetry_tag}{sampling_tag}{backend_tag}_"
        f"t{args.agent_max_turns}_"
        f"{slugify_name(args.model_name.split('/')[-1])}"
    )
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else BASE_DIR / "results" / f"{run_name}.jsonl"
    summary_json = (
        Path(args.summary_json) if args.summary_json else BASE_DIR / "results" / f"{run_name}.summary.json"
    )
    return output_jsonl, summary_json


def build_summary(
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_jsonl: Path,
) -> Dict[str, Any]:
    minicheck_gpu_memory_utilization = origbeam.resolve_minicheck_gpu_memory_utilization(args)
    summary = base_runner.build_summary(records=records, args=args, output_jsonl=output_jsonl)
    summary.update(
        {
            "backend": "agent_gated_calibrated_retrieval",
            "decision_mode": "agent",
            "agent_max_turns": args.agent_max_turns,
            "agent_max_new_tokens": args.agent_max_new_tokens,
            "agent_prompt_mode": args.agent_prompt_mode,
            "agent_max_context_tokens": args.agent_max_context_tokens,
            "agent_telemetry_mode": args.agent_telemetry_mode,
            "enable_sampling_dinco_telemetry": bool(getattr(args, "enable_sampling_dinco_telemetry", False)),
            "sampling_dinco_n_samples": int(getattr(args, "sampling_dinco_n_samples", 10)),
            "qwen_backend": str(getattr(args, "qwen_backend", "hf")),
            "qwen_gpu_memory_utilization": float(getattr(args, "qwen_gpu_memory_utilization", 0.78)),
            "model_name": args.model_name,
            "generator_device_map": args.generator_device_map,
            "generator_dtype": args.generator_dtype,
            "planner_max_new_tokens": args.planner_max_new_tokens,
            "max_retries": args.max_retries,
            "n_distractors": args.n_distractors,
            "n_sc_samples": args.n_sc_samples,
            "sc_match_threshold": args.sc_match_threshold,
            "dinco_nli_model_name": args.dinco_nli_model_name,
            "dinco_beam_max_new_tokens": args.dinco_beam_max_new_tokens,
            "dinco_beam_length_penalty": args.dinco_beam_length_penalty,
            "minicheck_model_name": args.minicheck_model_name,
            "allow_minicheck_cpu_fallback": args.allow_minicheck_cpu_fallback,
            "minicheck_cpu_fallback_model_name": args.minicheck_cpu_fallback_model_name,
            "minicheck_tensor_parallel_size": args.minicheck_tensor_parallel_size,
            "minicheck_max_model_len": args.minicheck_max_model_len,
            "minicheck_gpu_memory_gb": args.minicheck_gpu_memory_gb,
            "minicheck_gpu_memory_utilization": minicheck_gpu_memory_utilization,
        }
    )
    return summary


def load_completed_ids(output_jsonl: Path) -> set:
    """Load question IDs already completed for resume support."""
    completed = set()
    if not output_jsonl.exists():
        return completed
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                qid = row.get("id") or row.get("question_id") or ""
                if qid:
                    completed.add(str(qid))
            except json.JSONDecodeError:
                continue
    return completed


def main() -> None:
    args = parse_args()
    hotpot_utils.seed_everything(args.seed)

    limit = normalize_limit(args.limit)
    output_jsonl, summary_json = default_output_paths(args, limit=limit)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    examples = base_runner.load_examples(args, limit=limit)

    # Question-ID filter (paired comparison support).
    if args.question_ids_file:
        with open(args.question_ids_file, "r", encoding="utf-8") as _fp:
            wanted_ids = [ln.strip() for ln in _fp if ln.strip()]
        wanted_set = set(wanted_ids)
        wanted_index = {qid: i for i, qid in enumerate(wanted_ids)}
        filtered = []
        for ex in examples:
            qid = str(ex.get("id") or ex.get("question_id") or "")
            if qid in wanted_set:
                filtered.append(ex)
        # Preserve the order specified in the file
        filtered.sort(key=lambda ex: wanted_index.get(str(ex.get("id") or ex.get("question_id") or ""), 10**9))
        if not filtered:
            raise ValueError(
                f"--question_ids_file {args.question_ids_file!r} matched zero loaded examples. "
                f"Loaded {len(examples)} examples; first few IDs: "
                f"{[ex.get('id') for ex in examples[:3]]}; wanted first few: {wanted_ids[:3]}."
            )
        missing = wanted_set - {str(ex.get("id") or ex.get("question_id") or "") for ex in filtered}
        if missing:
            print(
                f"[question_ids_file] WARNING: {len(missing)} of {len(wanted_set)} requested IDs "
                f"were not present in the loaded examples (e.g. {sorted(missing)[:3]}).",
                flush=True,
            )
        examples = filtered
        print(f"[question_ids_file] Filtered to {len(examples)} paired examples.", flush=True)

    index = BM25Index.load(Path(args.index_dir))

    # Resume support: skip already-completed examples
    completed_ids = load_completed_ids(output_jsonl)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} examples already completed in {output_jsonl}")

    if args.dry_run:
        qwen_model: Any = hotpot_utils.MockQwenModel()
        planner: Any = planner_utils.MockQwenPlannerModel()
        dinco: Any = hotpot_utils.MockDincoCalibrator()
        grounder: Any = hotpot_utils.MockMiniCheckGrounder()
    else:
        minicheck_gpu_memory_utilization = origbeam.resolve_minicheck_gpu_memory_utilization(args)
        if minicheck_gpu_memory_utilization is None:
            # When Qwen runs on vLLM too, both engines share the GPU.
            # Default split: Qwen ~0.78, MiniCheck ~0.15 (leaves ~7 GB for NLI etc).
            # When Qwen runs on HF, MiniCheck has more room: 0.4 default.
            minicheck_gpu_memory_utilization = (
                0.15 if args.qwen_backend == "vllm" else 0.4
            )
        if args.qwen_backend == "vllm":
            print(
                f"[boot] Loading Qwen via vLLM backend (gpu_memory_utilization="
                f"{args.qwen_gpu_memory_utilization}, MiniCheck reserved="
                f"{minicheck_gpu_memory_utilization}).",
                flush=True,
            )
            qwen_model = hotpot_utils.QwenVLLMDincoModel(
                model_name=args.model_name,
                cache_dir=args.cache_dir,
                gpu_memory_utilization=args.qwen_gpu_memory_utilization,
                max_model_len=args.qwen_max_model_len,
                dtype=args.generator_dtype,
                enforce_eager=True,
                tensor_parallel_size=args.qwen_tensor_parallel_size,
            )
        else:
            qwen_model = hotpot_utils.QwenDincoModel(
                model_name=args.model_name,
                cache_dir=args.cache_dir,
                device_map=args.generator_device_map,
                dtype=args.generator_dtype,
            )
        # Per-role agent-action model (cell C ablation). Load a SECOND HF model
        # if --agent_action_model_name is set. Used only for _generate_agent_action;
        # planner, composer, root DINCO still use qwen_model.
        agent_action_model = None
        if args.agent_action_model_name:
            print(
                f"[boot] Loading per-role agent-action model "
                f"{args.agent_action_model_name} (cell C ablation; main model "
                f"{args.model_name} stays for planner+composer+DINCO).",
                flush=True,
            )
            agent_action_model = hotpot_utils.QwenDincoModel(
                model_name=args.agent_action_model_name,
                cache_dir=args.cache_dir,
                device_map=args.generator_device_map,
                dtype=args.generator_dtype,
            )
        planner = planner_utils.QwenPlannerModel(
            qwen_model=qwen_model,
            max_new_tokens=args.planner_max_new_tokens,
            max_retries=args.max_retries,
        )
        # Cell C-full-subq: DINCO (root + subq closed-book answer generation)
        # also routes through agent_action_model when available. Search-R1 was
        # RL-trained on HotpotQA-style answer generation, so the candidate
        # answer + confidence telemetry should benefit from it.
        dinco_backbone = agent_action_model if agent_action_model is not None else qwen_model
        dinco = origbeam.OriginalClosedBookBeamDincoCalibrator(
            qwen_model=dinco_backbone,
            cache_dir=args.cache_dir,
            n_sc_samples=args.n_sc_samples,
            sc_match_threshold=args.sc_match_threshold,
            nli_model_name=args.dinco_nli_model_name,
            beam_max_new_tokens=args.dinco_beam_max_new_tokens,
            beam_length_penalty=args.dinco_beam_length_penalty,
        )
        _saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if args.minicheck_cuda_visible_devices is not None:
            print(
                f"[boot] Pinning MiniCheck to CUDA_VISIBLE_DEVICES="
                f"{args.minicheck_cuda_visible_devices} (was {_saved_cvd!r}).",
                flush=True,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = args.minicheck_cuda_visible_devices
        try:
            grounder = hotpot_utils.MiniCheckGrounder(
                cache_dir=args.cache_dir,
                tensor_parallel_size=args.minicheck_tensor_parallel_size,
                max_model_len=args.minicheck_max_model_len,
                model_name=args.minicheck_model_name,
                allow_cpu_fallback=args.allow_minicheck_cpu_fallback,
                cpu_fallback_model_name=args.minicheck_cpu_fallback_model_name,
                gpu_memory_utilization=minicheck_gpu_memory_utilization,
            )
        finally:
            if args.minicheck_cuda_visible_devices is not None:
                if _saved_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd

    runner = AgentGatedPlannerRunner(
        agent_max_turns=args.agent_max_turns,
        agent_max_new_tokens=args.agent_max_new_tokens,
        agent_prompt_mode=args.agent_prompt_mode,
        agent_max_context_tokens=args.agent_max_context_tokens,
        agent_telemetry_mode=args.agent_telemetry_mode,
        use_guided_json=args.use_guided_json,
        agent_action_model=agent_action_model,
        planner=planner,
        qwen_model=qwen_model,
        subquestion_qwen_model=(agent_action_model if agent_action_model is not None else qwen_model),
        dinco=dinco,
        grounder=grounder,
        index=index,
        gate_on=args.gate_on,
        gate_threshold=args.gate_threshold,
        support_mean_threshold=args.support_mean_threshold,
        support_min_threshold=args.support_min_threshold,
        routing_mode="dinco_gate",  # base class needs this, but agent overrides routing
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
        enable_sampling_dinco_telemetry=args.enable_sampling_dinco_telemetry,
        sampling_dinco_n_samples=args.sampling_dinco_n_samples,
    )

    # Hybrid retrieval (BM25 + dense, RRF-fused). Attaches dense_bundle to runner;
    # _hybrid_or_bm25_search dispatches when getattr(runner, "dense_bundle", None) is set.
    if args.use_hybrid_retrieval:
        if not args.dense_index_dir:
            raise ValueError("--use_hybrid_retrieval requires --dense_index_dir to be set.")
        from telemetry_agent.retrieval.dense_retriever import load_dense_retriever
        dense_bundle = load_dense_retriever(
            model_name=args.dense_encoder_model,
            index_dir=Path(args.dense_index_dir),
            cache_dir=args.cache_dir,
        )
        if dense_bundle is None:
            raise RuntimeError(
                f"Failed to load dense retriever (model={args.dense_encoder_model!r}, "
                f"index_dir={args.dense_index_dir!r}). Check the path + encoder."
            )
        runner.dense_bundle = dense_bundle
        runner.dense_top_k = int(args.hybrid_dense_top_k)
        runner.rrf_k = int(args.hybrid_rrf_k)
        print(
            f"[hybrid] Dense bundle attached: encoder={args.dense_encoder_model}, "
            f"index_dir={args.dense_index_dir}, dense_top_k={args.hybrid_dense_top_k}, "
            f"rrf_k={args.hybrid_rrf_k}.",
            flush=True,
        )

    records: List[Dict[str, Any]] = []
    # Read existing records for summary computation
    if completed_ids and output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    try:
                        records.append(json.loads(text))
                    except json.JSONDecodeError:
                        continue

    with output_jsonl.open("a", encoding="utf-8") as writer:
        for example in examples:
            qid = str(example.get("_id") or example.get("id") or "")
            if qid in completed_ids:
                continue
            row = runner.run_example(example)
            writer.write(json.dumps(row, ensure_ascii=True) + "\n")
            writer.flush()
            records.append(row)

    summary = build_summary(records, args=args, output_jsonl=output_jsonl)

    summary["question_ids_file"] = args.question_ids_file
    summary["use_hybrid_retrieval"] = bool(args.use_hybrid_retrieval)
    summary["dense_index_dir"] = args.dense_index_dir if args.use_hybrid_retrieval else None
    summary["dense_encoder_model"] = args.dense_encoder_model if args.use_hybrid_retrieval else None
    summary["hybrid_dense_top_k"] = int(args.hybrid_dense_top_k) if args.use_hybrid_retrieval else 0
    summary["hybrid_rrf_k"] = int(args.hybrid_rrf_k) if args.use_hybrid_retrieval else 0

    write_json(summary, summary_json)
    print(f"Wrote {len(records)} records to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
