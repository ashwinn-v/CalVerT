"""M3 production reward: F1 + LM-call-budget penalty.

Implements the reward:

    r = ALPHA * F1(predicted, gold) - BETA * min(total_lm_calls / BUDGET, 1.0)

where:
  - ALPHA = 2.0  (F1 weight, locked in §6b)
  - BETA  = 0.15 (LM-call-cost weight)
  - BUDGET = 9   (measured p95 over 600 trajectories, 2026-05-10)

Public signature::
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

``extra_info`` carries the per-trajectory ``total_lm_calls`` counted by the
agent loop. If extra_info is None or missing the key, we treat the
trajectory as "used full budget" (worst case) so reward = ALPHA*F1 - BETA.

Final-answer extraction: agent commits with action="commit" + answer="...";
The trainer serializes the tool calls into ``solution_str``. We search for the last commit
answer, then the last "answer": "..." JSON field, then "Final answer:" /
"Answer:" text. If none of those are present, the answer is empty.
"""
from __future__ import annotations

import json
import re
from typing import Any

ALPHA = 2.0
BETA = 0.15
BUDGET = 9


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p or not g:
        return 0.0
    # Length-floor stereotype guard:
    # Reward leakage rewarded stereotype answers like "no" / "Speed" / "Jerry Jones"
    # because partial-token F1 gives positive credit on every gold containing those
    # tokens, even when the agent ignored the question. Zero F1 when the prediction
    # is a 1-token span AND the gold is >=3 tokens (the stereotype-hacking regime).
    # Yes/no questions (gold=1 token) and 2-token golds are unaffected.
    if len(p) < 2 and len(g) >= 3:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    n = sum(min(p.count(t), g.count(t)) for t in common)
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


_ANSWER_JSON_RE = re.compile(r'"answer"\s*:\s*"([^"]+)"', re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_COMMIT_BLOCK_RE = re.compile(r'"action"\s*:\s*"commit".*?"answer"\s*:\s*"([^"]+)"', re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Catches unclosed <think> from max_tokens truncation; must mirror the same
# pattern in role_beam_agent_loop._parse_json_action.
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)


def _extract_final_answer(solution_str: str) -> str:
    """Extract the committed answer from a multi-turn rollout transcript."""
    text = solution_str or ""
    # Strip Qwen3 reasoning blocks first so example-actions inside <think> can't
    # corrupt commit/answer extraction.
    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    # Prefer the answer field of the LAST `commit` action block (multi-turn
    # traces can in theory contain multiple commit-shaped blobs; the agent's
    # final action is what we score).
    commit_matches = _COMMIT_BLOCK_RE.findall(text)
    if commit_matches:
        return commit_matches[-1].strip()
    # Otherwise, take the last `"answer": "..."` JSON field in the trace.
    matches = _ANSWER_JSON_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: regex on "Final answer:" / "Answer:" keyword.
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # answerless-commit reward-hacking bug (canary 706761): no noisy text fallback.
    return ""


def _extract_total_lm_calls(extra_info: Any) -> int:
    """Pull the total_lm_calls counter from extra_info; treat missing as budget-max."""
    if extra_info is None:
        return BUDGET
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            return BUDGET
    if not isinstance(extra_info, dict):
        return BUDGET
    # The trainer wraps the agent-loop metrics into extra_info; the per-turn dict is at
    # extra_info["tool_metrics"] in some versions, or flat in others. Probe both.
    if "total_lm_calls" in extra_info:
        return int(extra_info["total_lm_calls"])
    tool_metrics = extra_info.get("tool_metrics") or extra_info.get("agent_loop_metrics") or {}
    if isinstance(tool_metrics, dict) and "total_lm_calls" in tool_metrics:
        return int(tool_metrics["total_lm_calls"])
    # As a final fallback, num_turns ≈ total_lm_calls (each turn invokes one tool call).
    if "num_turns" in extra_info:
        return int(extra_info["num_turns"])
    return BUDGET


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Reward function. Returns a float in roughly [-BETA, ALPHA+BETA].

    Examples:
      perfect F1, 1 LM call:  2.0 * 1.0 - 0.15 * (1/9) ≈ 1.983
      F1 = 0.5,    5 LM calls: 2.0 * 0.5 - 0.15 * (5/9) ≈ 0.917
      F1 = 0,      9 LM calls: 0 - 0.15 = -0.15
    """
    pred = _extract_final_answer(solution_str)
    f1 = _token_f1(pred, ground_truth)
    lm_cost = min(_extract_total_lm_calls(extra_info) / max(BUDGET, 1), 1.0)
    return float(ALPHA * f1 - BETA * lm_cost)
