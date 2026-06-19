"""Helpers for the role-beam agent rollout, used by the Tinker trainer.

The public release ships only the pure helpers that the Tinker SDK trainer
needs:

* :func:`build_user_telemetry_message` — render one turn's telemetry block.
* :func:`parse_json_action` — extract the strict-JSON action emitted by the
  agent (robust to ``<think>`` blocks and code fences).
* :func:`token_f1` — token-level F1 with the stereotype guard from the paper.
* :func:`run_rollout` — execute one trajectory using a caller-supplied policy
  callable. The Tinker trainer calls this through its ``--runtime_module``
  flag.

If you train via a different harness, the helpers are framework-agnostic and
easy to wire into any rollout loop.
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_user_telemetry_message(
    turn: int,
    max_turns: int,
    subquestion: str,
    candidate_answer: str,
    dinco_conf: float,
    nvc: float,
    sc_conf: float,
    g_mean: Optional[float],
    g_min: Optional[float],
    claim_supports: Optional[List[float]],
    budget_remaining: int,
) -> str:
    """Render one turn's telemetry block. Mirrors ``grpo/build_sft_corpus.py``.

    The agent reads the result as the ``user`` content; the strict-JSON action
    is its assistant emission.
    """
    g_mean_str = f"{g_mean:.3f}" if g_mean is not None else "n/a"
    g_min_str = f"{g_min:.3f}" if g_min is not None else "n/a"
    claim_str = (
        "[" + ", ".join(f"{x:.3f}" for x in claim_supports) + "]"
        if claim_supports
        else "[]"
    )
    return (
        f"TURN {turn}/{max_turns} — subquestion: {subquestion}\n\n"
        f"Current best answer: {candidate_answer}\n\n"
        f"Telemetry signals (role-mode):\n"
        f"  dinco_conf : {dinco_conf:.3f}\n"
        f"  nvc        : {nvc:.3f}\n"
        f"  sc_conf    : {sc_conf:.3f}\n"
        f"  g_mean     : {g_mean_str}\n"
        f"  g_min      : {g_min_str}\n"
        f"  claim_supports : {claim_str}\n\n"
        f"Budget: {budget_remaining} turn(s) remain. Choose one strict-JSON action."
    )


# ---------------------------------------------------------------------------
# JSON action extraction (handles Qwen3 <think> blocks and code fences)
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Catches unclosed <think> when the model hits max_tokens mid-think; without
# this guard, the entire response body looks like it sits inside a think tag
# and the parser sees garbage.
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)


def parse_json_action(text: str) -> Dict[str, Any]:
    """Extract the JSON action from one assistant turn.

    Strips Qwen3 ``<think>...</think>`` blocks first so their curly braces do
    not corrupt the greedy ``{.*}`` match. The action JSON always comes after
    the thinking block in Qwen3-style outputs.
    """
    text = (text or "").strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    text = _UNCLOSED_THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {"action": "commit", "answer": "", "_parse_error": "no_json"}
    try:
        return json.loads(m.group(0))
    except Exception as exc:  # noqa: BLE001
        return {"action": "commit", "answer": "", "_parse_error": str(exc)}


# ---------------------------------------------------------------------------
# Token-F1 reward (mirrors grpo.reward._token_f1)
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 with the length-floor stereotype guard."""
    p = _normalize_text(pred).split()
    g = _normalize_text(gold).split()
    if not p or not g:
        return 0.0
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


# ---------------------------------------------------------------------------
# Rollout integration point for the Tinker trainer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Rollout:
    """One full agent-loop trajectory on one prompt.

    Mirrors the dataclass in :mod:`grpo.tinker_train`. The trainer calls
    :func:`run_rollout` to obtain one of these, scores it via the reward
    module, and feeds ``(rollout, advantage)`` tuples into Tinker's
    ``forward_backward``.
    """

    prompt_id: str
    assistant_token_ids: List[int]
    assistant_text: str
    gold_answer: str
    n_lm_calls: int
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


PolicySampleFn = Callable[[Any, List[Dict[str, str]], int], Dict[str, Any]]
"""Caller-supplied policy sampler used inside :func:`run_rollout`.

Signature: ``(policy_handle, messages, seed) -> {"text": str, "token_ids": list[int]}``.

The default driver below assumes the Tinker SDK's ``sample`` API but is
deliberately not hard-wired so callers can plug in any chat-style policy.
"""


def _default_policy_sample(policy_handle: Any, messages: List[Dict[str, str]], seed: int) -> Dict[str, Any]:
    """Default sampler: dispatch to ``policy_handle.sample(messages, seed=seed)``.

    Tinker training clients expose a ``sample`` method that takes the same
    chat-format messages used at inference. Replace this function (or pass
    ``policy_sample_fn`` to :func:`run_rollout`) when your harness uses a
    different surface. Returns ``{"text": str, "token_ids": list[int]}``.
    """
    out = policy_handle.sample(messages=messages, seed=seed)
    if isinstance(out, dict):
        text = out.get("text", "")
        token_ids = out.get("token_ids", [])
    else:
        text = getattr(out, "text", "")
        token_ids = getattr(out, "token_ids", []) or []
    return {"text": str(text or ""), "token_ids": list(token_ids)}


def run_rollout(
    row: Dict[str, Any],
    policy_handle: Any,
    seed: int,
    *,
    policy_sample_fn: Optional[PolicySampleFn] = None,
    system_prompt: Optional[str] = None,
    max_turns: int = 8,
) -> Rollout:
    """Execute one trajectory and return the resulting :class:`Rollout`.

    Lightweight reference implementation: builds an initial chat with the
    paper's role-beam system prompt + the row's question, calls the policy
    once per turn, and treats the first ``commit``/``decompose`` emission as
    the trajectory end. This is the integration seam Tinker calls; replace
    the body with a full multi-turn agent loop (calling the runner's
    retriever / DINCO / MiniCheck modules) when you stand up a production
    trainer.
    """
    sampler = policy_sample_fn or _default_policy_sample

    if system_prompt is None:
        # Lazy import so this module can be used in unit tests without the
        # heavy runner stack on PYTHONPATH.
        from telemetry_agent.runners.hotpotqa_role_beam import AGENT_SYSTEM_PROMPT  # type: ignore

        system_prompt = AGENT_SYSTEM_PROMPT

    question = str(row.get("question") or "")
    gold = str(row.get("gold_answer") or "")
    prompt_id = str(row.get("question_id") or row.get("_id") or "unknown")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    n_calls = 0
    assistant_token_ids: List[int] = []
    assistant_text = ""

    for _ in range(max_turns):
        sample = sampler(policy_handle, messages, seed + n_calls)
        n_calls += 1
        assistant_text = sample["text"]
        assistant_token_ids.extend(sample.get("token_ids") or [])
        action = parse_json_action(assistant_text)
        action_type = str(action.get("action", "")).strip().lower()
        if action_type in {"commit", "decompose"} or action.get("_parse_error"):
            break
        # For non-terminal actions the stub does not invoke the retriever /
        # MiniCheck pipeline; production trainers should replace this loop
        # with a real environment step. Keeping the loop bounded ensures the
        # rollout always terminates.
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {
                "role": "user",
                "content": "Environment step not executed in this rollout stub.",
            }
        )

    return Rollout(
        prompt_id=prompt_id,
        assistant_token_ids=assistant_token_ids,
        assistant_text=assistant_text,
        gold_answer=gold,
        n_lm_calls=n_calls,
    )
