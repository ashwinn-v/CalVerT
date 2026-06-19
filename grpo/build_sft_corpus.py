"""Extract the SFT cold-start corpus from released paper trajectories.

Source: Qwen3-32B role-condition rollouts on HotpotQA + 2Wiki (paper data).
Filter: top quartile by per-question Pareto dominance (high F1, low LM-call cost).
Output: a parquet of (messages, target) pairs, one per agent_turn event in the
        kept trajectories.

Each row in the corpus:
  messages: list[dict] — FULL chat including the assistant turn whose content
            is the strict-schema JSON action. TRL's multi-turn SFT dataset masks
            non-assistant tokens automatically, so the loss is computed only on
            the action JSON.
  target:   str        — duplicate of the assistant content for convenience
                         when inspecting; not consumed by the trainer.
  question_id, gold_answer, dataset_source, turn, question_total_lm_calls,
  question_f1 — provenance.

See ``grpo/README.md`` for the cold-start corpus rules.

CLI::

    python -m grpo.build_sft_corpus \\
        --hf_hotpot ${HF_HOTPOT_TRAJECTORIES} \\
        --local_2wiki data/raw_jsonl/2wiki300_full_vllm/role_s42.jsonl \\
        --top_quartile_only \\
        --output data/sft_corpus_v0.parquet


"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------


def _iter_hf_rows(repo_id: str) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="train")
    for row in ds:
        yield dict(row)


def _iter_local_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Per-question filtering — Pareto top quartile
# ---------------------------------------------------------------------------


def _count_lm_calls(row: Dict[str, Any]) -> int:
    return sum(
        1 for ev in (row.get("policy_trace") or [])
        if ev.get("action") == "agent_turn"
    )


def _pareto_top_quartile(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep questions in the top quartile of (high F1, low LM calls).

    Rank twice: by F1 desc, by LM calls asc. A row is "top quartile" if the
    sum of its rank percentiles is in the top quartile of the joint score.
    """
    if not rows:
        return []
    f1_sorted = sorted(rows, key=lambda r: -float(r.get("f1") or 0.0))
    cost_sorted = sorted(rows, key=lambda r: _count_lm_calls(r))
    n = len(rows)
    rank_f1 = {id(r): i for i, r in enumerate(f1_sorted)}
    rank_cost = {id(r): i for i, r in enumerate(cost_sorted)}
    scored = [
        (r, rank_f1[id(r)] + rank_cost[id(r)])
        for r in rows
    ]
    scored.sort(key=lambda x: x[1])
    cutoff = max(1, n // 4)
    return [r for r, _ in scored[:cutoff]]


# ---------------------------------------------------------------------------
# Per-turn extraction
# ---------------------------------------------------------------------------


def _build_target_json(ev_details: Dict[str, Any], final_answer: str) -> Dict[str, Any]:
    """Reconstruct the strict-schema action JSON the agent emitted at this turn.

    On `commit` we use `final_answer` (the row's `pred_answer_raw`) since
    that's the answer the agent ultimately committed to. On all other actions,
    the `answer` field is omitted (the strict schema doesn't require it for
    non-commit actions).
    """
    action = ev_details["action"]
    out: Dict[str, Any] = {
        "action": action,
        "analysis": (ev_details.get("action_analysis") or "").strip(),
        "reason": (ev_details.get("action_reason") or "").strip(),
    }
    if action == "retrieve":
        out["query"] = (ev_details.get("action_query") or "").strip()
    elif action == "commit":
        # answerless-commit reward-hacking bug (canary 706761): SFT commit targets must include answer.
        out["answer"] = (final_answer or "").strip()
    return out


_USER_TEMPLATE = """\
TURN {turn}/{max_turns} — subquestion: {question}

Current best answer: {answer}

Telemetry signals (role-mode):
  dinco_conf : {dinco_conf}
  nvc        : {nvc}
  sc_conf    : {sc_conf}
  g_mean     : {g_mean}
  g_min      : {g_min}
  claim_supports : {claim_supports}

Budget: {budget_remaining} turn(s) remain. Choose one strict-JSON action."""


def _build_user_message(ev_details: Dict[str, Any], question: str) -> str:
    """Compact role-mode telemetry message for SFT.

    Mirrors the runner's `format_agent_prompt` shape but as a simplified
    deterministic template. Exact-byte parity with the runner's prompted
    format is a v2 polish (future work); for SFT cold-start, the
    *signals* + format consistency are what teach the JSON schema and the
    "use the numbers in your analysis" behavior.
    """
    snap = ev_details.get("telemetry_snapshot") or {}

    def _fmt(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    return _USER_TEMPLATE.format(
        turn=ev_details.get("turn", "?"),
        max_turns=ev_details.get("max_turns", 8),
        question=question,
        answer=snap.get("answer") or "(no answer yet)",
        dinco_conf=_fmt(snap.get("dinco_conf")),
        nvc=_fmt(snap.get("nvc")),
        sc_conf=_fmt(snap.get("sc_conf")),
        g_mean=_fmt(snap.get("g_mean")),
        g_min=_fmt(snap.get("g_min")),
        claim_supports=list(snap.get("claim_supports") or []),
        budget_remaining=max(
            0, int(ev_details.get("max_turns", 8)) - int(ev_details.get("turn", 0))
        ),
    )


def _extract_pairs(
    row: Dict[str, Any],
    dataset_source: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """One row of the released JSONL → list of (messages, target) SFT pairs."""
    pairs: List[Dict[str, Any]] = []
    final_answer = (row.get("pred_answer_raw") or row.get("pred_answer") or "").strip()
    question = (row.get("question") or "").strip()
    qid = str(row.get("question_id") or row.get("id") or "")
    f1 = float(row.get("f1") or 0.0)
    total_lm = _count_lm_calls(row)

    for ev in row.get("policy_trace") or []:
        if ev.get("action") != "agent_turn":
            continue
        details = ev.get("details") or {}
        # Per-turn user message based on telemetry at the time of THIS turn.
        user_msg = _build_user_message(details, question)
        target = _build_target_json(details, final_answer)
        # Field-coverage gate per-pair: skip if analysis/reason too short.
        if len(target.get("analysis", "")) < 20:
            continue
        if len(target.get("reason", "")) < 10:
            continue
        target_json_str = json.dumps(target, ensure_ascii=False)
        pairs.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": target_json_str},
                ],
                "target": target_json_str,
                "question_id": qid,
                "gold_answer": str(row.get("gold_answer") or ""),
                "dataset_source": dataset_source,
                "turn": int(details.get("turn", -1)),
                "question_total_lm_calls": total_lm,
                "question_f1": f1,
                "action_type": target["action"],
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_hotpot", default=None,
                   help="HF dataset repo id for HotpotQA Qwen role trajectories.")
    p.add_argument("--local_2wiki", default=None,
                   help="Local JSONL path for 2Wiki Qwen role trajectories.")
    p.add_argument("--local_hotpot", default=None,
                   help="Alt: local JSONL path for HotpotQA (if not on HF).")
    p.add_argument(
        "--top_quartile_only", action="store_true",
        help="Apply per-dataset Pareto top-quartile filter .",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--max_pairs", type=int, default=None,
        help="Cap total pairs (for debug). None = keep all that pass filters.",
    )
    args = p.parse_args()

    # System prompt: import the live one so SFT/GRPO/eval stay in sync.
    from telemetry_agent.runners.hotpotqa_role_beam import AGENT_SYSTEM_PROMPT

    pairs: List[Dict[str, Any]] = []
    by_source: Dict[str, int] = defaultdict(int)

    # ---- HotpotQA ----
    if args.hf_hotpot:
        print(f"[sft] loading hotpot from HF: {args.hf_hotpot}", flush=True)
        rows = list(_iter_hf_rows(args.hf_hotpot))
        print(f"[sft]   {len(rows)} hotpot rows", flush=True)
        if args.top_quartile_only:
            rows = _pareto_top_quartile(rows)
            print(f"[sft]   pareto top quartile → {len(rows)} hotpot rows", flush=True)
        for r in rows:
            ps = _extract_pairs(r, "hotpot", AGENT_SYSTEM_PROMPT)
            pairs.extend(ps)
            by_source["hotpot"] += len(ps)
    elif args.local_hotpot:
        print(f"[sft] loading hotpot from local: {args.local_hotpot}", flush=True)
        rows = list(_iter_local_jsonl(Path(args.local_hotpot)))
        if args.top_quartile_only:
            rows = _pareto_top_quartile(rows)
            print(f"[sft]   pareto top quartile → {len(rows)} hotpot rows", flush=True)
        for r in rows:
            ps = _extract_pairs(r, "hotpot", AGENT_SYSTEM_PROMPT)
            pairs.extend(ps)
            by_source["hotpot"] += len(ps)

    # ---- 2Wiki ----
    if args.local_2wiki:
        print(f"[sft] loading 2wiki from local: {args.local_2wiki}", flush=True)
        rows = list(_iter_local_jsonl(Path(args.local_2wiki)))
        print(f"[sft]   {len(rows)} 2wiki rows", flush=True)
        if args.top_quartile_only:
            rows = _pareto_top_quartile(rows)
            print(f"[sft]   pareto top quartile → {len(rows)} 2wiki rows", flush=True)
        for r in rows:
            ps = _extract_pairs(r, "2wiki", AGENT_SYSTEM_PROMPT)
            pairs.extend(ps)
            by_source["2wiki"] += len(ps)

    if not pairs:
        print("[sft] no pairs produced — check input flags.", file=sys.stderr)
        return 2

    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]
        print(f"[sft] capped to --max_pairs={args.max_pairs}", flush=True)

    # ---- Field-coverage gate ----
    n = len(pairs)
    n_with_full_fields = sum(
        1 for p_ in pairs
        if all(json.loads(p_["target"]).get(k) for k in ("analysis", "reason"))
    )
    coverage = n_with_full_fields / n
    print(f"\n[sft] {n} total pairs, field-coverage = {coverage*100:.1f}%")
    print(f"[sft] by source: {dict(by_source)}")
    action_dist: Dict[str, int] = defaultdict(int)
    for pp in pairs:
        action_dist[pp["action_type"]] += 1
    print(f"[sft] action distribution: {dict(action_dist)}")
    if coverage < 0.85:
        print(
            "[sft] FAIL field-coverage gate < 85% — supplement corpus before SFT.",
            file=sys.stderr,
        )
        return 1

    # ---- Write parquet ----
    import pandas as pd
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_parquet(args.output, index=False)
    print(f"\n[sft] wrote {n} pairs → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
