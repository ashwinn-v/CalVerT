"""Minimal reward function for the tiny GRPO canary smoke test.

The reward manager dispatches on ``data['data_source']`` to a scorer
registry; our canary parquet uses `dataset_source` and 'hotpot' which isn't
in the registry. Rather than fight the registry, we provide a custom scorer.

Reward = token-level F1 between extracted answer and ground truth. Crude but
sufficient to validate the GRPO pipeline end-to-end.

Signature matches the trainer's custom_reward_function contract:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)
"""
import re
import string


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p or not g:
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


def _extract_answer(text: str) -> str:
    # Try common answer markers; fall back to last 50 chars.
    text = text or ""
    for pat in [r"\\boxed\{([^}]+)\}", r"answer[:\s]+([^\n]+)", r"final[:\s]+([^\n]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text.strip().split("\n")[-1][-100:]


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = _extract_answer(solution_str)
    return _f1(pred, ground_truth)
