"""Validator-with-context confound test (R-1, W-8 in red_team_brief).

Picks rows from the canary set that match each of 5 control categories,
generates paraphrase variants via deterministic transforms (NEVER LLM-based,
to avoid contaminating the test with bad paraphrases), and a "swapped"
variant from another row.

Expected behavior:
    P(True | greedy)        > P(True | paraphrase) >> P(True | swapped)

If P(True | greedy) ≈ P(True | paraphrase) ≈ 1.0, the validator is acting
as a substring detector. If there's a clear gap, DINCO is doing real
epistemic work.

Output: list of dicts written to canary_controls_results.jsonl, one row per
(canary_example, variant_kind) pair.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple


def is_two_word_proper_noun(answer: str) -> bool:
    parts = answer.strip().split()
    return (len(parts) == 2 and all(p[:1].isupper() and p[1:].islower() for p in parts))


def is_yes_no(answer: str) -> bool:
    return answer.strip().lower() in {"yes", "no"}


def is_year(answer: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", answer.strip()))


def is_single_capitalized_word(answer: str) -> bool:
    parts = answer.strip().split()
    return len(parts) == 1 and parts[0][:1].isupper()


# Category: (filter_fn, paraphrase_fn, name)
CONTROL_CATEGORIES: List[Tuple[Callable[[Dict], bool], Callable[[str], List[str]], str]] = [
    # Two-word proper noun (e.g., person name): swap order, no comma
    (
        lambda ex: ex["type"] == "bridge" and is_two_word_proper_noun(ex["answer"]),
        lambda a: [f"{a.split()[1]} {a.split()[0]}"],  # "Albert Einstein" → "Einstein Albert"
        "person-name-swap",
    ),
    # Yes/No: lengthen
    (
        lambda ex: is_yes_no(ex["answer"]),
        lambda a: ["Yes, that is correct." if a.lower() == "yes" else "No, that is not correct."],
        "yes-no-restate",
    ),
    # Year: prefix
    (
        lambda ex: is_year(ex["answer"]),
        lambda a: [f"the year {a}", f"in {a}"],
        "year-prefix",
    ),
    # Single capitalized word (place/entity): prefix
    (
        lambda ex: ex["type"] == "bridge" and is_single_capitalized_word(ex["answer"]),
        lambda a: [f"the {a}", f"{a} itself"],
        "place-prefix",
    ),
    # Comparison row: rewrite as a sentence
    (
        lambda ex: ex["type"] == "comparison",
        lambda a: [f"the answer is {a}", f"{a} is correct"],
        "comparison-rewrite",
    ),
]


def select_control_rows(rows: List[Dict]) -> List[Tuple[Dict, str, List[str]]]:
    """Pick one matching example per control category. Returns list of
    (example, category_name, paraphrases). Categories with no match are skipped.
    """
    out: List[Tuple[Dict, str, List[str]]] = []
    used_ids = set()
    for filter_fn, paraphrase_fn, name in CONTROL_CATEGORIES:
        for ex in rows:
            if ex["id"] in used_ids:
                continue
            if filter_fn(ex):
                paraphrases = paraphrase_fn(ex["answer"])
                out.append((ex, name, paraphrases))
                used_ids.add(ex["id"])
                break
    return out


def find_swap_answer(ex: Dict, candidates: List[Dict]) -> Optional[str]:
    """Find an answer from another row that's structurally similar but
    different. Naive: any other row's answer that's not equal to this one's.
    """
    for other in candidates:
        if other["id"] == ex["id"]:
            continue
        if other["answer"].strip().lower() == ex["answer"].strip().lower():
            continue
        return other["answer"]
    return None
