#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import re
import string
from typing import Sequence


def normalize_answer(text: str) -> str:
    """Match the official TriviaQA normalization rules."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def handle_punc(value: str) -> str:
        exclude = set(string.punctuation + "‘’´`")
        return "".join(ch if ch not in exclude else " " for ch in value)

    def lower(value: str) -> str:
        return value.lower()

    def replace_underscore(value: str) -> str:
        return value.replace("_", " ")

    normalized = lower(replace_underscore(text or ""))
    normalized = handle_punc(normalized)
    normalized = remove_articles(normalized)
    return white_space_fix(normalized).strip()


def exact_match_any(prediction: str, ground_truths: Sequence[str]) -> int:
    normalized_prediction = normalize_answer(prediction)
    return int(any(normalized_prediction == normalize_answer(ground_truth) for ground_truth in ground_truths))


def f1_score_any(prediction: str, ground_truths: Sequence[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens and not ground_truths:
        return 1.0
    if not prediction_tokens:
        return 0.0

    best = 0.0
    prediction_counter = Counter(prediction_tokens)
    for ground_truth in ground_truths:
        ground_truth_tokens = normalize_answer(ground_truth).split()
        if not ground_truth_tokens:
            continue
        overlap = prediction_counter & Counter(ground_truth_tokens)
        num_same = sum(overlap.values())
        if num_same == 0:
            continue
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        best = max(best, (2.0 * precision * recall) / (precision + recall))
    return float(best)
