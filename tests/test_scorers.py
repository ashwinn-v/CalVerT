"""Smoke tests for the token-F1 and EM scorers."""
from __future__ import annotations

import pytest

from telemetry_agent.scorers import (
    exact_match_any,
    f1_score_any,
    normalize_answer,
)


def test_normalize_strips_articles_and_punctuation():
    assert normalize_answer("The Great Gatsby.") == "great gatsby"
    assert normalize_answer("A bird in HAND") == "bird in hand"


def test_em_any_exact_match():
    assert exact_match_any("Paris", ["Paris"]) == 1
    assert exact_match_any("paris.", ["Paris"]) == 1


def test_f1_any_exact_match():
    assert f1_score_any("Paris", ["Paris"]) == pytest.approx(1.0)


def test_f1_any_substring_overlap():
    assert 0 < f1_score_any("the Eiffel Tower in Paris", ["Eiffel Tower"]) <= 1.0


def test_f1_any_no_overlap():
    assert f1_score_any("apple", ["orange"]) == pytest.approx(0.0)


def test_f1_any_picks_best_gold():
    # When multiple golds are provided, the scorer reports max F1.
    assert f1_score_any("apple pie", ["banana", "apple"]) > 0
