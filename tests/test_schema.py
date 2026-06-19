"""Strict-JSON action schema sanity checks."""
from __future__ import annotations

import json

import pytest

from telemetry_agent.schema import (
    HOTPOT_AGENT_JSON_SCHEMA_STRICT,
    validate_strict,
)


def test_schema_has_all_four_actions():
    actions = HOTPOT_AGENT_JSON_SCHEMA_STRICT["properties"]["action"]["enum"]
    assert set(actions) == {"commit", "retrieve", "refine", "decompose"}


def test_commit_with_answer_is_valid():
    obj = {
        "action": "commit",
        "answer": "Paris",
        "analysis": "The closed-book confidence is high and the answer is well-grounded.",
        "reason": "high confidence answer",
    }
    assert validate_strict(json.dumps(obj))["action"] == "commit"


def test_retrieve_with_query_is_valid():
    obj = {
        "action": "retrieve",
        "query": "Eiffel Tower height",
        "analysis": "Grounding scores are low so more evidence is required before committing.",
        "reason": "evidence pool is thin",
    }
    assert validate_strict(json.dumps(obj))["action"] == "retrieve"


def test_commit_without_answer_is_rejected():
    import jsonschema

    obj = {
        "action": "commit",
        "analysis": "Confidence is high and the grounding pool looks supportive enough.",
        "reason": "no need to retrieve",
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_strict(json.dumps(obj))
