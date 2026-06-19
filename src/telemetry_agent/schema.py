"""Strict JSON schema for the GRPO env wrapper.

This lives separately from the runner's live `HOTPOT_AGENT_JSON_SCHEMA` (which is
intentionally lax to preserve paper resume-job behavior). The wrapper drives sglang's
guided decoding from this constant, validates emitted actions against it, and treats
violations as `done=True, reward=0`.


"""
from __future__ import annotations

from typing import Any, Dict

HOTPOT_AGENT_JSON_SCHEMA_STRICT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["commit", "retrieve", "refine", "decompose"]},
        "query": {"type": "string", "minLength": 5},
        "analysis": {"type": "string", "minLength": 20},
        "reason": {"type": "string", "minLength": 10},
        "answer": {"type": "string", "minLength": 1},
    },
    "required": ["action", "analysis", "reason"],
    "allOf": [
        {
            "if": {"properties": {"action": {"const": "commit"}}},
            "then": {"required": ["answer"]},
        },
        {
            "if": {"properties": {"action": {"const": "retrieve"}}},
            "then": {"required": ["query"]},
        },
    ],
    "additionalProperties": False,
}


def validate_strict(action_text: str) -> Dict[str, Any]:
    """Parse + validate. Raise on any failure. Caller treats failure as schema-violation."""
    import json

    import jsonschema  # required dependency in pyproject

    parsed = json.loads(action_text)
    jsonschema.validate(parsed, HOTPOT_AGENT_JSON_SCHEMA_STRICT)
    return parsed
