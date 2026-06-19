"""Single source of truth for chat-template construction.

Every vLLM call site in this experiment (greedy answer, sampling distractors,
beam-search distractors, validator P(True)) uses this helper. Pinning
`enable_thinking=False` here avoids drift across call sites.
"""

from typing import List, Dict


def build_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Format a chat message list as a prompt string for vLLM.

    Always passes `enable_thinking=False` for Qwen3 — short-answer regime,
    no reasoning trace.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
