#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

import torch


PROMPT = """
Here are 2 sets of example prompt and answer.

Example Prompt: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
Example Answer: Sinclair Lewis

Example Prompt: Where in England was Dame Judi Dench born?
Example Answer: York

---

Now, here is a new prompt to answer. Answer with a concise phrase, as in the examples.

Prompt: {question}
Answer:
""".strip()

PTRUE_PROMPT = """
Below is a question and a candidate answer. Your task is to determine whether the answer is correct or not. Only output "Yes" (correct) or "No" (incorrect).

Question: {question}
Candidate answer: {candidate_answer}
""".strip()

FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"
END_MARKER = "<|end|>"
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
YES_TOKEN_OPTIONS = ("Yes", " yes")
NO_TOKEN_OPTIONS = ("No", " no")


def supports_model(model_name: str) -> bool:
    return str(model_name or "").strip().lower().startswith("openai/gpt-oss")


def clean_candidate_text(text: str) -> str:
    out = str(text or "").split("\n")[0]
    out = out.replace("Answer:", "")
    out = out.strip()
    out = re.sub(r"\s+", " ", out)
    return out


def canonical_binary_label(text: str) -> str | None:
    cleaned = str(text or "").strip().lower().strip(".,:;!?\"'`()[]{}")
    if cleaned in {"yes", "y", "true", "1"}:
        return "yes"
    if cleaned in {"no", "n", "false", "0"}:
        return "no"
    return None


def extract_final_channel_text(raw_text: str) -> str:
    text = str(raw_text or "")
    if FINAL_CHANNEL_MARKER in text:
        text = text.split(FINAL_CHANNEL_MARKER)[-1]
    elif "<|message|>" in text:
        text = text.split("<|message|>")[-1]
    if END_MARKER in text:
        text = text.split(END_MARKER, 1)[0]
    text = SPECIAL_TOKEN_RE.sub("", text)
    return text.strip()


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _pad_token_id(tokenizer: Any) -> int | None:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is not None:
        return int(token_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return None


def _attention_mask_from_input_ids(input_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    pad_token_id = _pad_token_id(tokenizer)
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return input_ids.ne(pad_token_id).long()


def apply_chat_template_batch(
    tokenizer: Any,
    conversations: Sequence[Sequence[Mapping[str, str]]],
    *,
    reasoning_effort: str = "low",
) -> Dict[str, torch.Tensor]:
    kwargs = {
        "add_generation_prompt": True,
        "padding": True,
        "return_tensors": "pt",
    }
    attempts = (
        {"reasoning_effort": reasoning_effort, "return_dict": True},
        {"return_dict": True},
        {"reasoning_effort": reasoning_effort},
        {},
    )

    last_exc: Exception | None = None
    rendered: Any = None
    for extra_kwargs in attempts:
        try:
            rendered = tokenizer.apply_chat_template(conversations, **kwargs, **extra_kwargs)
            break
        except TypeError as exc:
            last_exc = exc
    if rendered is None:
        assert last_exc is not None
        raise last_exc

    if isinstance(rendered, torch.Tensor):
        input_ids = rendered
        return {
            "input_ids": input_ids,
            "attention_mask": _attention_mask_from_input_ids(input_ids, tokenizer),
        }

    input_ids = rendered["input_ids"]
    attention_mask = rendered.get("attention_mask")
    if attention_mask is None:
        attention_mask = _attention_mask_from_input_ids(input_ids, tokenizer)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _move_inputs_to_model(inputs: Mapping[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
    device = _model_device(model)
    return {key: value.to(device) for key, value in inputs.items()}


def _resolve_single_token_ids(
    tokenizer: Any,
    *,
    cache_attr: str,
    options: Sequence[str],
) -> List[int]:
    cached = getattr(tokenizer, cache_attr, None)
    if cached is not None:
        return list(cached)

    resolved: List[int] = []
    seen: set[int] = set()
    for option in options:
        token_ids = tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        if token_id not in seen:
            seen.add(token_id)
            resolved.append(token_id)
    setattr(tokenizer, cache_attr, tuple(resolved))
    return resolved


def _resolve_token_sequence(
    tokenizer: Any,
    *,
    cache_attr: str,
    text: str,
) -> List[int]:
    cached = getattr(tokenizer, cache_attr, None)
    if cached is not None:
        return list(cached)
    token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    setattr(tokenizer, cache_attr, tuple(token_ids))
    return token_ids


def _final_marker_token_ids(tokenizer: Any) -> List[int]:
    return _resolve_token_sequence(
        tokenizer,
        cache_attr="_gpt_oss_final_marker_token_ids",
        text=FINAL_CHANNEL_MARKER,
    )


def _binary_token_ids(tokenizer: Any) -> tuple[List[int], List[int]]:
    yes_ids = _resolve_single_token_ids(
        tokenizer,
        cache_attr="_gpt_oss_yes_token_ids",
        options=YES_TOKEN_OPTIONS,
    )
    no_ids = _resolve_single_token_ids(
        tokenizer,
        cache_attr="_gpt_oss_no_token_ids",
        options=NO_TOKEN_OPTIONS,
    )
    return yes_ids, no_ids


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    last_start = len(sequence) - len(subsequence)
    for start in range(last_start + 1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return -1


def _decode_generated_suffixes(
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_length: int,
) -> List[str]:
    decoded: List[str] = []
    for row in sequences[:, prompt_length:]:
        raw_text = tokenizer.decode(row, skip_special_tokens=False)
        final_text = extract_final_channel_text(raw_text)
        decoded.append(final_text or clean_candidate_text(raw_text))
    return decoded


def _generate_kwargs(tokenizer: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "use_cache": True,
    }
    pad_token_id = _pad_token_id(tokenizer)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def _yes_probability_from_scores(
    tokenizer: Any,
    *,
    raw_text: str,
    generated_token_ids: Sequence[int],
    scores: Sequence[torch.Tensor],
) -> float:
    yes_ids, no_ids = _binary_token_ids(tokenizer)
    marker_ids = _final_marker_token_ids(tokenizer)
    marker_start = _find_subsequence(generated_token_ids, marker_ids)
    if marker_start >= 0:
        answer_index = marker_start + len(marker_ids)
        if answer_index < len(scores) and yes_ids and no_ids:
            step_scores = scores[answer_index]
            yes_logits = step_scores[yes_ids]
            no_logits = step_scores[no_ids]
            combined = torch.cat((yes_logits, no_logits), dim=0)
            yes_log_mass = torch.logsumexp(yes_logits, dim=0)
            total_log_mass = torch.logsumexp(combined, dim=0)
            return float(torch.exp(yes_log_mass - total_log_mass).item())

    parsed_answer = clean_candidate_text(extract_final_channel_text(raw_text))
    label = canonical_binary_label(parsed_answer)
    if label == "yes":
        return 1.0
    if label == "no":
        return 0.0
    raise RuntimeError(f"Unable to recover binary Yes/No probability from response: {raw_text!r}")


def beam_search(
    ds: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    num_beams: int = 5,
    length_penalty: float = 0.0,
    max_new_tokens: int = 100,
) -> tuple[List[List[str]], torch.Tensor]:
    beam_strs: List[List[str]] = []
    beam_lls = torch.zeros((len(ds), num_beams), dtype=torch.float32)

    for ex_i, ex in enumerate(ds):
        messages = [[{"role": "user", "content": PROMPT.format(question=ex["question"])}]]
        model_inputs = _move_inputs_to_model(apply_chat_template_batch(tokenizer, messages), model)
        prompt_length = int(model_inputs["input_ids"].shape[1])

        outputs = model.generate(
            **model_inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            output_scores=True,
            return_dict_in_generate=True,
            **_generate_kwargs(tokenizer),
        )

        beam_strs.append(_decode_generated_suffixes(tokenizer, outputs.sequences, prompt_length))
        beam_lls[ex_i] = outputs.sequences_scores.detach().cpu().to(torch.float32)

    return beam_strs, beam_lls


def get_ptrue(
    ds: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    beam_strs: Sequence[Sequence[str]],
    *,
    max_new_tokens: int = 64,
) -> torch.Tensor:
    max_width = max((len(strs) for strs in beam_strs), default=0)
    ptrues = -torch.ones((len(ds), max_width), dtype=torch.float32)
    if max_width == 0:
        return ptrues

    for ex_i, ex in enumerate(ds):
        answers = list(beam_strs[ex_i])
        if not answers:
            continue

        messages = [
            [{"role": "user", "content": PTRUE_PROMPT.format(question=ex["question"], candidate_answer=answer)}]
            for answer in answers
        ]
        model_inputs = _move_inputs_to_model(apply_chat_template_batch(tokenizer, messages), model)
        prompt_length = int(model_inputs["input_ids"].shape[1])

        outputs = model.generate(
            **model_inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            **_generate_kwargs(tokenizer),
        )

        for ans_i in range(len(answers)):
            generated_row = outputs.sequences[ans_i, prompt_length:]
            raw_text = tokenizer.decode(generated_row, skip_special_tokens=False)
            step_scores = [score_step[ans_i] for score_step in outputs.scores]
            ptrues[ex_i, ans_i] = _yes_probability_from_scores(
                tokenizer,
                raw_text=raw_text,
                generated_token_ids=[int(token_id) for token_id in generated_row.tolist()],
                scores=step_scores,
            )

    return ptrues


def sample_generations(
    ds: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    n_sample: int = 5,
    max_new_tokens: int = 100,
) -> List[List[str]]:
    sampled_strs: List[List[str]] = []

    for ex in ds:
        messages = [[{"role": "user", "content": PROMPT.format(question=ex["question"])}]]
        model_inputs = _move_inputs_to_model(apply_chat_template_batch(tokenizer, messages), model)
        prompt_length = int(model_inputs["input_ids"].shape[1])
        repeated_inputs = {
            key: value.repeat(n_sample, 1)
            for key, value in model_inputs.items()
        }

        outputs = model.generate(
            **repeated_inputs,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            **_generate_kwargs(tokenizer),
        )
        sampled_strs.append(_decode_generated_suffixes(tokenizer, outputs.sequences, prompt_length))

    return sampled_strs
