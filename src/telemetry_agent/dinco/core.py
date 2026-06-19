"""DINCO core: distractor generation, P(True) extraction, NLI, NVC formula.

This module covers:

- vLLM-backed distractor generation (beam search + stochastic sampling),
- P(True) extraction from top-K logprobs with explicit Yes/No token sets and
  a documented fallback when neither token family appears at the head of the
  distribution,
- Pairwise NLI scoring with DeBERTa,
- The Normalized Verbalized Confidence (NVC) formula from the paper.

The TriviaQA single-turn variant is in :mod:`.triviaqa`; the multi-hop
HotpotQA flow lives in :mod:`.beam_search`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# Lexical cleaning + dedupe
# ---------------------------------------------------------------------------

def lexical_clean_str(s: str) -> str:
    s = s.split('\n')[0]
    s = s.replace('Answer:', '')
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def dedupe_distractors(strs: Sequence[str], scores: Optional[Sequence[float]] = None) -> Tuple[List[str], List[float]]:
    """Lexical-clean + dedupe.

    Keeps the first occurrence of each (lower, no-period)-normalized string.
    Returns parallel lists of cleaned strings and (optional) scores.
    """
    out_strs: List[str] = []
    out_scores: List[float] = []
    seen: Set[str] = set()

    if scores is not None:
        order = sorted(range(len(strs)), key=lambda i: -scores[i])  # high score first
    else:
        order = list(range(len(strs)))

    for i in order:
        s = lexical_clean_str(strs[i])
        if not s:
            continue
        norm = s.lower().replace('.', '')
        if norm in seen:
            continue
        seen.add(norm)
        out_strs.append(s)
        if scores is not None:
            out_scores.append(float(scores[i]))
    return out_strs, out_scores


# ---------------------------------------------------------------------------
# Yes/No token-set construction (VL.7)
# ---------------------------------------------------------------------------

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]


def build_yes_no_sets(tokenizer) -> Tuple[Set[int], Set[int]]:
    """Build {yes_token_ids}, {no_token_ids} for top-K logprob extraction.

    A variant is included only if it tokenizes to exactly one token id under
    the given tokenizer. Asserts that both sets are non-empty (Qwen3 has at
    least one form each; this guards against tokenizer surprises).
    """
    def _ids(variants: List[str]) -> Set[int]:
        out: Set[int] = set()
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
        return out

    yes_set = _ids(YES_VARIANTS)
    no_set = _ids(NO_VARIANTS)
    if not yes_set or not no_set:
        raise RuntimeError(
            f"Yes/No token sets must be non-empty. yes={yes_set}, no={no_set}. "
            f"Tokenizer: {tokenizer.__class__.__name__}"
        )
    return yes_set, no_set


# ---------------------------------------------------------------------------
# P(True) extraction with fallback (VL.7, W-5)
# ---------------------------------------------------------------------------

PTRUE_FALLBACK_NAN = float('nan')
PTRUE_TOTAL_PROB_THRESHOLD = 0.5  # if sum(P[yes_set]+P[no_set]) < this, fallback


def extract_ptrue(top_logprobs: Dict[int, float], yes_set: Set[int], no_set: Set[int]) -> float:
    """Extract P(True) from top-K logprobs at the first generated token.

    `top_logprobs` is {token_id: logprob}. Returns float in [0, 1] or NaN if
    the validator failed (neither Yes nor No families appear with sufficient
    mass in the top-K). Callers should treat NaN as a failure signal and drop
    the row from ALL conditions.
    """
    p_yes = 0.0
    p_no = 0.0
    for tid, lp in top_logprobs.items():
        if tid in yes_set:
            p_yes += math.exp(lp)
        elif tid in no_set:
            p_no += math.exp(lp)
    total = p_yes + p_no
    if total < PTRUE_TOTAL_PROB_THRESHOLD:
        return PTRUE_FALLBACK_NAN
    return p_yes / total


# ---------------------------------------------------------------------------
# vLLM call wrappers
# ---------------------------------------------------------------------------

@dataclass
class GenResult:
    text: str                     # cleaned text
    raw_text: str                 # uncleaned (for finish_reason audit)
    finish_reason: str
    cum_logprob: Optional[float] = None


def vllm_greedy(llm, prompt: str, max_tokens: int) -> GenResult:
    """Single greedy generation. Used for the per-row main answer (computed once)."""
    from vllm import SamplingParams
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)[0]
    seq = out.outputs[0]
    return GenResult(
        text=lexical_clean_str(seq.text),
        raw_text=seq.text,
        finish_reason=seq.finish_reason,
        cum_logprob=getattr(seq, 'cumulative_logprob', None),
    )


def vllm_sample_distractors(llm, prompt: str, n: int, temperature: float, top_p: float,
                             max_tokens: int, seed: int) -> List[GenResult]:
    """Sampling-based distractors via vLLM SamplingParams(n=n)."""
    from vllm import SamplingParams
    out = llm.generate(
        [prompt],
        SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,
        ),
        use_tqdm=False,
    )[0]
    return [
        GenResult(
            text=lexical_clean_str(seq.text),
            raw_text=seq.text,
            finish_reason=seq.finish_reason,
            cum_logprob=getattr(seq, 'cumulative_logprob', None),
        )
        for seq in out.outputs
    ]


def vllm_beam_distractors(llm, prompt: str, beam_width: int, max_tokens: int,
                           length_penalty: float = 0.0) -> List[GenResult]:
    """Beam-search distractors via vLLM LLM.beam_search().

    NOTE: vLLM beam_search API has shifted across releases. This call targets
    vLLM 0.16+. If the import or signature fails on the target cluster,
    canary boot will surface the error before the main loop runs.
    """
    from vllm.sampling_params import BeamSearchParams
    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=max_tokens,
        ignore_eos=False,
        temperature=0.0,
        length_penalty=length_penalty,
    )
    outs = llm.beam_search([prompt], params=params)
    out = outs[0]
    results: List[GenResult] = []
    # vLLM BeamSearchOutput.sequences[i].text returns the FULL sequence text
    # (prompt + completion). Strip the prompt prefix to isolate the new tokens.
    # cum_logprob attribute name varies across vLLM releases.
    for seq in out.sequences:
        full_text = seq.text or ""
        if full_text.startswith(prompt):
            new_text = full_text[len(prompt):]
        else:
            # Tokenization round-trip can shift whitespace; fall back to last-line
            # heuristic: take everything after the last "<|im_start|>assistant" marker.
            marker = "<|im_start|>assistant"
            if marker in full_text:
                new_text = full_text.rsplit(marker, 1)[-1].lstrip("\n")
            else:
                new_text = full_text
        cum = None
        for attr in ("cum_logprob", "cumulative_logprob"):
            if hasattr(seq, attr):
                v = getattr(seq, attr)
                if v is not None:
                    cum = float(v)
                    break
        results.append(GenResult(
            text=lexical_clean_str(new_text),
            raw_text=new_text,
            finish_reason='stop',  # beam search doesn't expose per-beam finish_reason
            cum_logprob=cum,
        ))
    return results


def vllm_validator_ptrue(llm, prompts: Sequence[str], yes_set: Set[int], no_set: Set[int],
                          logprobs_k: int = 20) -> List[float]:
    """Batch validator P(True) via top-K logprobs at the first output token.

    Returns parallel list of P(True) floats (NaN on fallback failure).
    """
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=logprobs_k)
    outs = llm.generate(list(prompts), sp, use_tqdm=False)
    results: List[float] = []
    for out in outs:
        seq = out.outputs[0]
        # seq.logprobs: List[Dict[int, Logprob]] — first index is the first generated token
        if not seq.logprobs:
            results.append(PTRUE_FALLBACK_NAN)
            continue
        first_step = seq.logprobs[0]
        # Logprob objects expose .logprob (newer vLLM). Older versions use raw float.
        top: Dict[int, float] = {}
        for tid, lp in first_step.items():
            top[tid] = lp.logprob if hasattr(lp, 'logprob') else float(lp)
        results.append(extract_ptrue(top, yes_set, no_set))
    return results


# ---------------------------------------------------------------------------
# NLI (DeBERTa) for NVC formula
# ---------------------------------------------------------------------------

ENTAIL_IDX = 0
NEUTRAL_IDX = 1
CONTRA_IDX = 2

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def load_nli(device: str = 'cuda', dtype=torch.float16):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=dtype).to(device)
    model.eval()
    return tok, model


def compute_pairwise_nli(nli_tok, nli_model, question: str, candidates: List[str], batch_size: int = 32) -> torch.Tensor:
    """Pairwise NLI matrix over candidates. Returns [N, N, 3] where [i, j] is
    NLI(premise=Q+candidates[i], hypothesis=Q+candidates[j]). Diagonal is 0.
    """
    n = len(candidates)
    device = next(nli_model.parameters()).device
    out = torch.zeros(n, n, 3)
    if n < 2:
        return out

    premises: List[str] = []
    hypotheses: List[str] = []
    pair_ij: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            premises.append(f"Question: {question}\nAnswer: {candidates[i]}")
            hypotheses.append(f"Answer: {candidates[j]}")
            pair_ij.append((i, j))

    for start in range(0, len(premises), batch_size):
        end = start + batch_size
        ins = nli_tok(premises[start:end], hypotheses[start:end], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = nli_model(**ins).logits
            probs = torch.softmax(logits, dim=-1).cpu()
        for k, (i, j) in enumerate(pair_ij[start:end]):
            out[i, j] = probs[k]
    return out


def compute_nvc(ptrues: torch.Tensor, nli: torch.Tensor) -> float:
    """Normalized Verbalized Confidence (DINCO NVC) for a single example.

    Args:
      ptrues: [N] tensor of P(True) per candidate. Index 0 is the main answer.
      nli:    [N, N, 3] pairwise NLI probabilities (entail, neutral, contra).

    Implements the formula from `dinco_triviaqa.py:get_normalized_verbalized_confidence`.
    """
    main_i = 0
    sym_nli = (nli + nli.swapdims(0, 1)) / 2
    contra_w = sym_nli[..., CONTRA_IDX]
    sims = nli[..., ENTAIL_IDX]
    degrees = torch.sum(torch.maximum(torch.tensor(0.0), sims), dim=0) + 1.0

    numerator = ptrues[main_i].clone()
    denominator = numerator.clone()
    for ans_i in range(ptrues.shape[0]):
        if ans_i == main_i:
            continue
        if torch.isnan(ptrues[ans_i]):
            continue
        denom_term = (
            ptrues[ans_i] * contra_w[main_i, ans_i]
            / (degrees[ans_i] - sims[main_i, ans_i] + 1e-9)
        )
        denominator = denominator + denom_term
    if denominator > 1.0:
        return float(numerator / denominator)
    return float(numerator)
