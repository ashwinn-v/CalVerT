#!/usr/bin/env python3
"""
Iterative multi-hop retrieval controller using:
- DINCO normalized verbal confidence (NVC) from Qwen-8B
- MiniCheck grounding scores from Bespoke-MiniCheck-7B

Target dataset:
- hotpotqa/hotpot_qa
- subset: distractor
- split: validation

Key policy:
1) Retrieve one passage at a time.
2) Generate answer + decontextualized support claims from current evidence.
3) Compute NVC on the answer.
4) Only if NVC > threshold, ground claims with MiniCheck and aggregate grounding g.
5) If NVC > threshold but g < threshold, run an evidence-only refinement pass.
6) Keep retrieving until both confidence and grounding priors are sufficient, or budget ends.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Passage:
    index: int
    title: str
    text: str


@dataclass
class GenerationOutput:
    answer: str
    support_claims: List[str]
    answer_support_claims: List[str]
    raw_text: str
    structured: Dict[str, Any]


@dataclass
class DincoResult:
    nvc: float
    candidates: List[str]
    ptrues: List[float]
    nli: List[List[List[float]]]
    sc_conf: float = 0.0
    final_conf: float = 0.0
    sampled_generations: List[str] = field(default_factory=list)
    sc_entailments: List[float] = field(default_factory=list)


@dataclass
class SamplingDincoResult:
    """Post-retrieval sampling-based DINCO with graceful-degenerate.

    When samples collapse onto the greedy answer (n_unique_distractors == 0),
    `sampling_dinco_conf` falls back to `raw_verbal_ptrue` and `degenerate=True`
    is set so downstream consumers can distinguish "DINCO normalized real
    distractors" from "DINCO had nothing to normalize."
    """
    sampling_dinco_conf: float        # equals raw_verbal_ptrue when degenerate
    degenerate: bool                  # True iff 0 unique distractors after dedupe
    agreement_rate: float             # fraction of N samples matching greedy after norm
    n_unique_distractors: int
    candidates: List[str]             # cleaned + deduped, greedy at index 0
    raw_samples: List[str]            # pre-dedupe (for trace logging)
    ptrues: List[float]               # parallel to candidates
    raw_verbal_ptrue: float           # P(True) on greedy alone
    nli: List[List[List[float]]]      # [N,N,3] entail/neutral/contra; [] if degenerate


@dataclass
class CandidateState:
    answer: str
    support_claims: List[str]
    grounding_claims: List[str]
    claim_grounding_scores: List[float]
    selected_indices: List[int]
    selected_titles: List[str]
    nvc: float
    g: float
    combined_prior: float
    hop: int
    refined: bool


@dataclass
class PolicyDecision:
    action: str
    reason: str
    raw_text: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    guard_triggers: List[str] = field(default_factory=list)


ANSWER_AND_CLAIMS_PROMPT = """
You are solving a multi-hop HotpotQA question using only the provided evidence paragraphs.

Question:
{question}

Evidence paragraphs:
{evidence}

Return STRICT JSON only with this schema:
{{
  "answer": "short Hotpot-style answer phrase",
  "support_claims": [
    {{
      "id": "c1",
      "claim": "atomic, decontextualized sentence with explicit entities and no pronouns",
      "hop": 1,
      "depends_on": [],
      "evidence_titles": ["exact title from evidence"]
    }}
  ]
}}

Rules:
- Answer style must match HotpotQA gold answers: short span/entity/date/number/yes/no.
- Prefer 1-5 words for the answer when possible.
- Do not output full-sentence explanations as the answer.
- Decontextualize every support_claim so each claim stands alone.
- support_claims is the grounding target. The list must contain the hop-wise evidence facts you actually used to reach the answer.
- Order support_claims by hop. Earlier hops must appear before later hops that depend on them.
- Claims must form a reasoning chain for multi-hop inference.
- Use short atomic claims (one fact each).
- Use only facts present in the evidence.
- Each support_claim must express exactly one fact. Do not merge multiple facts into one claim.
- If the answer requires multiple hops, output one support_claim per hop you used.
- Split claims whenever a sentence would otherwise contain multiple linked entities or multiple relations.
- Each support_claim must be explicit and disambiguated enough for verification. Include the named entity, event, title, role, year, or comparison anchor needed to identify the subject of that one fact.
- Do not add side facts, dates, locations, appositives, or relative clauses unless they are strictly necessary to identify the subject of that one fact.
- Do not use conjunctions like "and", "but", or "while" to combine facts inside a support_claim.
- If a later hop depends on an intermediate entity, write that intermediate fact as its own earlier support_claim.
- If the answer is unsupported or is an abstention/meta answer such as "unknown" or "not mentioned", return an empty support_claims list.
- Preferred pattern for support_claims:
  1. Name the subject explicitly rather than using a generic placeholder when the evidence identifies it.
  2. Assert only one relation or slot per claim.
  3. Keep only the minimum disambiguating anchor needed for verification.
- Avoid these anti-patterns:
  1. generic slot-only claims that drop the subject or event anchor
  2. multi-fact claims that combine identification, description, and answer in one sentence
  3. claims that collapse multiple hops into one sentence
  4. claims that omit the comparison, title, season, event, or role needed to distinguish the correct subject
- Do not include extra keys or markdown.
""".strip()


REFINEMENT_PROMPT = """
Your previous answer had insufficient grounding. Re-answer using ONLY the evidence below.
Do not use outside knowledge.

Some of the previous support claims were weakly grounded. Repair them so that each support claim is independently verifiable by a claim checker.

Question:
{question}

Evidence paragraphs:
{evidence}

Previous answer:
{previous_answer}

Previous support claims:
{previous_claims}

Return STRICT JSON only with this schema:
{{
  "answer": "concise answer phrase or 'insufficient evidence'",
  "support_claims": [
    {{
      "id": "c1",
      "claim": "atomic, decontextualized sentence with explicit entities and no pronouns",
      "hop": 1,
      "depends_on": [],
      "evidence_titles": ["exact title from evidence"]
    }}
  ]
}}

Rules:
- Keep only claims directly supported by evidence.
- Claims must still preserve multi-hop reasoning compatibility.
- Prefer fewer high-precision claims over many weak claims.
- Keep answer short in HotpotQA style (span/entity/date/number/yes/no).
- support_claims is the grounding target. Output the hop-wise evidence facts you actually used to derive the answer.
- Order support_claims by hop and keep the dependency order explicit through the list.
- Each support_claim must express exactly one fact.
- If the answer requires multiple hops, output one support_claim per hop you actually used.
- Make each support_claim explicit and disambiguated enough for verification. Keep the precise subject/relation anchors needed to identify that one fact.
- Rewrite or drop any previous claim that was vague, bundled, or under-specified.
- Every support_claim should be able to stand alone for verifier grounding without relying on another claim to supply its subject.
- If a previous claim mixed multiple entities, multiple roles, or multiple relations, split it into separate hop claims.
- If a later hop depends on an intermediate entity, keep that intermediate fact as its own earlier claim instead of collapsing the chain.
- Prefer a verifier-safe claim over a shorter but ambiguous claim.
- Do not bundle descriptors, multiple relations, or multiple hops into one support_claim.
- If the answer is unsupported or abstention-style, return an empty support_claims list.
- Preferred pattern for support_claims:
  1. explicitly name the subject identified by the evidence
  2. assert only one relation or slot
  3. preserve only the disambiguating anchor needed for verifier grounding
- Avoid these anti-patterns:
  1. vague slot-only claims that drop the subject or event anchor
  2. bundled claims that include multiple facts, multiple entities, or multiple descriptors
  3. claims that remove the season, title, comparison, event, or role needed to verify the answer
  4. claims that skip an intermediate hop and jump straight to a final answer bundle
- Do not include extra keys or markdown.
""".strip()


DINCO_BEAM_PROMPT = """
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


DINCO_BEAM_WITH_EVIDENCE_PROMPT = """
Here are 2 sets of example prompt and answer.

Example Prompt: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
Example Answer: Sinclair Lewis

Example Prompt: Where in England was Dame Judi Dench born?
Example Answer: York

---

Now, here is a new question with evidence paragraphs.
Answer with a concise phrase grounded in the evidence.

Question: {question}
Evidence paragraphs:
{evidence}

Answer:
""".strip()


PTRUE_PROMPT = """
Question: {question}
Candidate answer: {candidate_answer}

Reply with exactly one word: Yes or No.
Answer:
""".strip()


PTRUE_WITH_EVIDENCE_PROMPT = """
Question: {question}
Evidence:
{evidence}
Candidate answer: {candidate_answer}

Based only on the evidence, reply with exactly one word: Yes or No.
Answer:
""".strip()


DISTRACTOR_WITH_EVIDENCE_PROMPT = """
You are generating alternative candidate answers for confidence calibration.
Use the question and evidence to propose plausible but different short answers.

Question:
{question}

Evidence paragraphs:
{evidence}

Main candidate answer:
{main_candidate}

Other candidate hints (may include weak/redundant options):
{other_candidates}

Return STRICT JSON only:
{{
  "distractors": [
    "short answer phrase"
  ]
}}

Rules:
- Generate exactly {n_distractors} distractors when possible.
- Each distractor must be a concise answer phrase (not a sentence).
- Distractors must differ from each other and from the main candidate.
- Keep distractors grounded in the evidence when possible.
- No explanations, no markdown, no extra keys.
""".strip()


POLICY_DECISION_PROMPT = """
You are a retrieval policy controller for multi-hop HotpotQA.
You must choose exactly one action:
- "commit": stop retrieving and accept the current answer.
- "retrieve": retrieve the next passage.

You must output STRICT JSON only:
{{
  "action": "commit" | "retrieve",
  "reason": "brief reason"
}}

========================
Inputs (State JSON)
========================
You will be given a JSON object named state_json with at least:
- question: string
- current_answer: string
- hop: integer (0-indexed)
- max_hops: integer
- nvc: float in [0,1]
- grounding_g: float in [0,1] or null
- claim_grounding_scores: array of objects:
  [
    {{
      "claim": string,
      "support_prob": float in [0,1],
      "is_answer_critical": boolean
    }}, ...
  ]
- thresholds: object:
  {{
    "nvc_low": float,
    "nvc_high": float,
    "grounding_low": float,
    "support_low": float,
    "support_high": float,
    "min_critical_supported": integer
  }}

Optional fields:
- predicted_answer_type: one of [PERSON, LOCATION, DATE, NUMBER, ORG, TITLE_ROLE, YESNO, OTHER]
- required_entities: array of strings
- evidence_summary: string

========================
Step 1: Determine expected answer type
========================
Infer expected answer type from question if predicted_answer_type is missing.
Heuristics:
- "Who" -> PERSON (or ORG depending on wording)
- "Where" -> LOCATION
- "When" / "What year" -> DATE
- "How many" / "How much" -> NUMBER
- "What position/role/title" -> TITLE_ROLE
- Yes/no phrasing -> YESNO

If current_answer is incompatible with expected type,
action MUST be "retrieve" unless hop == max_hops.

========================
Step 2: Hard rules (must-follow)
========================
1) If nvc < thresholds.nvc_low -> MUST "retrieve" (unless hop == max_hops, then "commit").
2) If grounding_g is null or grounding_g < thresholds.grounding_low -> MUST "retrieve" (unless hop == max_hops, then "commit").
3) Define critical_supported as number of claims where:
   - is_answer_critical == true AND support_prob >= thresholds.support_high
   If critical_supported < thresholds.min_critical_supported -> MUST "retrieve" (unless hop == max_hops).
4) If required_entities is provided:
   - If any required entity is absent from claims with support_prob >= thresholds.support_low,
     then MUST "retrieve" (unless hop == max_hops).
5) If claim_grounding_scores is empty or missing answer-critical claims -> MUST "retrieve" (unless hop == max_hops).

========================
Step 3: Multi-hop readiness checks (strong preference to retrieve)
========================
Prefer "retrieve" when any holds (unless hop == max_hops):
- Claims are related but do not directly entail current_answer.
- Intermediate facts are present but bridge to final answer is missing.
- Multiple plausible answers remain unresolved.
- nvc is moderate/high but only non-critical claims are supported.
- Answer is compound and only partly supported.

========================
Step 4: Commit criteria
========================
Choose "commit" only if ALL are true:
- nvc >= thresholds.nvc_high
- grounding_g is not null and grounding_g >= thresholds.grounding_low
- expected answer type matches current_answer
- All answer-critical claims have support_prob >= thresholds.support_high
- required_entities coverage passes (if provided)
- Claims directly entail current_answer (not merely related)

If unsure, choose "retrieve" (unless hop == max_hops).

Return STRICT JSON only with keys: action, reason.

State JSON:
{state_json}
""".strip()

RESULTS_DIR_NAME = "results"


def resolve_json_output_path(path_str: str) -> Path:
    path = Path(path_str)
    package_root = Path(__file__).resolve().parent
    return package_root / RESULTS_DIR_NAME / path.name


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def parse_generation_output(text: str) -> GenerationOutput:
    parsed = extract_json_dict(text)
    if parsed is not None:
        answer = str(parsed.get("answer", "")).strip()
        claims_raw = parsed.get("support_claims", [])
        answer_support_claims_raw = parsed.get("answer_support_claims", [])
        claims: List[str] = []
        answer_support_claims: List[str] = []
        if isinstance(claims_raw, list):
            for item in claims_raw:
                if isinstance(item, dict):
                    claim = str(item.get("claim", "")).strip()
                    if claim:
                        claims.append(claim)
                elif isinstance(item, str):
                    item = item.strip()
                    if item:
                        claims.append(item)
        if isinstance(answer_support_claims_raw, list):
            for item in answer_support_claims_raw:
                if isinstance(item, dict):
                    claim = str(item.get("claim", "")).strip()
                    if claim:
                        answer_support_claims.append(claim)
                elif isinstance(item, str):
                    item = item.strip()
                    if item:
                        answer_support_claims.append(item)
        return GenerationOutput(
            answer=answer,
            support_claims=claims,
            answer_support_claims=answer_support_claims,
            raw_text=text,
            structured=parsed,
        )

    answer = ""
    claims: List[str] = []
    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        if not answer and re.search(r"^answer\s*[:\-]", line_clean, flags=re.IGNORECASE):
            answer = re.split(r"[:\-]", line_clean, maxsplit=1)[-1].strip()
            continue
        if line_clean.startswith(("-", "*")):
            claims.append(line_clean[1:].strip())
    if not answer:
        answer = text.splitlines()[0].strip() if text.strip() else "insufficient evidence"
    return GenerationOutput(
        answer=answer,
        support_claims=claims,
        answer_support_claims=list(claims),
        raw_text=text,
        structured={},
    )


def extract_claims_from_list_obj(items: Any) -> List[str]:
    claims: List[str] = []
    if not isinstance(items, list):
        return claims
    for item in items:
        if isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            if claim:
                claims.append(claim)
        elif isinstance(item, str):
            item = item.strip()
            if item:
                claims.append(item)
    return claims


def parse_distractor_output(text: str) -> List[str]:
    parsed = extract_json_dict(text)
    if parsed and isinstance(parsed.get("distractors", None), list):
        return [str(x).strip() for x in parsed["distractors"] if str(x).strip()]

    distractors: List[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if line:
            distractors.append(line)
    return distractors


def parse_policy_output(text: str) -> PolicyDecision:
    parsed = extract_json_dict(text)
    action = ""
    reason = ""
    if parsed is not None:
        action = str(parsed.get("action", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()

    if not action:
        t = text.lower()
        if "commit" in t or "stop" in t or "answerable" in t:
            action = "commit"
        else:
            action = "retrieve"

    # Only two policy actions are supported: commit or retrieve.
    if action not in {"commit", "retrieve"}:
        action = "retrieve"

    return PolicyDecision(
        action=action,
        reason=reason,
        raw_text=text,
    )


def unique_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        norm = normalize_answer(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item.strip())
    return out


def format_evidence(passages: Sequence[Passage]) -> str:
    blocks = []
    for p in passages:
        blocks.append(f"[{p.index}] Title: {p.title}\nParagraph: {p.text}")
    return "\n\n".join(blocks)


def build_passages(example: Dict[str, Any]) -> List[Passage]:
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    passages: List[Passage] = []
    for i, (title, sent_list) in enumerate(zip(titles, sentences)):
        paragraph = " ".join(sent_list).strip()
        passages.append(Passage(index=i, title=title, text=paragraph))
    return passages


def rank_passages(question: str, passages: Sequence[Passage]) -> Tuple[List[int], List[float]]:
    if len(passages) <= 1:
        return list(range(len(passages))), [1.0] * len(passages)

    corpus = [question] + [f"{p.title}. {p.text}" for p in passages]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).reshape(-1)
    order = np.argsort(-sims).tolist()
    return order, sims.tolist()


def disable_broken_torchvision_for_transformers() -> None:
    """
    Work around environments where torchvision is installed but incompatible with torch.
    In that case, transformers marks torchvision as available, then fails at import time.
    """
    try:
        import torchvision  # type: ignore # pylint: disable=unused-import,import-outside-toplevel
    except Exception:
        try:
            import transformers.utils.import_utils as hf_import_utils  # pylint: disable=import-outside-toplevel

            hf_import_utils._torchvision_available = False  # type: ignore[attr-defined]
            if "torchvision" in sys.modules:
                del sys.modules["torchvision"]
            os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
        except Exception:
            pass


class QwenDincoModel:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "float16",
    ) -> None:
        disable_broken_torchvision_for_transformers()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": None,
        }
        torch_dtype = dtype_map[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        self.yes_token_id = self._resolve_binary_token_id(["Yes", " yes"])
        self.no_token_id = self._resolve_binary_token_id(["No", " no"])

    def _resolve_binary_token_id(self, options: Sequence[str]) -> int:
        for opt in options:
            ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        fallback = self.tokenizer.encode(options[0], add_special_tokens=False)
        if not fallback:
            raise ValueError(f"Unable to resolve token id for {options[0]}")
        return fallback[0]

    @staticmethod
    def _fold_system_into_first_user(
        conversations: List[List[Dict[str, str]]],
    ) -> List[List[Dict[str, str]]]:
        """Merge any system messages into the first user turn.

        Gemma's chat template (and a handful of others) raise
        `jinja2.exceptions.TemplateError: System role not supported`.
        Prepend the system content to the first user message and drop
        the system turn so the template can render.
        """
        folded: List[List[Dict[str, str]]] = []
        for convo in conversations:
            sys_texts: List[str] = []
            remaining: List[Dict[str, str]] = []
            for msg in convo:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if content:
                        sys_texts.append(content)
                else:
                    remaining.append(msg)
            if sys_texts and remaining:
                first = dict(remaining[0])
                if first.get("role") == "user":
                    first["content"] = "\n\n".join(sys_texts + [first.get("content", "")]).strip()
                    remaining[0] = first
                else:
                    remaining.insert(
                        0,
                        {"role": "user", "content": "\n\n".join(sys_texts).strip()},
                    )
            elif sys_texts and not remaining:
                remaining = [{"role": "user", "content": "\n\n".join(sys_texts).strip()}]
            folded.append(remaining)
        return folded

    def _apply_chat_template_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        enable_thinking: bool = False,
    ) -> torch.Tensor:
        kwargs = dict(add_generation_prompt=True, padding=True, return_tensors="pt")
        try:
            return self.tokenizer.apply_chat_template(
                conversations,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            pass
        except Exception as exc:
            if "System role" not in str(exc):
                raise
            folded = self._fold_system_into_first_user(conversations)
            try:
                return self.tokenizer.apply_chat_template(
                    folded,
                    enable_thinking=enable_thinking,
                    **kwargs,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(folded, **kwargs)

        try:
            return self.tokenizer.apply_chat_template(conversations, **kwargs)
        except Exception as exc:
            if "System role" not in str(exc):
                raise
            folded = self._fold_system_into_first_user(conversations)
            return self.tokenizer.apply_chat_template(folded, **kwargs)

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 384, enable_thinking: bool = False) -> str:
        conversations = [[{"role": "user", "content": prompt}]]
        input_ids = self._apply_chat_template_batch(
            conversations,
            enable_thinking=enable_thinking,
        ).to(self._device())
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        text = self.tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True)
        return text.strip()

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 384,
        enable_thinking: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,  # noqa: ARG002 — HF backend ignores this
    ) -> Tuple[str, int]:
        """Generate from a full chat message list. Backend-agnostic.

        Returns ``(text, n_input_tokens)``. The agent runner's
        ``_generate_agent_action`` uses this so it works on either HF or vLLM
        without reaching into ``qwen_model.model.generate`` directly.
        ``json_schema`` is honoured by the vLLM subclass; HF ignores it.
        """
        conversations = [list(messages)]
        input_ids = self._apply_chat_template_batch(
            conversations,
            enable_thinking=enable_thinking,
        ).to(self._device())
        n_input_tokens = int(input_ids.shape[1])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        text = self.tokenizer.decode(
            outputs[0, input_ids.shape[1] :], skip_special_tokens=True,
        )
        return text.strip(), n_input_tokens

    @staticmethod
    def _clean_candidate_text(s: str) -> str:
        s = s.split("\n")[0]
        s = s.replace("Answer:", "")
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def clean_answer_for_dinco(self, answer: str) -> str:
        """Apply the same answer cleanup convention used by DINCO."""
        return self._clean_candidate_text(answer)

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        return bool(
            re.match(
                r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b",
                question,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _map_text_to_yes_no(text: str) -> Optional[str]:
        t = f" {normalize_answer(text)} "
        if not t.strip():
            return None
        if " yes " in t:
            return "yes"
        if " no " in t:
            return "no"
        if any(
            cue in t
            for cue in [
                " not ",
                " never ",
                " neither ",
                " different ",
                " not same ",
                " not the same ",
                " not located in the same ",
                " not located in same ",
            ]
        ):
            return "no"
        if any(cue in t for cue in [" both ", " same ", " equally ", " identical "]):
            return "yes"
        return None

    def shorten_answer_for_hotpot(self, question: str, answer: str) -> str:
        text = self.clean_answer_for_dinco(answer)
        text = text.strip(" \t\n\r\"'`")
        if not text:
            return "insufficient evidence"

        if self._is_yes_no_question(question):
            mapped = self._map_text_to_yes_no(text)
            if mapped is not None:
                return mapped

        match_series = re.match(r"^\s*(?:the\s+)?(.+?)\s+series\b", text, flags=re.IGNORECASE)
        if match_series:
            series_name = match_series.group(1).strip(" \t\n\r\"'`.,;:!?")
            if series_name:
                return series_name

        match_role = re.search(r"\bheld the position of\s+([^.,;]+)", text, flags=re.IGNORECASE)
        if match_role:
            role = match_role.group(1).strip(" \t\n\r\"'`.,;:!?")
            if role:
                return role

        text = re.sub(r"^(?:the answer is|it is|it's|this is|that is)\s+", "", text, flags=re.IGNORECASE)
        clause = re.split(
            r"(?:[.;]|\s+because\s+|\s+while\s+|\s+although\s+|\s+but\s+)",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        clause = clause.strip(" \t\n\r\"'`.,;:!?")
        if not clause:
            clause = text.strip(" \t\n\r\"'`.,;:!?")

        match_head = re.match(
            r"^(.+?)\s+\b(is|was|were|are|has|have|had|held|served|located|born|based)\b",
            clause,
            flags=re.IGNORECASE,
        )
        if match_head:
            head = match_head.group(1).strip(" \t\n\r\"'`.,;:!?")
            if head:
                clause = head

        words = clause.split()
        if len(words) > 8:
            clause = " ".join(words[:8])
        return clause if clause else "insufficient evidence"

    @staticmethod
    def _infer_expected_answer_type(question: str) -> str:
        q = question.lower().strip()
        if QwenDincoModel._is_yes_no_question(question):
            return "yes_no"
        if re.search(r"\b(what|which)\s+(government\s+)?(position|role|title|office|post)\b", q):
            return "role_title"
        if re.search(r"\b(who|whom|whose)\b", q):
            return "person_name"
        if re.search(r"\b(when|what year|what date)\b", q):
            return "date"
        if re.search(r"\b(how many|how much|what number)\b", q):
            return "number"
        if re.search(r"\b(where|which city|which state|which country|what country)\b", q):
            return "location"
        if re.search(r"\b(which organization|which company|which institution|which university)\b", q):
            return "organization"
        return "entity"

    @staticmethod
    def _looks_like_person_name(answer: str) -> bool:
        tokens = [t for t in re.split(r"\s+", answer.strip()) if t]
        if not (1 <= len(tokens) <= 4):
            return False
        good = 0
        for tok in tokens:
            if re.match(r"^[A-Z][A-Za-z'\\-\\.]+$", tok):
                good += 1
        return good == len(tokens)

    @staticmethod
    def _infer_answer_type(answer: str) -> str:
        a = (answer or "").strip()
        if not a:
            return "unknown"
        norm = normalize_answer(a)
        if norm in {"yes", "no"}:
            return "yes_no"
        if re.fullmatch(r"\d+(?:\.\d+)?", norm):
            return "number"
        if re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            norm,
        ) or re.search(r"\b\d{4}\b", norm):
            return "date"

        role_keywords = [
            "chief",
            "president",
            "minister",
            "secretary",
            "governor",
            "senator",
            "ambassador",
            "director",
            "officer",
            "protocol",
            "prime minister",
            "vice president",
            "position",
            "title",
            "office",
            "post",
        ]
        if any(k in norm for k in role_keywords):
            return "role_title"

        org_keywords = [
            "university",
            "college",
            "company",
            "corporation",
            "inc",
            "ltd",
            "committee",
            "agency",
            "department",
            "government",
            "association",
            "organization",
        ]
        if any(k in norm for k in org_keywords):
            return "organization"

        loc_keywords = [
            "city",
            "state",
            "country",
            "province",
            "county",
            "village",
            "town",
            "island",
            "river",
            "mountain",
            "district",
            "region",
        ]
        if any(k in norm for k in loc_keywords):
            return "location"

        if QwenDincoModel._looks_like_person_name(a):
            return "person_name"
        return "entity"

    @staticmethod
    def _answer_mentioned_in_claims(answer: str, claims: Sequence[str]) -> bool:
        norm_answer = normalize_answer(answer)
        if not norm_answer:
            return False
        for claim in claims:
            if norm_answer and norm_answer in normalize_answer(claim):
                return True
        return False

    @staticmethod
    def _extract_question_entities(question: str) -> List[str]:
        """
        Lightweight entity extraction for policy diagnostics.
        We favor conservative precision over recall.
        """
        q = re.sub(r"\s+", " ", question).strip()
        tokens = re.findall(r"[A-Za-z0-9'`.-]+", q)
        connectors = {"of", "the", "for", "in", "on", "to", "de", "la", "&", "and"}

        spans: List[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not re.match(r"^[A-Z]", tok):
                i += 1
                continue

            span_tokens = [tok]
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]
                if re.match(r"^[A-Z]", nxt):
                    span_tokens.append(nxt)
                    j += 1
                    continue
                if nxt.lower() in connectors and (j + 1) < len(tokens) and re.match(r"^[A-Z]", tokens[j + 1]):
                    span_tokens.append(nxt)
                    span_tokens.append(tokens[j + 1])
                    j += 2
                    continue
                break
            span = " ".join(span_tokens).strip()
            if span:
                spans.append(span)
            i = max(i + 1, j)

        # Remove common question lead tokens and overly short fragments.
        blocked = {
            "What",
            "Which",
            "Who",
            "Whom",
            "Whose",
            "When",
            "Where",
            "Why",
            "How",
            "Is",
            "Are",
            "Was",
            "Were",
            "Did",
            "Do",
            "Does",
            "Can",
            "Could",
            "Would",
            "Should",
            "Will",
            "Have",
            "Has",
            "Had",
        }
        blocked_lower = {b.lower() for b in blocked}
        entities: List[str] = []
        seen = set()
        for ent in spans:
            ent = ent.strip(" ,.;:!?\"'`()[]{}")
            if not ent or ent in blocked:
                continue
            # Remove leading blocked question words from the span.
            toks = ent.split()
            while toks and toks[0].lower() in blocked_lower:
                toks = toks[1:]
            if not toks:
                continue
            ent = " ".join(toks).strip()
            if not ent:
                continue

            # Split coordinated spans into separate entities only when each side
            # looks like a standalone multi-token entity (typically person names).
            # Keep title-like single-token coordination intact: "Kiss and Tell".
            coord_parts = [p.strip() for p in re.split(r"\s+(?:and|&)\s+", ent) if p.strip()]
            if len(coord_parts) > 1 and all(len(p.split()) >= 2 for p in coord_parts):
                parts = coord_parts
            else:
                parts = [ent]

            for part in parts:
                if part in blocked:
                    continue
                toks = part.split()
                if len(toks) == 1 and len(toks[0]) <= 2 and not toks[0].isupper():
                    continue
                norm_ent = normalize_answer(part)
                if not norm_ent or norm_ent in seen:
                    continue
                seen.add(norm_ent)
                entities.append(part)
        return entities

    @staticmethod
    def _entity_in_text(entity: str, text: str) -> bool:
        norm_ent = normalize_answer(entity)
        norm_text = normalize_answer(text)
        if not norm_ent or not norm_text:
            return False
        if norm_ent in norm_text:
            return True
        ent_tokens = [t for t in norm_ent.split() if t]
        text_tokens = set(norm_text.split())
        if not ent_tokens:
            return False
        overlap = sum(1 for t in ent_tokens if t in text_tokens) / max(1, len(ent_tokens))
        return overlap >= 0.8

    @staticmethod
    def _is_type_compatible(expected_answer_type: str, predicted_answer_type: str) -> bool:
        if expected_answer_type == "entity":
            return predicted_answer_type != "unknown"
        if expected_answer_type == "yes_no":
            return predicted_answer_type == "yes_no"
        if expected_answer_type == "role_title":
            return predicted_answer_type == "role_title"
        if expected_answer_type == "person_name":
            return predicted_answer_type == "person_name"
        if expected_answer_type == "date":
            return predicted_answer_type in {"date", "number"}
        if expected_answer_type == "number":
            return predicted_answer_type == "number"
        if expected_answer_type == "location":
            return predicted_answer_type in {"location", "entity"}
        if expected_answer_type == "organization":
            return predicted_answer_type in {"organization", "entity"}
        return True

    @staticmethod
    def _policy_answer_type_label(answer_type: str) -> str:
        mapping = {
            "person_name": "PERSON",
            "location": "LOCATION",
            "date": "DATE",
            "number": "NUMBER",
            "organization": "ORG",
            "role_title": "TITLE_ROLE",
            "yes_no": "YESNO",
            "entity": "OTHER",
            "unknown": "OTHER",
        }
        return mapping.get(answer_type, "OTHER")

    @staticmethod
    def _build_evidence_summary(passages: Sequence[Passage], max_chars: int = 900) -> str:
        if not passages:
            return ""
        snippets = []
        for p in passages:
            text = re.sub(r"\s+", " ", p.text).strip()
            if len(text) > 120:
                text = text[:120].rsplit(" ", 1)[0].strip() + " ..."
            snippets.append(f"{p.title}: {text}")
        summary = " | ".join(snippets)
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0].strip() + " ..."
        return summary

    def _build_policy_diagnostics(
        self,
        question: str,
        answer: str,
        grounding_claims: Sequence[str],
        claim_grounding_scores: Sequence[float],
    ) -> Dict[str, Any]:
        expected_type = self._infer_expected_answer_type(question)
        answer_type = self._infer_answer_type(answer)
        answer_type_match = self._is_type_compatible(expected_type, answer_type)
        answer_in_claims = self._answer_mentioned_in_claims(answer=answer, claims=grounding_claims)
        required_entities = self._extract_question_entities(question)
        covered_entities: List[str] = []
        missing_entities: List[str] = []
        for ent in required_entities:
            if any(self._entity_in_text(ent, c) for c in grounding_claims):
                covered_entities.append(ent)
            else:
                missing_entities.append(ent)

        # Relevant claims are those tied to required entities and/or explicit answer mention.
        relevant_claim_indices: List[int] = []
        norm_answer = normalize_answer(answer)
        for i, claim in enumerate(grounding_claims):
            is_relevant = False
            if norm_answer and self._entity_in_text(norm_answer, claim):
                is_relevant = True
            if not is_relevant and any(self._entity_in_text(ent, claim) for ent in required_entities):
                is_relevant = True
            if is_relevant:
                relevant_claim_indices.append(i)

        relevant_claim_scores: List[float] = []
        for idx in relevant_claim_indices:
            if 0 <= idx < len(claim_grounding_scores):
                relevant_claim_scores.append(float(claim_grounding_scores[idx]))

        relevant_min_grounding: Optional[float] = None
        relevant_mean_grounding: Optional[float] = None
        relevant_high_ground_fraction: Optional[float] = None
        if relevant_claim_scores:
            relevant_min_grounding = float(min(relevant_claim_scores))
            relevant_mean_grounding = float(np.mean(relevant_claim_scores))
            relevant_high_ground_fraction = float(
                np.mean([1.0 if float(s) >= 0.70 else 0.0 for s in relevant_claim_scores])
            )

        high_ground_frac: Optional[float] = None
        if claim_grounding_scores:
            high_ground_frac = float(np.mean([1.0 if float(s) >= 0.70 else 0.0 for s in claim_grounding_scores]))
        entity_coverage_ratio: Optional[float] = None
        if required_entities:
            entity_coverage_ratio = float(len(covered_entities) / max(1, len(required_entities)))

        return {
            "expected_answer_type": expected_type,
            "predicted_answer_type": answer_type,
            "answer_type_match": bool(answer_type_match),
            "answer_mentioned_in_grounding_claims": bool(answer_in_claims),
            "high_grounding_fraction_ge_0_70": high_ground_frac,
            "required_question_entities": required_entities,
            "covered_required_entities": covered_entities,
            "missing_required_entities": missing_entities,
            "entity_coverage_ratio": entity_coverage_ratio,
            "relevant_claim_indices": relevant_claim_indices,
            "relevant_claim_scores": relevant_claim_scores,
            "relevant_claims_min_grounding": relevant_min_grounding,
            "relevant_claims_mean_grounding": relevant_mean_grounding,
            "relevant_claims_high_grounding_fraction_ge_0_70": relevant_high_ground_fraction,
        }

    def lexical_clean_candidates(
        self,
        candidates: Sequence[str],
        candidate_scores: Sequence[float],
    ) -> Tuple[List[str], List[float]]:
        if not candidates:
            return [], []

        scores = torch.tensor(candidate_scores, dtype=torch.float32)
        sorted_is = torch.topk(scores, k=len(scores)).indices.tolist()

        cleaned: List[str] = []
        cleaned_scores: List[float] = []
        norm_seen = set()

        for i in sorted_is:
            c = self._clean_candidate_text(str(candidates[i]))
            if not c:
                continue

            norm_c = c.lower().replace(".", "")
            if norm_c in norm_seen:
                continue
            norm_seen.add(norm_c)

            cleaned.append(c)
            cleaned_scores.append(float(scores[i].item()))

        return cleaned, cleaned_scores

    def _build_dinco_answer_prompt(
        self,
        question: str,
        passages: Optional[Sequence[Passage]] = None,
    ) -> str:
        if passages:
            evidence = format_evidence(passages)
            return DINCO_BEAM_WITH_EVIDENCE_PROMPT.format(question=question, evidence=evidence)
        return DINCO_BEAM_PROMPT.format(question=question)

    def beam_search_answer_candidates(
        self,
        question: str,
        passages: Optional[Sequence[Passage]] = None,
        num_beams: int = 5,
        length_penalty: float = 0.0,
        max_new_tokens: int = 100,
    ) -> Tuple[List[str], List[float]]:
        prompt = self._build_dinco_answer_prompt(question=question, passages=passages)
        msg = [{"role": "user", "content": prompt}]
        input_ids = self._apply_chat_template_batch([msg]).to(self._device())
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
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
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        beam_strs = self.tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1] :], skip_special_tokens=True)
        beam_scores = outputs.sequences_scores.detach().cpu().tolist()
        return self.lexical_clean_candidates(beam_strs, beam_scores)

    def sample_answer_candidates(
        self,
        question: str,
        passages: Optional[Sequence[Passage]] = None,
        n_sample: int = 5,
        max_new_tokens: int = 100,
    ) -> List[str]:
        prompt = self._build_dinco_answer_prompt(question=question, passages=passages)
        msg = [{"role": "user", "content": prompt}]
        input_ids = self._apply_chat_template_batch([msg]).to(self._device())
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids.repeat(max(1, int(n_sample)), 1),
                attention_mask=attention_mask.repeat(max(1, int(n_sample)), 1),
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=None,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        sampled = self.tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1] :], skip_special_tokens=True)
        return [self.clean_answer_for_dinco(text) for text in sampled]

    def batch_yes_probability(
        self,
        question: str,
        candidates: Sequence[str],
        passages: Optional[Sequence[Passage]] = None,
    ) -> List[float]:
        evidence = format_evidence(passages) if passages else ""
        prompts = []
        for cand in candidates:
            if passages:
                content = PTRUE_WITH_EVIDENCE_PROMPT.format(
                    question=question,
                    evidence=evidence,
                    candidate_answer=cand,
                )
            else:
                content = PTRUE_PROMPT.format(question=question, candidate_answer=cand)
            prompts.append([{"role": "user", "content": content}])
        input_ids = self._apply_chat_template_batch(prompts).to(self._device())
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, [self.yes_token_id, self.no_token_id]]
        probs = torch.softmax(logits, dim=-1)[:, 0]
        return probs.detach().cpu().tolist()

    def generate_answer_and_claims(
        self,
        question: str,
        passages: Sequence[Passage],
        refinement: bool = False,
        previous_answer: str = "",
        previous_claims: Optional[Sequence[str]] = None,
    ) -> GenerationOutput:
        evidence = format_evidence(passages)
        if refinement:
            prompt = REFINEMENT_PROMPT.format(
                question=question,
                evidence=evidence,
                previous_answer=previous_answer,
                previous_claims=json.dumps(list(previous_claims or []), ensure_ascii=True),
            )
        else:
            prompt = ANSWER_AND_CLAIMS_PROMPT.format(question=question, evidence=evidence)

        raw = self.generate(prompt, max_new_tokens=700)
        parsed = parse_generation_output(raw)
        parsed.answer = self.shorten_answer_for_hotpot(question=question, answer=parsed.answer)
        if not parsed.answer:
            parsed.answer = "insufficient evidence"
        parsed.support_claims = unique_keep_order(parsed.support_claims)
        parsed.answer_support_claims = unique_keep_order(parsed.answer_support_claims)
        if not parsed.answer_support_claims:
            parsed.answer_support_claims = list(parsed.support_claims)
        return parsed

    def generate_distractors(
        self,
        question: str,
        passages: Sequence[Passage],
        main_candidate: str,
        seed_candidates: Optional[Sequence[str]] = None,
        n_distractors: int = 5,
    ) -> List[str]:
        evidence = format_evidence(passages)
        other_candidates = [
            self.clean_answer_for_dinco(str(c)) for c in (seed_candidates or []) if str(c).strip()
        ]
        prompt = DISTRACTOR_WITH_EVIDENCE_PROMPT.format(
            question=question,
            evidence=evidence,
            main_candidate=self.clean_answer_for_dinco(main_candidate),
            other_candidates=json.dumps(other_candidates, ensure_ascii=True),
            n_distractors=max(1, n_distractors),
        )
        raw = self.generate(prompt, max_new_tokens=320)
        parsed = parse_distractor_output(raw)

        cleaned: List[str] = []
        seen = {normalize_answer(main_candidate)}
        for cand in parsed:
            c = self.clean_answer_for_dinco(str(cand))
            norm_c = normalize_answer(c)
            if not c or not norm_c or norm_c in seen:
                continue
            seen.add(norm_c)
            cleaned.append(c)

        return cleaned[: max(0, n_distractors)]

    def decide_policy_action(
        self,
        original_question: str,
        effective_question: str,
        passages: Sequence[Passage],
        answer: str,
        support_claims: Sequence[str],
        answer_support_claims: Sequence[str],
        grounding_claims: Sequence[str],
        nvc: float,
        grounding_g: Optional[float],
        claim_grounding_scores: Sequence[float],
        hop: int,
        max_hops: int,
        nvc_low: float = 0.70,
        nvc_high: float = 0.80,
        grounding_low: float = 0.70,
        support_low: float = 0.50,
        support_high: float = 0.70,
        min_critical_supported: int = 1,
        max_evidence_chars: int = 5000,
    ) -> PolicyDecision:
        diagnostics = self._build_policy_diagnostics(
            question=effective_question,
            answer=answer,
            grounding_claims=grounding_claims,
            claim_grounding_scores=claim_grounding_scores,
        )
        evidence = re.sub(r"\s+", " ", format_evidence(passages)).strip()
        if len(evidence) > max_evidence_chars:
            evidence = f"{evidence[:max_evidence_chars].rsplit(' ', 1)[0].strip()} ..."

        relevant_indices = set(diagnostics.get("relevant_claim_indices", []) or [])
        claim_objects: List[Dict[str, Any]] = []
        for i, claim in enumerate(grounding_claims):
            support_prob = 0.0
            if i < len(claim_grounding_scores):
                support_prob = float(claim_grounding_scores[i])
            claim_objects.append(
                {
                    "claim": str(claim),
                    "support_prob": max(0.0, min(1.0, float(support_prob))),
                    "is_answer_critical": bool(i in relevant_indices),
                }
            )

        # Policy-facing grounding score should focus on answer-critical claims.
        policy_grounding_g = grounding_g
        relevant_mean = diagnostics.get("relevant_claims_mean_grounding", None)
        if relevant_mean is not None:
            policy_grounding_g = float(relevant_mean)

        state = {
            "question": effective_question,
            "current_answer": answer,
            "hop": hop,
            "max_hops": max_hops,
            "nvc": float(nvc),
            "grounding_g": None if policy_grounding_g is None else float(policy_grounding_g),
            "claim_grounding_scores": claim_objects,
            "thresholds": {
                "nvc_low": float(nvc_low),
                "nvc_high": float(nvc_high),
                "grounding_low": float(grounding_low),
                "support_low": float(support_low),
                "support_high": float(support_high),
                "min_critical_supported": int(min_critical_supported),
            },
            "predicted_answer_type": self._policy_answer_type_label(str(diagnostics.get("predicted_answer_type", "other"))),
            "required_entities": diagnostics.get("required_question_entities", []),
            "evidence_summary": self._build_evidence_summary(passages),
            # Additional context retained for debugging.
            "original_question": original_question,
            "support_claims": list(support_claims),
            "answer_support_claims": list(answer_support_claims),
            "grounding_claims": list(grounding_claims),
            "policy_diagnostics": diagnostics,
            "overall_grounding_g": None if grounding_g is None else float(grounding_g),
        }
        prompt = POLICY_DECISION_PROMPT.format(state_json=json.dumps(state, ensure_ascii=True, indent=2))
        raw = self.generate(prompt, max_new_tokens=320)
        decision = parse_policy_output(raw)
        decision.diagnostics = diagnostics

        # Advisory signals for analysis/debugging only.
        # These are not hard vetoes; the LLM policy action is preserved.
        advisory_flags: List[str] = []
        if not diagnostics["answer_type_match"]:
            advisory_flags.append("answer_type_mismatch")
        if diagnostics["expected_answer_type"] != "yes_no" and not diagnostics["answer_mentioned_in_grounding_claims"]:
            advisory_flags.append("answer_not_in_grounding_claims")

        required_entities = diagnostics.get("required_question_entities", []) or []
        if nvc < nvc_low:
            advisory_flags.append(f"nvc_below_low:{nvc:.3f}")
        if grounding_g is None:
            advisory_flags.append("grounding_missing")
        elif float(grounding_g) < grounding_low:
            advisory_flags.append(f"grounding_below_low:{float(grounding_g):.3f}")

        if not claim_objects:
            advisory_flags.append("no_claim_grounding_scores")

        critical_claims = [c for c in claim_objects if c["is_answer_critical"]]
        if not critical_claims:
            advisory_flags.append("no_answer_critical_claims")
        critical_supported = sum(1 for c in critical_claims if float(c["support_prob"]) >= support_high)
        if critical_supported < int(min_critical_supported):
            advisory_flags.append(
                f"critical_supported_below_min:{critical_supported}<{int(min_critical_supported)}"
            )

        if required_entities:
            missing_required = []
            for ent in required_entities:
                ok = any(
                    self._entity_in_text(ent, c["claim"]) and float(c["support_prob"]) >= support_low for c in claim_objects
                )
                if not ok:
                    missing_required.append(ent)
            if missing_required:
                advisory_flags.append("required_entities_not_supported:" + ",".join(missing_required[:6]))

        decision.guard_triggers = advisory_flags
        return decision

class DincoCalibrator:
    def __init__(
        self,
        qwen_model: QwenDincoModel,
        nli_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        cache_dir: Optional[str] = None,
        n_sc_samples: int = 5,
        sc_match_threshold: float = 0.9,
    ) -> None:
        self.qwen_model = qwen_model
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, cache_dir=cache_dir)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name, device_map="auto", cache_dir=cache_dir
        )
        self.n_sc_samples = max(1, int(n_sc_samples))
        self.sc_match_threshold = float(sc_match_threshold)

    @staticmethod
    def _short_evidence_for_nli(passages: Sequence[Passage], max_chars: int = 1800) -> str:
        if not passages:
            return ""
        text = re.sub(r"\s+", " ", format_evidence(passages)).strip()
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
        return f"{truncated} ..."

    def _shorten_main_candidate_with_beam(
        self,
        question: str,
        main_candidate: str,
        beam_candidates: Sequence[str],
        passages: Optional[Sequence[Passage]] = None,
        max_words: int = 6,
        min_semantic_equivalence: float = 0.55,
    ) -> str:
        """
        Keep candidate[0] aligned to claims_generation_answer, but prefer a short
        phrase if beam produced a semantically equivalent concise candidate.
        """
        main_clean = self.qwen_model.clean_answer_for_dinco(main_candidate)
        if not main_clean:
            return "insufficient evidence"
        if len(main_clean.split()) <= max_words:
            return main_clean

        short_candidates: List[str] = []
        seen = set()
        for cand in beam_candidates:
            c = self.qwen_model.clean_answer_for_dinco(str(cand))
            if not c:
                continue
            if len(c.split()) > max_words:
                continue
            norm_c = normalize_answer(c)
            if not norm_c or norm_c in seen:
                continue
            seen.add(norm_c)
            short_candidates.append(c)

        if not short_candidates:
            return main_clean

        evidence_text = self._short_evidence_for_nli(passages or [])
        premises: List[str] = []
        hypotheses: List[str] = []
        for cand in short_candidates:
            if evidence_text:
                premises.append(f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {main_clean}")
                hypotheses.append(f"Answer: {cand}")
                premises.append(f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {cand}")
                hypotheses.append(f"Answer: {main_clean}")
            else:
                premises.append(f"Question: {question}\nAnswer: {main_clean}")
                hypotheses.append(f"Answer: {cand}")
                premises.append(f"Question: {question}\nAnswer: {cand}")
                hypotheses.append(f"Answer: {main_clean}")

        inputs = self.nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(next(self.nli_model.parameters()).device)
        with torch.no_grad():
            probs = torch.softmax(self.nli_model(**inputs).logits, dim=-1).cpu()

        best_candidate = main_clean
        best_score = -1.0
        for i, cand in enumerate(short_candidates):
            entail_forward = float(probs[2 * i, 0].item())
            entail_backward = float(probs[2 * i + 1, 0].item())
            semantic_equiv = min(entail_forward, entail_backward)
            if semantic_equiv > best_score:
                best_score = semantic_equiv
                best_candidate = cand
            elif semantic_equiv == best_score:
                if len(cand.split()) < len(best_candidate.split()) or (
                    len(cand.split()) == len(best_candidate.split()) and len(cand) < len(best_candidate)
                ):
                    best_candidate = cand

        if best_score >= min_semantic_equivalence:
            return best_candidate
        return main_clean

    def _pairwise_nli(
        self,
        question: str,
        candidates: Sequence[str],
        passages: Optional[Sequence[Passage]] = None,
    ) -> torch.Tensor:
        n = len(candidates)
        nlis = torch.zeros((n, n, 3), dtype=torch.float32)
        if n <= 1:
            return nlis

        evidence_text = self._short_evidence_for_nli(passages or [])
        premises, hypotheses, pairs = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if evidence_text:
                    premises.append(f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {candidates[i]}")
                else:
                    premises.append(f"Question: {question}\nAnswer: {candidates[i]}")
                hypotheses.append(f"Answer: {candidates[j]}")
                pairs.append((i, j))

        inputs = self.nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(next(self.nli_model.parameters()).device)

        with torch.no_grad():
            probs = torch.softmax(self.nli_model(**inputs).logits, dim=-1).cpu()
        for (i, j), p in zip(pairs, probs):
            nlis[i, j] = p
        return nlis

    def _self_consistency_entailments(
        self,
        question: str,
        main_candidate: str,
        sampled_generations: Sequence[str],
        passages: Optional[Sequence[Passage]] = None,
    ) -> List[float]:
        main_clean = self.qwen_model.clean_answer_for_dinco(main_candidate)
        if not main_clean:
            return []

        evidence_text = self._short_evidence_for_nli(passages or [])
        entail_i = 0
        entailments = torch.zeros(len(sampled_generations), dtype=torch.float32)
        premises: List[str] = []
        hypotheses: List[str] = []
        sample_indices: List[int] = []

        for sample_i, sampled in enumerate(sampled_generations):
            sampled_clean = self.qwen_model.clean_answer_for_dinco(sampled)
            if not sampled_clean:
                continue
            if sampled_clean == main_clean:
                entailments[sample_i] = 1.0
                continue
            if evidence_text:
                premises.append(f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {main_clean}")
                hypotheses.append(f"Answer: {sampled_clean}")
                premises.append(f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {sampled_clean}")
                hypotheses.append(f"Answer: {main_clean}")
            else:
                premises.append(f"Question: {question}\nAnswer: {main_clean}")
                hypotheses.append(f"Answer: {sampled_clean}")
                premises.append(f"Question: {question}\nAnswer: {sampled_clean}")
                hypotheses.append(f"Answer: {main_clean}")
            sample_indices.append(sample_i)

        if premises:
            inputs = self.nli_tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(next(self.nli_model.parameters()).device)
            with torch.no_grad():
                probs = torch.softmax(self.nli_model(**inputs).logits, dim=-1)[:, entail_i].cpu()
            for idx, sample_i in enumerate(sample_indices):
                entailments[sample_i] = float((probs[2 * idx] + probs[2 * idx + 1]).item() / 2.0)

        return [float(x) for x in entailments.tolist()]

    def _compute_sc_conf(self, sc_entailments: Sequence[float]) -> float:
        values = list(sc_entailments) + [1.0]
        if not values:
            return 1.0
        matches = [1.0 if float(v) > self.sc_match_threshold else 0.0 for v in values]
        return float(np.mean(matches))

    def _entail_both_score(
        self,
        question: str,
        a: str,
        b: str,
        passages: Optional[Sequence[Passage]] = None,
    ) -> float:
        evidence_text = self._short_evidence_for_nli(passages or [])
        if evidence_text:
            premises = [
                f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {a}",
                f"Question: {question}\nEvidence: {evidence_text}\nAnswer: {b}",
            ]
        else:
            premises = [
                f"Question: {question}\nAnswer: {a}",
                f"Question: {question}\nAnswer: {b}",
            ]
        hypotheses = [f"Answer: {b}", f"Answer: {a}"]
        inputs = self.nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(next(self.nli_model.parameters()).device)
        with torch.no_grad():
            probs = torch.softmax(self.nli_model(**inputs).logits, dim=-1).cpu()

        entail_ab = float(probs[0, 0].item())
        entail_ba = float(probs[1, 0].item())
        return float(min(entail_ab, entail_ba))

    def question_conditioned_bidirectional_entailment(
        self,
        question: str,
        pred_answer: str,
        gold_answer: str,
        passages: Optional[Sequence[Passage]] = None,
    ) -> float:
        pred_clean = self.qwen_model.clean_answer_for_dinco(pred_answer)
        gold_clean = self.qwen_model.clean_answer_for_dinco(gold_answer)
        if not pred_clean or not gold_clean:
            return 0.0
        return self._entail_both_score(
            question=question,
            a=pred_clean,
            b=gold_clean,
            passages=passages,
        )

    @staticmethod
    def _text_cosine_similarity(a: str, b: str) -> float:
        try:
            tfidf = TfidfVectorizer(stop_words="english").fit_transform([a, b])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
            return float(sim)
        except ValueError:
            # Happens when both strings are empty/stopwords-only.
            return 0.0

    def _filter_semantic_distractors(
        self,
        question: str,
        passages: Sequence[Passage],
        main_candidate: str,
        distractors: Sequence[str],
        entail_threshold: float = 0.8,
        cosine_threshold: float = 0.9,
    ) -> List[str]:
        kept: List[str] = []
        entail_cache: Dict[Tuple[str, str], float] = {}
        cosine_cache: Dict[Tuple[str, str], float] = {}

        def pair_key(x: str, y: str) -> Tuple[str, str]:
            return (x, y) if x <= y else (y, x)

        def entail_both(x: str, y: str) -> float:
            k = pair_key(x, y)
            if k not in entail_cache:
                entail_cache[k] = self._entail_both_score(question=question, a=x, b=y, passages=passages)
            return entail_cache[k]

        def cos_sim(x: str, y: str) -> float:
            k = pair_key(x, y)
            if k not in cosine_cache:
                cosine_cache[k] = self._text_cosine_similarity(x, y)
            return cosine_cache[k]

        for cand in distractors:
            # Drop if semantically equivalent to main candidate.
            if entail_both(main_candidate, cand) > entail_threshold:
                continue

            # Drop if too similar/equivalent to already-kept distractors.
            duplicate = False
            for kept_cand in kept:
                if cos_sim(cand, kept_cand) > cosine_threshold or entail_both(cand, kept_cand) > entail_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(cand)
        return kept

    @staticmethod
    def _compute_nvc(ptrues: torch.Tensor, nlis: torch.Tensor) -> float:
        if ptrues.numel() == 1:
            return float(ptrues[0].item())

        entail_i = 0
        contra_i = 2
        sym_nlis = (nlis + nlis.transpose(0, 1)) / 2
        contra_weights = sym_nlis[:, :, contra_i]
        sims = nlis[:, :, entail_i]
        degrees = torch.sum(torch.maximum(torch.tensor(0.0), sims), dim=0) + 1.0

        main = 0
        numerator = ptrues[main]
        denominator = numerator.clone()
        eps = 1e-6
        for j in range(ptrues.shape[0]):
            if j == main:
                continue
            contrib = ptrues[j] * contra_weights[main, j] / (degrees[j] - sims[main, j] + eps)
            denominator += contrib

        if denominator > 1:
            return float((numerator / denominator).item())
        return float(numerator.item())

    @staticmethod
    def _sampling_norm(s: str) -> str:
        """Normalize a string for sampling-DINCO dedupe + agreement comparison.

        Updated to use HotpotQA-style ``normalize_answer`` semantics:
        - lowercase
        - strip articles (a/an/the)
        - remove all punctuation
        - collapse whitespace

        This is stronger than the original (lowercase + strip period only) so
        that lexical variants of the same factoid answer (e.g. "Einstein" vs
        "Albert Einstein" vs "Einstein, Albert") collapse to the same form
        rather than being treated as distinct distractors. The original
        per-DINCO formula used a weaker norm; for short-answer HotpotQA the
        agent-facing agreement_rate metric needs the stronger one to avoid
        false-alarm "low agreement" signals on lexical paraphrases.
        """
        # First strip generation cruft.
        s = (s or "").split('\n')[0]
        s = s.replace('Answer:', '').strip()
        # Then HotpotQA-style normalize.
        s = s.lower()
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ' '.join(s.split())
        return s

    @classmethod
    def _dedupe_with_greedy_first(cls, greedy: str, samples: Sequence[str]) -> List[str]:
        """Return cleaned candidate list with greedy at index 0 and no duplicates."""
        def clean(s: str) -> str:
            t = (s or "").split('\n')[0]
            t = t.replace('Answer:', '').strip()
            t = re.sub(r'\s+', ' ', t)
            return t

        out: List[str] = []
        seen: set = set()
        greedy_clean = clean(greedy)
        if greedy_clean:
            out.append(greedy_clean)
            seen.add(cls._sampling_norm(greedy_clean))
        for s in samples:
            cs = clean(s)
            if not cs:
                continue
            n = cls._sampling_norm(cs)
            if n in seen:
                continue
            seen.add(n)
            out.append(cs)
        return out

    def compute_post_retrieval_sampling(
        self,
        *,
        question: str,
        answer: str,
        passages: Sequence[Passage],
        n_samples: int = 10,
    ) -> SamplingDincoResult:
        """Sampling-DINCO with graceful-degenerate, post-retrieval.

        Differs from `compute()`:
        - Distractors come from stochastic sampling (T=1.0, top_p=0.95) instead
          of beam search.
        - When samples collapse onto the greedy answer (n_unique_distractors==0),
          NVC degenerates to raw P(True) on the greedy and `degenerate=True`
          is exposed so downstream consumers can distinguish "DINCO normalized
          real distractors" from "DINCO had nothing to normalize."

        Validated in the dinco-beam-vs-sampling-hotpotqa canary (N=18 cases
        beam-DINCO would have dropped). Better-calibrated than beam-DINCO in
        the context-grounded regime.
        """
        # 1. Stochastic samples conditioned on the retrieved passages
        raw_samples = self.qwen_model.sample_answer_candidates(
            question=question,
            passages=passages,
            n_sample=n_samples,
            max_new_tokens=100,
        )
        # 2. Lexical clean + dedupe; greedy at index 0
        cleaned = self._dedupe_with_greedy_first(answer, raw_samples)
        if not cleaned:
            fallback = self.qwen_model.clean_answer_for_dinco(answer) if answer else ""
            cleaned = [fallback or "insufficient evidence"]
        n_unique_distractors = max(0, len(cleaned) - 1)
        degenerate = n_unique_distractors == 0
        # 3. Agreement rate — fraction of samples matching greedy after normalize
        greedy_norm = self._sampling_norm(answer)
        if raw_samples:
            agreement_rate = sum(
                1 for s in raw_samples if self._sampling_norm(s) == greedy_norm
            ) / len(raw_samples)
        else:
            agreement_rate = 1.0
        # 4. P(True) for each candidate WITH context
        ptrues_list = self.qwen_model.batch_yes_probability(
            question=question,
            candidates=cleaned,
            passages=passages,
        )
        raw_verbal_ptrue = float(ptrues_list[0])
        # 5. NVC with graceful degenerate
        if not degenerate:
            ptrues_t = torch.tensor(ptrues_list, dtype=torch.float32)
            nlis_t = self._pairwise_nli(question=question, candidates=cleaned, passages=passages)
            sampling_dinco_conf = self._compute_nvc(ptrues=ptrues_t, nlis=nlis_t)
            nli_list = nlis_t.tolist()
        else:
            sampling_dinco_conf = raw_verbal_ptrue
            nli_list = []
        return SamplingDincoResult(
            sampling_dinco_conf=float(sampling_dinco_conf),
            degenerate=degenerate,
            agreement_rate=float(agreement_rate),
            n_unique_distractors=int(n_unique_distractors),
            candidates=list(cleaned),
            raw_samples=list(raw_samples),
            ptrues=[float(x) for x in ptrues_list],
            raw_verbal_ptrue=raw_verbal_ptrue,
            nli=nli_list,
        )

    def compute(
        self,
        question: str,
        answer: str,
        passages: Sequence[Passage],
        n_distractors: int,
    ) -> DincoResult:
        beam_candidates, _ = self.qwen_model.beam_search_answer_candidates(
            question=question,
            passages=passages,
            num_beams=max(2, n_distractors),
            length_penalty=0.0,
            max_new_tokens=100,
        )
        candidates = list(beam_candidates)
        if not candidates:
            fallback = self.qwen_model.clean_answer_for_dinco(answer) if answer else ""
            candidates = [fallback or "insufficient evidence"]
        ptrues_list = self.qwen_model.batch_yes_probability(
            question=question,
            candidates=candidates,
            passages=passages,
        )

        ptrues = torch.tensor(ptrues_list, dtype=torch.float32)
        nlis = self._pairwise_nli(question=question, candidates=candidates, passages=passages)
        nvc = self._compute_nvc(ptrues=ptrues, nlis=nlis)
        sampled_generations = self.qwen_model.sample_answer_candidates(
            question=question,
            passages=passages,
            n_sample=self.n_sc_samples,
            max_new_tokens=100,
        )
        sc_entailments = self._self_consistency_entailments(
            question=question,
            main_candidate=candidates[0],
            sampled_generations=sampled_generations,
            passages=passages,
        )
        sc_conf = self._compute_sc_conf(sc_entailments=sc_entailments)
        final_conf = float((nvc + sc_conf) / 2.0)
        return DincoResult(
            nvc=nvc,
            candidates=list(candidates),
            ptrues=[float(x) for x in ptrues_list],
            nli=nlis.tolist(),
            sc_conf=float(sc_conf),
            final_conf=final_conf,
            sampled_generations=list(sampled_generations),
            sc_entailments=list(sc_entailments),
        )


class MiniCheckGrounder:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        enable_prefix_caching: bool = True,
        model_name: str = "Bespoke-MiniCheck-7B",
        allow_cpu_fallback: bool = False,
        cpu_fallback_model_name: str = "roberta-large",
        gpu_memory_utilization: float = 0.8,
    ) -> None:
        disable_broken_torchvision_for_transformers()

        try:
            import nltk

            for resource in ("punkt", "punkt_tab"):
                try:
                    nltk.data.find(f"tokenizers/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)
        except Exception:
            # MiniCheck can still run if tokenizers are already available.
            pass

        minicheck_dir = Path(__file__).resolve().parent / "MiniCheck"
        if str(minicheck_dir) not in sys.path:
            sys.path.insert(0, str(minicheck_dir))
        from minicheck.minicheck import MiniCheck  # pylint: disable=import-error

        requested_model = model_name
        selected_model = requested_model

        if requested_model == "Bespoke-MiniCheck-7B" and (not torch.cuda.is_available()) and allow_cpu_fallback:
            print(
                f"[MiniCheckGrounder] CUDA not available; falling back from Bespoke-MiniCheck-7B to {cpu_fallback_model_name}.",
                flush=True,
            )
            selected_model = cpu_fallback_model_name

        try:
            init_kwargs: Dict[str, Any] = dict(
                model_name=selected_model,
                cache_dir=cache_dir,
                max_model_len=max_model_len,
            )
            if selected_model == "Bespoke-MiniCheck-7B":
                init_kwargs.update(
                    tensor_parallel_size=tensor_parallel_size,
                    enable_prefix_caching=enable_prefix_caching,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
            self.model = MiniCheck(**init_kwargs)
            self.model_name = selected_model
        except Exception as exc:
            if requested_model == "Bespoke-MiniCheck-7B" and allow_cpu_fallback:
                print(
                    f"[MiniCheckGrounder] Failed to init Bespoke-MiniCheck-7B ({type(exc).__name__}: {exc}). "
                    f"Falling back to {cpu_fallback_model_name}.",
                    flush=True,
                )
                self.model = MiniCheck(
                    model_name=cpu_fallback_model_name,
                    cache_dir=cache_dir,
                    max_model_len=max_model_len,
                )
                self.model_name = cpu_fallback_model_name
            else:
                raise RuntimeError(
                    "MiniCheck initialization failed. If this environment is CPU-only, run with "
                    "--minicheck_model_name roberta-large/deberta-v3-large/flan-t5-large "
                    "or add --allow_minicheck_cpu_fallback. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc

    def score(self, passages: Sequence[Passage], claims: Sequence[str]) -> Tuple[float, List[float]]:
        if not claims:
            return 0.0, []
        evidence_doc = format_evidence(passages)
        docs = [evidence_doc] * len(claims)
        _, probs, _, _ = self.model.score(docs=docs, claims=list(claims))
        probs = [float(x) for x in probs]
        g = float(np.mean(probs)) if probs else 0.0
        return g, probs


class QwenVLLMDincoModel(QwenDincoModel):
    """vLLM-backed sibling of QwenDincoModel with the same public surface.

    Inherits the text helpers (`_clean_candidate_text`, `clean_answer_for_dinco`,
    `lexical_clean_candidates`, `_build_dinco_answer_prompt`, `_resolve_binary_token_id`,
    `_is_yes_no_question`, `_map_text_to_yes_no`, `shorten_answer_for_hotpot`) and
    high-level methods (`generate_answer_and_claims`, `generate_distractors`,
    `decide_policy_action`) which all route through `self.generate(prompt, ...)`.

    Overrides the model-bound methods to use vLLM:
    - `__init__`: skips parent HF load; loads vLLM `LLM(...)` instead.
    - `generate`: vLLM `LLM.generate(SamplingParams(temperature=0))`
    - `generate_chat`: NEW abstraction also added to the parent class.
    - `sample_answer_candidates`: vLLM `SamplingParams(n=N, T=1.0, top_p=0.95)`.
    - `batch_yes_probability`: vLLM logprobs extraction over Yes/No token SETS
      (richer than parent's single-token Yes/No id).
    - `beam_search_answer_candidates`: vLLM `LLM.beam_search` + prompt-prefix strip
      and `<|im_end|>` cleanup (matches the canary's lessons).

    Memory: when both Qwen-vLLM and MiniCheck-vLLM coexist on a your GPU, set
    `gpu_memory_utilization` here to ~0.78 and MiniCheck to ~0.15.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        gpu_memory_utilization: float = 0.78,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        tensor_parallel_size: int = 1,
    ) -> None:
        # Skip QwenDincoModel.__init__ — it loads HF AutoModelForCausalLM.
        # Manually replicate the tokenizer setup, then load via vLLM.
        disable_broken_torchvision_for_transformers()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from vllm import LLM  # local import to avoid hard dep when only HF backend used

        print(
            f"[QwenVLLMDincoModel] Loading {model_name} via vLLM — dtype={dtype} "
            f"max_model_len={max_model_len} gpu_memory_utilization={gpu_memory_utilization} "
            f"enforce_eager={enforce_eager} tensor_parallel_size={tensor_parallel_size}",
            flush=True,
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            download_dir=cache_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
        )

        # No HF model — set None so any accidental .model.generate fails loudly.
        self.model = None

        # Single-token Yes/No id (parent compatibility — used by `compute()` paths
        # that expect a scalar token id).
        self.yes_token_id = self._resolve_binary_token_id(["Yes", " yes"])
        self.no_token_id = self._resolve_binary_token_id(["No", " no"])

        # Richer token sets for top-K logprob extraction (canary lesson: tokenizer
        # may emit "Yes", " Yes", "yes", " yes", etc. as distinct tokens).
        self.yes_token_set: set = self._build_token_set(
            ["Yes", " Yes", "yes", " yes", "YES", " YES"]
        )
        self.no_token_set: set = self._build_token_set(
            ["No", " No", "no", " no", "NO", " NO"]
        )
        if not self.yes_token_set or not self.no_token_set:
            raise RuntimeError(
                f"Yes/No token sets must be non-empty. yes={self.yes_token_set}, "
                f"no={self.no_token_set}. Tokenizer: {self.tokenizer.__class__.__name__}"
            )

    def _build_token_set(self, variants: Sequence[str]) -> set:
        out: set = set()
        for v in variants:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
        return out

    def _device(self) -> torch.device:
        # vLLM manages its own GPU placement. Return CPU for any caller that
        # tries to .to(self._device()) — those should be refactored.
        return torch.device("cpu")

    def _build_chat_prompt_string(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
    ) -> str:
        """Render a chat message list to a prompt string for vLLM.

        Pins `enable_thinking=False` for Qwen3 by default (matches the canary).
        Folds system messages into the first user turn if the chat template
        doesn't accept them (matches parent's _fold_system_into_first_user).
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            if "System role" not in str(exc):
                raise
            folded = self._fold_system_into_first_user([list(messages)])[0]
            try:
                return self.tokenizer.apply_chat_template(
                    folded,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    folded,
                    tokenize=False,
                    add_generation_prompt=True,
                )

    def generate(self, prompt: str, max_new_tokens: int = 384, enable_thinking: bool = False) -> str:
        """Single greedy generation. Same surface as parent."""
        from vllm import SamplingParams
        prompt_str = self._build_chat_prompt_string(
            [{"role": "user", "content": prompt}], enable_thinking=enable_thinking,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        out = self.llm.generate([prompt_str], params, use_tqdm=False)[0]
        return out.outputs[0].text.strip()

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 384,
        enable_thinking: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """Generate from a full chat message list. Returns (text, n_input_tokens).

        Backend-agnostic abstraction used by the agent runner's
        `_generate_agent_action`. Parent class has the same method (HF-backed).
        If `json_schema` is provided, vLLM's structured-outputs constraint is
        applied so the model emits parseable JSON (necessary for Mistral-Small,
        which otherwise hits ~100% JSON parse-failure under long agent prompts).
        """
        from vllm import SamplingParams
        prompt_str = self._build_chat_prompt_string(messages, enable_thinking=enable_thinking)
        kwargs: Dict[str, Any] = {"temperature": 0.0, "max_tokens": max_new_tokens}
        if json_schema is not None:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
                if not getattr(self, "_json_logged", False):
                    print("[hotpot-agent] guided JSON: using StructuredOutputsParams (vLLM 0.16+)", flush=True)
                    self._json_logged = True
            except ImportError:
                try:
                    from vllm.sampling_params import GuidedDecodingParams
                    kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
                    if not getattr(self, "_json_logged", False):
                        print("[hotpot-agent] guided JSON: using legacy GuidedDecodingParams", flush=True)
                        self._json_logged = True
                except ImportError:
                    if not getattr(self, "_json_logged", False):
                        print("[hotpot-agent] WARN: no guided-decoding API in vLLM; running unconstrained", flush=True)
                        self._json_logged = True
        params = SamplingParams(**kwargs)
        out = self.llm.generate([prompt_str], params, use_tqdm=False)[0]
        n_input_tokens = len(out.prompt_token_ids) if out.prompt_token_ids is not None else len(self.tokenizer.encode(prompt_str))
        return out.outputs[0].text.strip(), int(n_input_tokens)

    def sample_answer_candidates(
        self,
        question: str,
        passages: Optional[Sequence[Passage]] = None,
        n_sample: int = 5,
        max_new_tokens: int = 100,
    ) -> List[str]:
        from vllm import SamplingParams
        prompt = self._build_dinco_answer_prompt(question=question, passages=passages)
        prompt_str = self._build_chat_prompt_string([{"role": "user", "content": prompt}])
        params = SamplingParams(
            n=int(max(1, n_sample)),
            temperature=1.0,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )
        out = self.llm.generate([prompt_str], params, use_tqdm=False)[0]
        return [self.clean_answer_for_dinco(o.text) for o in out.outputs]

    def batch_yes_probability(
        self,
        question: str,
        candidates: Sequence[str],
        passages: Optional[Sequence[Passage]] = None,
    ) -> List[float]:
        from vllm import SamplingParams
        evidence = format_evidence(passages) if passages else ""
        prompt_strs: List[str] = []
        for cand in candidates:
            if passages:
                content = PTRUE_WITH_EVIDENCE_PROMPT.format(
                    question=question, evidence=evidence, candidate_answer=cand,
                )
            else:
                content = PTRUE_PROMPT.format(question=question, candidate_answer=cand)
            prompt_strs.append(
                self._build_chat_prompt_string([{"role": "user", "content": content}])
            )
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outs = self.llm.generate(prompt_strs, params, use_tqdm=False)
        results: List[float] = []
        for out in outs:
            seq = out.outputs[0]
            if not seq.logprobs:
                # Fallback: 0.5 ≈ "validator failure"; cleaner than NaN since
                # the surrounding pipeline expects a probability in [0,1].
                results.append(0.5)
                continue
            top = seq.logprobs[0]
            p_yes = 0.0
            p_no = 0.0
            for tid, lp in top.items():
                lp_val = lp.logprob if hasattr(lp, "logprob") else float(lp)
                if tid in self.yes_token_set:
                    p_yes += math.exp(lp_val)
                elif tid in self.no_token_set:
                    p_no += math.exp(lp_val)
            total = p_yes + p_no
            if total < 0.5:
                results.append(0.5)
            else:
                results.append(p_yes / total)
        return results

    def beam_search_answer_candidates(
        self,
        question: str,
        passages: Optional[Sequence[Passage]] = None,
        num_beams: int = 5,
        length_penalty: float = 0.0,
        max_new_tokens: int = 100,
    ) -> Tuple[List[str], List[float]]:
        from vllm.sampling_params import BeamSearchParams
        prompt = self._build_dinco_answer_prompt(question=question, passages=passages)
        prompt_str = self._build_chat_prompt_string([{"role": "user", "content": prompt}])
        params = BeamSearchParams(
            beam_width=int(max(1, num_beams)),
            max_tokens=max_new_tokens,
            ignore_eos=False,
            temperature=0.0,
            length_penalty=length_penalty,
        )
        # vLLM 0.10+ expects [{"prompt": str}]; older versions accept either form. Use
        # the dict form for forward compatibility (LS6 vllm 0.11.2 requires it).
        outs = self.llm.beam_search([{"prompt": prompt_str}], params=params)
        out = outs[0]
        beam_strs: List[str] = []
        beam_scores: List[float] = []
        for seq in out.sequences:
            full_text = seq.text or ""
            # vLLM beam_search returns prompt + completion; strip prefix.
            if full_text.startswith(prompt_str):
                new_text = full_text[len(prompt_str):]
            else:
                marker = "<|im_start|>assistant"
                if marker in full_text:
                    new_text = full_text.rsplit(marker, 1)[-1].lstrip("\n")
                else:
                    new_text = full_text
            # Strip <|im_end|> and anything past it (canary lesson).
            new_text = new_text.split("<|im_end|>")[0]
            beam_strs.append(new_text)
            cum: Optional[float] = None
            for attr in ("cum_logprob", "cumulative_logprob"):
                if hasattr(seq, attr):
                    v = getattr(seq, attr)
                    if v is not None:
                        cum = float(v)
                        break
            beam_scores.append(cum if cum is not None else 0.0)
        return self.lexical_clean_candidates(beam_strs, beam_scores)


class MockQwenModel:
    def generate_answer_and_claims(
        self,
        question: str,
        passages: Sequence[Passage],
        refinement: bool = False,
        previous_answer: str = "",
        previous_claims: Optional[Sequence[str]] = None,
    ) -> GenerationOutput:
        title_a = passages[0].title if passages else "Unknown"
        title_b = passages[1].title if len(passages) > 1 else title_a
        answer = title_b if not refinement else title_a
        claims = [
            f"{title_a} is evidence related to the question {question}.",
            f"{title_a} connects with {title_b} for a multi-hop answer.",
        ]
        return GenerationOutput(
            answer=answer,
            support_claims=claims,
            answer_support_claims=list(claims),
            raw_text="mock",
            structured={"answer": answer, "support_claims": claims, "answer_support_claims": claims},
        )

    def decide_policy_action(
        self,
        original_question: str,
        effective_question: str,
        passages: Sequence[Passage],
        answer: str,
        support_claims: Sequence[str],
        answer_support_claims: Sequence[str],
        grounding_claims: Sequence[str],
        nvc: float,
        grounding_g: Optional[float],
        claim_grounding_scores: Sequence[float],
        hop: int,
        max_hops: int,
        nvc_low: float = 0.70,
        nvc_high: float = 0.80,
        grounding_low: float = 0.70,
        support_low: float = 0.50,
        support_high: float = 0.70,
        min_critical_supported: int = 1,
        max_evidence_chars: int = 5000,
    ) -> PolicyDecision:
        del (
            original_question,
            passages,
            support_claims,
            answer_support_claims,
            grounding_claims,
            claim_grounding_scores,
            hop,
            max_hops,
            nvc_low,
            nvc_high,
            grounding_low,
            support_low,
            support_high,
            min_critical_supported,
            max_evidence_chars,
        )
        expected_type = QwenDincoModel._infer_expected_answer_type(effective_question)
        predicted_type = QwenDincoModel._infer_answer_type(answer)
        diagnostics = {
            "expected_answer_type": expected_type,
            "predicted_answer_type": predicted_type,
            "answer_type_match": QwenDincoModel._is_type_compatible(expected_type, predicted_type),
        }
        if nvc >= 0.75 and (grounding_g is None or grounding_g >= 0.70) and diagnostics["answer_type_match"]:
            return PolicyDecision(
                action="commit",
                reason="mock high confidence",
                raw_text="mock",
                diagnostics=diagnostics,
            )
        return PolicyDecision(
            action="retrieve",
            reason="mock needs more evidence",
            raw_text="mock",
            diagnostics=diagnostics,
        )


class MockDincoCalibrator:
    def compute(
        self,
        question: str,
        answer: str,
        passages: Sequence[Passage],
        n_distractors: int,
    ) -> DincoResult:
        del question, answer, n_distractors
        nvc = min(0.45 + 0.15 * len(passages), 0.95)
        sc_conf = min(0.55 + 0.10 * len(passages), 0.95)
        return DincoResult(
            nvc=float(nvc),
            candidates=["mock_answer", "mock_d1", "mock_d2"],
            ptrues=[float(nvc), 0.3, 0.2],
            nli=[[[1.0, 0.0, 0.0]]],
            sc_conf=float(sc_conf),
            final_conf=float((nvc + sc_conf) / 2.0),
            sampled_generations=["mock_answer", "mock_alt"],
            sc_entailments=[1.0, 0.6],
        )

    def question_conditioned_bidirectional_entailment(
        self,
        question: str,
        pred_answer: str,
        gold_answer: str,
        passages: Optional[Sequence[Passage]] = None,
    ) -> float:
        del question, passages
        return float(exact_match(pred_answer, gold_answer))


class MockMiniCheckGrounder:
    def score(self, passages: Sequence[Passage], claims: Sequence[str]) -> Tuple[float, List[float]]:
        if not claims:
            return 0.0, []
        g = min(0.25 + 0.12 * len(passages), 0.92)
        claim_scores = [float(max(0.01, min(0.99, g - 0.05 + 0.01 * i))) for i, _ in enumerate(claims)]
        return float(g), claim_scores


# ---------------------------------------------------------------------------
# Single-signal ablation stand-ins.
# Used by agent_telemetry_mode in {"dinco_only", "minicheck_only"} so the
# orchestrator's downstream code keeps working without DINCO/MiniCheck
# instantiation overhead OR pollution of the agent prompt (the prompt
# template surgery in run_agent_gated_retrieval_hotpotqa.py drops the
# disabled signal's section entirely; these stand-ins exist to keep the
# .compute() / .score() return-shape contract without burning compute).
# Sentinels: dinco final_conf/nvc/sc_conf = -1.0 (never threshold-compared
# downstream in the disabled mode); g_mean = None; claim_scores = [].
# ---------------------------------------------------------------------------

class DisabledDincoCalibrator:
    """Cheap stand-in for `DincoCalibrator` when DINCO is ablated.

    Returns a sentinel `DincoResult` with empty candidate / ptrues / nli /
    sc_entailments lists so any downstream `.attr` dereference (e.g.\
    `result.candidates[0]`) lands on a well-formed empty list instead of
    crashing. The `final_conf = -1.0` sentinel is never threshold-compared
    in disabled mode because the agent prompt no longer surfaces DINCO.
    """

    def compute(
        self,
        question: str,
        answer: str,
        passages: Sequence[Passage],
        n_distractors: int,
    ) -> DincoResult:
        del question, answer, passages, n_distractors
        return DincoResult(
            nvc=-1.0,
            candidates=[],
            ptrues=[],
            nli=[],
            sc_conf=-1.0,
            final_conf=-1.0,
            sampled_generations=[],
            sc_entailments=[],
        )

    def question_conditioned_bidirectional_entailment(
        self,
        question: str,
        pred_answer: str,
        gold_answer: str,
        passages: Optional[Sequence[Passage]] = None,
    ) -> float:
        del question, passages
        return float(exact_match(pred_answer, gold_answer))


class DisabledMiniCheckGrounder:
    """Cheap stand-in for `MiniCheckGrounder` when MiniCheck is ablated.

    Returns `(None, [])` for any (passages, claims) input. Downstream code
    that gates on `g is not None` already short-circuits when the grounder
    is unavailable (see `multihop_dinco_minicheck_hotpotqa.py` line ~2953
    and runner `support.get("g_mean", 0.0)` defaults), so this is a no-op
    from the agent's perspective once the prompt template surgery removes
    the grounding section.
    """

    def score(self, passages: Sequence[Passage], claims: Sequence[str]) -> Tuple[Optional[float], List[float]]:
        del passages, claims
        return None, []


class MultiHopController:
    def __init__(
        self,
        qwen_model: Any,
        dinco: Any,
        grounder: Any,
        nvc_threshold: float,
        grounding_threshold: float,
        combined_threshold: float,
        prior_weight: float,
        n_distractors: int,
        retrieval_order_mode: str = "sequential",
        policy_mode: str = "threshold",
        policy_max_evidence_chars: int = 5000,
    ) -> None:
        self.qwen_model = qwen_model
        self.dinco = dinco
        self.grounder = grounder
        self.nvc_threshold = nvc_threshold
        self.grounding_threshold = grounding_threshold
        self.combined_threshold = combined_threshold
        self.prior_weight = prior_weight
        self.n_distractors = n_distractors
        self.retrieval_order_mode = retrieval_order_mode
        self.policy_mode = policy_mode
        self.policy_max_evidence_chars = policy_max_evidence_chars

    def _combined_prior(self, nvc: float, g: float) -> float:
        return float(self.prior_weight * nvc + (1.0 - self.prior_weight) * g)

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        return bool(re.match(r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b", question, re.IGNORECASE))

    @staticmethod
    def _map_text_to_yes_no(text: str) -> Optional[str]:
        t = f" {normalize_answer(text)} "
        if not t.strip():
            return None
        if " yes " in t:
            return "yes"
        if " no " in t:
            return "no"

        negative_cues = [
            " not ",
            " never ",
            " neither ",
            " different ",
            " not same ",
            " not the same ",
            " not located in the same ",
            " not located in same ",
        ]
        positive_cues = [
            " both ",
            " same ",
            " equally ",
            " identical ",
        ]
        if any(cue in t for cue in negative_cues):
            return "no"
        if any(cue in t for cue in positive_cues):
            return "yes"
        return None

    @staticmethod
    def _clean_phrase_span(text: str, max_words: int = 12) -> str:
        span = re.sub(r"\s+", " ", (text or "").strip())
        span = span.strip(" \t\n\r\"'`.,;:!?")
        span = re.sub(r"^(?:in|at|from|of|to|the)\s+", "", span, flags=re.IGNORECASE)
        span = re.split(r"(?:\s+which\s+|\s+that\s+|\s+where\s+)", span, maxsplit=1, flags=re.IGNORECASE)[0]
        span = span.strip(" \t\n\r\"'`.,;:!?")
        if not span:
            return ""
        words = span.split()
        if len(words) > max_words:
            return ""
        if re.search(r"\b(is|was|were|are|has|have|had|held|served|located|born|based)\b", span, flags=re.IGNORECASE):
            return ""
        return span

    def _prefer_specific_grounded_span(
        self,
        question: str,
        base_answer: str,
        grounding_claims: Sequence[str],
        claim_grounding_scores: Sequence[float],
        selected_passages: Sequence[Passage],
    ) -> Optional[str]:
        """
        Expand a broad grounded answer to a more specific grounded span when available.
        Example: "New York City" -> "Greenwich Village, New York City".
        """
        base = re.sub(r"\s+", " ", (base_answer or "").strip())
        base = base.strip(" \t\n\r\"'`.,;:!?")
        base_norm = normalize_answer(base)
        if not base_norm:
            return None

        expected = QwenDincoModel._infer_expected_answer_type(question)
        if expected not in {"location", "entity"}:
            return None

        units: List[Tuple[str, float]] = []
        for i, claim in enumerate(grounding_claims):
            score = float(claim_grounding_scores[i]) if i < len(claim_grounding_scores) else 0.0
            if score >= 0.50:
                units.append((claim, score))

        # If no grounded claims pass threshold, use evidence as weak fallback.
        if not units:
            for p in selected_passages:
                txt = re.sub(r"\s+", " ", p.text).strip()
                if not txt:
                    continue
                for sent in re.split(r"(?<=[.!?])\s+", txt):
                    if sent:
                        units.append((sent, 0.35))

        best: Optional[str] = None
        best_score = -1e9
        base_wc = len(base_norm.split())

        for text, support_prob in units:
            norm_text = normalize_answer(text)
            if not norm_text or base_norm not in norm_text:
                continue

            # Candidate 1: local comma phrase ending with base answer.
            low = text.lower()
            idx = low.find(base.lower())
            candidates: List[str] = []
            if idx >= 0:
                left = text[:idx].rstrip()
                right = text[idx : idx + len(base)].strip()
                if "," in left:
                    left_head = left[: left.rfind(",")].rstrip()
                    m = re.search(r"([A-Z][A-Za-z0-9'’`.\-]*(?:\s+[A-Z][A-Za-z0-9'’`.\-]*){0,6})\s*$", left_head)
                    if m:
                        candidates.append(f"{m.group(1)}, {right}")

            # Candidate 2: sentence-level phrase before punctuation around base.
            for chunk in re.split(r"[.;:!?]", text):
                if base_norm in normalize_answer(chunk):
                    candidates.append(chunk)

            for cand in candidates:
                phrase = self._clean_phrase_span(cand, max_words=10)
                if not phrase:
                    continue
                norm_phrase = normalize_answer(phrase)
                if not norm_phrase or base_norm not in norm_phrase:
                    continue

                wc = len(norm_phrase.split())
                if wc < base_wc or wc > 10:
                    continue

                # Prefer specific supersets with comma structure.
                specificity_bonus = 0.05 * (wc - base_wc)
                if "," in phrase:
                    specificity_bonus += 0.20
                if expected == "location" and "," in phrase:
                    specificity_bonus += 0.10

                score = float(support_prob) + specificity_bonus
                if norm_phrase == base_norm:
                    score -= 0.25
                if score > best_score:
                    best_score = score
                    best = phrase

        if best and normalize_answer(best) != base_norm:
            return best
        return None

    def _shorten_final_answer(
        self,
        question: str,
        answer: str,
        grounding_claims: Optional[Sequence[str]] = None,
        claim_grounding_scores: Optional[Sequence[float]] = None,
        selected_passages: Optional[Sequence[Passage]] = None,
    ) -> str:
        """
        Keep retrieval/reasoning behavior unchanged and only canonicalize final output form.
        """
        text = re.sub(r"\s+", " ", (answer or "").strip())
        text = text.strip(" \t\n\r\"'`")
        text = re.sub(r"^(?:answer\s*[:\-]\s*)", "", text, flags=re.IGNORECASE)
        if not text:
            return "insufficient evidence"

        # Force strict yes/no outputs for yes/no questions.
        if self._is_yes_no_question(question):
            mapped = self._map_text_to_yes_no(text)
            if mapped is not None:
                return mapped

        # Common long-form pattern from claim-based generations:
        # "The Animorphs series ... " -> "Animorphs"
        m = re.match(r"^\s*(?:the\s+)?(.+?)\s+series\b", text, flags=re.IGNORECASE)
        if m:
            series_name = m.group(1).strip(" \t\n\r\"'`.,;:!?")
            if series_name:
                text = series_name

        # QA role/position pattern:
        # "... held the position of X" -> "X"
        m = re.search(r"\bheld the position of\s+([^.,;]+)", text, flags=re.IGNORECASE)
        if m:
            role = m.group(1).strip(" \t\n\r\"'`.,;:!?")
            if role:
                text = role

        # Trim to first clause and remove boilerplate lead-ins.
        text = re.sub(r"^(?:the answer is|it is|it's|this is|that is)\s+", "", text, flags=re.IGNORECASE)
        clause = re.split(r"(?:[.;]|\s+because\s+|\s+while\s+|\s+although\s+|\s+but\s+)", text, maxsplit=1, flags=re.IGNORECASE)[0]
        clause = clause.strip(" \t\n\r\"'`.,;:!?")
        if not clause:
            clause = text.strip(" \t\n\r\"'`.,;:!?")

        # If still sentence-like, keep phrase before the first linking verb.
        m = re.match(
            r"^(.+?)\s+\b(is|was|were|are|has|have|had|held|served|located|born|based)\b",
            clause,
            flags=re.IGNORECASE,
        )
        if m:
            head = m.group(1).strip(" \t\n\r\"'`.,;:!?")
            if head:
                clause = head

        # Keep concise phrase length.
        words = clause.split()
        if len(words) > 8:
            clause = " ".join(words[:8])

        clause = clause if clause else "insufficient evidence"

        # Prefer a more specific grounded span (claims/evidence) over broad parent entity.
        if grounding_claims and selected_passages is not None:
            specific = self._prefer_specific_grounded_span(
                question=question,
                base_answer=clause,
                grounding_claims=list(grounding_claims),
                claim_grounding_scores=list(claim_grounding_scores or []),
                selected_passages=list(selected_passages),
            )
            if specific:
                return specific

        return clause

    def _question_conditioned_bidirectional_entailment(
        self,
        question: str,
        pred_answer: str,
        gold_answer: str,
        selected_passages: Sequence[Passage],
    ) -> float:
        if hasattr(self.dinco, "question_conditioned_bidirectional_entailment"):
            try:
                score = self.dinco.question_conditioned_bidirectional_entailment(
                    question=question,
                    pred_answer=pred_answer,
                    gold_answer=gold_answer,
                    passages=selected_passages,
                )
                return float(score)
            except Exception:
                pass
        return float(exact_match(pred_answer, gold_answer))

    def run_example(self, example: Dict[str, Any], max_passages: int) -> Dict[str, Any]:
        question = example["question"]
        current_question = question
        gold_answer = example["answer"]
        passages = build_passages(example)
        if self.retrieval_order_mode == "tfidf":
            ranking, ranking_scores = rank_passages(question=question, passages=passages)
        else:
            ranking = list(range(len(passages)))
            ranking_scores = [0.0] * len(passages)
        ranking = ranking[: max_passages if max_passages > 0 else len(ranking)]

        best_state: Optional[CandidateState] = None
        final_state: Optional[CandidateState] = None
        iteration_logs: List[Dict[str, Any]] = []
        selected: List[int] = []

        for hop_idx, passage_idx in enumerate(ranking, start=1):
            selected.append(passage_idx)
            selected_passages = [passages[i] for i in selected]

            gen = self.qwen_model.generate_answer_and_claims(question=current_question, passages=selected_passages)
            claims_generation_answer = gen.answer
            dinco_result = self.dinco.compute(
                question=current_question,
                answer=claims_generation_answer,
                passages=selected_passages,
                n_distractors=self.n_distractors,
            )
            hop_answer = dinco_result.candidates[0] if dinco_result.candidates else gen.answer
            raw_nvc = float(dinco_result.nvc)
            sc_conf = float(dinco_result.sc_conf)
            nvc = float(dinco_result.final_conf)
            answer_support_claims = gen.answer_support_claims
            grounding_claims = gen.support_claims if gen.support_claims else answer_support_claims

            g: Optional[float] = None
            claim_scores: List[float] = []
            is_final_hop = hop_idx >= len(ranking)
            if nvc > self.nvc_threshold or is_final_hop:
                g, claim_scores = self.grounder.score(passages=selected_passages, claims=grounding_claims)

            combined = self._combined_prior(nvc=nvc, g=g) if g is not None else nvc

            log_rec: Dict[str, Any] = {
                "hop": hop_idx,
                "selected_indices": list(selected),
                "selected_titles": [passages[i].title for i in selected],
                "effective_question": current_question,
                # Keep DINCO top candidate as the hop answer (short QA style), mirroring original DINCO setup.
                "generation_answer": hop_answer,
                # Preserve the answer from the claims-generation prompt for multi-hop debugging.
                "claims_generation_answer": claims_generation_answer,
                "answer": hop_answer,
                "support_claims": gen.support_claims,
                "answer_support_claims": answer_support_claims,
                "grounding_claims": grounding_claims,
                "nvc": nvc,
                "raw_nvc": raw_nvc,
                "sc_conf": sc_conf,
                "dinco_final_conf": nvc,
                "dinco_candidates": dinco_result.candidates,
                "dinco_ptrues": dinco_result.ptrues,
                "dinco_sc_entailments": dinco_result.sc_entailments,
                "grounding_ran": bool(g is not None),
                "grounding_g": g,
                "claim_grounding_scores": claim_scores,
                "combined_prior": combined,
                "policy_mode": self.policy_mode,
                "policy_action": None,
                "policy_reason": None,
                "policy_diagnostics": {},
                "policy_guard_triggers": [],
                "action": None,
                "refined": None,
            }

            candidate = CandidateState(
                answer=hop_answer,
                support_claims=gen.support_claims,
                grounding_claims=grounding_claims,
                claim_grounding_scores=list(claim_scores),
                selected_indices=list(selected),
                selected_titles=[passages[i].title for i in selected],
                nvc=nvc,
                g=float(g if g is not None else 0.0),
                combined_prior=combined,
                hop=hop_idx,
                refined=False,
            )
            if best_state is None or candidate.combined_prior > best_state.combined_prior:
                best_state = candidate

            if self.policy_mode == "llm":
                if not hasattr(self.qwen_model, "decide_policy_action"):
                    raise RuntimeError("policy_mode=llm requires qwen_model.decide_policy_action(...) support.")

                # Enforced policy rule: low NVC must retrieve next passage.
                has_next_hop = hop_idx < len(ranking)
                if nvc <= self.nvc_threshold and has_next_hop:
                    log_rec["policy_action"] = "retrieve"
                    log_rec["policy_reason"] = (
                        f"forced_retrieve_low_nvc: nvc={nvc:.3f} <= threshold={self.nvc_threshold:.3f}"
                    )
                    log_rec["policy_diagnostics"] = {}
                    log_rec["policy_guard_triggers"] = [f"nvc_low:{nvc:.3f}"]
                    log_rec["action"] = "policy_retrieve_low_nvc"
                    iteration_logs.append(log_rec)
                    continue

                decision = self.qwen_model.decide_policy_action(
                    original_question=question,
                    effective_question=current_question,
                    passages=selected_passages,
                    answer=hop_answer,
                    support_claims=gen.support_claims,
                    answer_support_claims=answer_support_claims,
                    grounding_claims=grounding_claims,
                    nvc=nvc,
                    grounding_g=g,
                    claim_grounding_scores=claim_scores,
                    hop=hop_idx - 1,
                    max_hops=max(0, len(ranking) - 1),
                    nvc_low=self.nvc_threshold,
                    nvc_high=max(self.nvc_threshold, 0.80),
                    grounding_low=self.grounding_threshold,
                    support_low=0.50,
                    support_high=0.70,
                    min_critical_supported=1,
                    max_evidence_chars=self.policy_max_evidence_chars,
                )
                log_rec["policy_action"] = decision.action
                log_rec["policy_reason"] = decision.reason
                log_rec["policy_diagnostics"] = decision.diagnostics
                log_rec["policy_guard_triggers"] = decision.guard_triggers

                if decision.action == "commit":
                    log_rec["action"] = "policy_commit"
                    iteration_logs.append(log_rec)
                    final_state = candidate
                    break

                log_rec["action"] = "policy_retrieve"
                iteration_logs.append(log_rec)
                continue

            # Threshold policy (legacy behavior).
            if nvc <= self.nvc_threshold:
                log_rec["action"] = "retrieve_more_low_nvc"
                iteration_logs.append(log_rec)
                continue

            # Required gating: only run MiniCheck when NVC > threshold.
            if g is not None and g < self.grounding_threshold:
                log_rec["action"] = "retrieve_more_low_grounding"
                iteration_logs.append(log_rec)
                continue

            # Strong grounding + confidence.
            if combined >= self.combined_threshold:
                log_rec["action"] = "stop_accept"
                iteration_logs.append(log_rec)
                final_state = candidate
                break

            log_rec["action"] = "retrieve_more_low_combined"
            iteration_logs.append(log_rec)

        if final_state is None:
            final_state = best_state

        if final_state is None:
            # Extremely defensive fallback.
            final_state = CandidateState(
                answer="insufficient evidence",
                support_claims=[],
                grounding_claims=[],
                claim_grounding_scores=[],
                selected_indices=[],
                selected_titles=[],
                nvc=0.0,
                g=0.0,
                combined_prior=0.0,
                hop=0,
                refined=False,
            )

        supporting_titles = set(example["supporting_facts"]["title"])
        selected_title_set = set(final_state.selected_titles)
        supporting_title_recall = (
            len(selected_title_set.intersection(supporting_titles)) / max(1, len(supporting_titles))
        )
        final_selected_passages = [passages[i] for i in final_state.selected_indices]
        pred_answer_raw = final_state.answer
        pred_answer = self._shorten_final_answer(
            question=question,
            answer=pred_answer_raw,
            grounding_claims=final_state.grounding_claims,
            claim_grounding_scores=final_state.claim_grounding_scores,
            selected_passages=final_selected_passages,
        )
        qc_bidirectional_entailment = self._question_conditioned_bidirectional_entailment(
            question=question,
            pred_answer=pred_answer,
            gold_answer=gold_answer,
            selected_passages=final_selected_passages,
        )

        em = exact_match(pred_answer, gold_answer)
        return {
            "id": example["id"],
            "question": question,
            "effective_question_final": current_question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "pred_answer_raw": pred_answer_raw,
            "pred_support_claims": final_state.support_claims,
            "pred_grounding_claims": final_state.grounding_claims,
            "grounder_model_name": getattr(self.grounder, "model_name", "unknown"),
            "em": em,
            "nvc": final_state.nvc,
            "grounding_g": final_state.g,
            "combined_prior": final_state.combined_prior,
            "qc_bidirectional_entailment": qc_bidirectional_entailment,
            "refined": final_state.refined,
            "hops_used": final_state.hop,
            "selected_passage_indices": final_state.selected_indices,
            "selected_titles": final_state.selected_titles,
            "supporting_title_recall": supporting_title_recall,
            "retrieval_order_mode": self.retrieval_order_mode,
            "policy_mode": self.policy_mode,
            "retrieval_order": ranking,
            "retrieval_scores": ranking_scores,
            "iterations": iteration_logs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-hop DINCO+MiniCheck retrieval controller on HotpotQA")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument("--dataset_subset", type=str, default="distractor")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument(
        "--example_id",
        type=str,
        default=None,
        help="Run a specific dataset example by id (overrides --start_idx/--max_examples).",
    )
    parser.add_argument("--max_passages", type=int, default=6)

    parser.add_argument("--nvc_threshold", type=float, default=0.70)
    parser.add_argument("--grounding_threshold", type=float, default=0.70)
    parser.add_argument("--combined_threshold", type=float, default=0.65)
    parser.add_argument("--prior_weight", type=float, default=0.60)
    parser.add_argument("--n_distractors", type=int, default=5)
    parser.add_argument(
        "--retrieval_order_mode",
        type=str,
        default="sequential",
        choices=["sequential", "tfidf"],
        help="Passage order policy: sequential simulates web-search append; tfidf enables ranking.",
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="threshold",
        choices=["threshold", "llm"],
        help="Controller mode: threshold uses fixed gates; llm asks the policy model for commit/retrieve.",
    )
    parser.add_argument(
        "--policy_max_evidence_chars",
        type=int,
        default=5000,
        help="Max evidence chars passed to the LLM policy state.",
    )

    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--qwen_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--minicheck_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--minicheck_max_model_len", type=int, default=None)
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument(
        "--minicheck_model_name",
        type=str,
        default="Bespoke-MiniCheck-7B",
        choices=["Bespoke-MiniCheck-7B", "flan-t5-large", "deberta-v3-large", "roberta-large"],
        help="MiniCheck backend model. Bespoke-MiniCheck-7B requires CUDA + vLLM.",
    )
    parser.add_argument(
        "--allow_minicheck_cpu_fallback",
        action="store_true",
        help="If Bespoke-MiniCheck-7B init fails (or CUDA missing), fall back to a smaller MiniCheck model.",
    )
    parser.add_argument(
        "--minicheck_cpu_fallback_model_name",
        type=str,
        default="roberta-large",
        choices=["roberta-large", "deberta-v3-large", "flan-t5-large"],
        help="Fallback MiniCheck model used when --allow_minicheck_cpu_fallback is set.",
    )
    parser.add_argument(
        "--minicheck_gpu_memory_utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization for MiniCheck-7B. Lower if startup fails due memory headroom.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL filename (relative paths are written under results).",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Summary JSON filename (relative paths are written under results).",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry_run", action="store_true", help="Run pipeline logic with mock models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_path = resolve_json_output_path(args.output_jsonl)
    summary_path = (
        resolve_json_output_path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_name, args.dataset_subset, split=args.split)
    selected_start_idx: Optional[int] = None
    selected_end_idx: Optional[int] = None
    if args.example_id:
        if "id" not in ds.column_names:
            raise ValueError("Dataset does not contain an 'id' column; cannot use --example_id.")
        ids = ds["id"]
        match_indices = [i for i, ex_id in enumerate(ids) if str(ex_id) == str(args.example_id)]
        if not match_indices:
            raise ValueError(
                f"example_id='{args.example_id}' not found in {args.dataset_name}/{args.dataset_subset} [{args.split}]"
            )
        selected_start_idx = int(match_indices[0])
        selected_end_idx = selected_start_idx + 1
        ds = ds.select([selected_start_idx])
    else:
        selected_start_idx = args.start_idx
        selected_end_idx = min(args.start_idx + args.max_examples, len(ds))
        if selected_start_idx >= selected_end_idx:
            raise ValueError(f"Empty slice requested: start_idx={selected_start_idx}, end_idx={selected_end_idx}")
        ds = ds.select(range(selected_start_idx, selected_end_idx))

    if args.dry_run:
        qwen_model = MockQwenModel()
        dinco = MockDincoCalibrator()
        grounder = MockMiniCheckGrounder()
    else:
        qwen_model = QwenDincoModel(
            model_name=args.qwen_model_name,
            cache_dir=args.cache_dir,
            dtype=args.qwen_dtype,
        )
        dinco = DincoCalibrator(qwen_model=qwen_model, cache_dir=args.cache_dir)
        grounder = MiniCheckGrounder(
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.minicheck_tensor_parallel_size,
            max_model_len=args.minicheck_max_model_len,
            enable_prefix_caching=not args.disable_prefix_caching,
            model_name=args.minicheck_model_name,
            allow_cpu_fallback=args.allow_minicheck_cpu_fallback,
            cpu_fallback_model_name=args.minicheck_cpu_fallback_model_name,
            gpu_memory_utilization=args.minicheck_gpu_memory_utilization,
        )

    controller = MultiHopController(
        qwen_model=qwen_model,
        dinco=dinco,
        grounder=grounder,
        nvc_threshold=args.nvc_threshold,
        grounding_threshold=args.grounding_threshold,
        combined_threshold=args.combined_threshold,
        prior_weight=args.prior_weight,
        n_distractors=args.n_distractors,
        retrieval_order_mode=args.retrieval_order_mode,
        policy_mode=args.policy_mode,
        policy_max_evidence_chars=args.policy_max_evidence_chars,
    )

    metrics = {
        "count": 0,
        "em_sum": 0,
        "mean_nvc": [],
        "mean_g": [],
        "mean_combined": [],
        "mean_qc_bidirectional_entailment": [],
        "mean_hops": [],
        "mean_supporting_title_recall": [],
    }

    with output_path.open("w", encoding="utf-8") as writer:
        for ex in tqdm(ds, desc="Running multi-hop DINCO+MiniCheck"):
            rec = controller.run_example(ex, max_passages=args.max_passages)
            writer.write(json.dumps(rec, ensure_ascii=True) + "\n")
            writer.flush()

            metrics["count"] += 1
            metrics["em_sum"] += rec["em"]
            metrics["mean_nvc"].append(rec["nvc"])
            metrics["mean_g"].append(rec["grounding_g"])
            metrics["mean_combined"].append(rec["combined_prior"])
            metrics["mean_qc_bidirectional_entailment"].append(rec["qc_bidirectional_entailment"])
            metrics["mean_hops"].append(rec["hops_used"])
            metrics["mean_supporting_title_recall"].append(rec["supporting_title_recall"])

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "split": args.split,
        "start_idx": selected_start_idx,
        "end_idx": selected_end_idx,
        "example_id": args.example_id,
        "count": metrics["count"],
        "em": metrics["em_sum"] / max(1, metrics["count"]),
        "avg_nvc": float(np.mean(metrics["mean_nvc"])) if metrics["mean_nvc"] else math.nan,
        "avg_grounding_g": float(np.mean(metrics["mean_g"])) if metrics["mean_g"] else math.nan,
        "avg_combined_prior": float(np.mean(metrics["mean_combined"])) if metrics["mean_combined"] else math.nan,
        "avg_qc_bidirectional_entailment": (
            float(np.mean(metrics["mean_qc_bidirectional_entailment"]))
            if metrics["mean_qc_bidirectional_entailment"]
            else math.nan
        ),
        "avg_hops_used": float(np.mean(metrics["mean_hops"])) if metrics["mean_hops"] else math.nan,
        "avg_supporting_title_recall": (
            float(np.mean(metrics["mean_supporting_title_recall"]))
            if metrics["mean_supporting_title_recall"]
            else math.nan
        ),
        "config": vars(args),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
