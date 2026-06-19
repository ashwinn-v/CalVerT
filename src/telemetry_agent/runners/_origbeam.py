#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    BASE_DIR,
    normalize_limit,
    slugify_name,
    write_json,
)
from retrieval_index import BM25Index

LLMRECOURSE_DIR = Path(__file__).resolve().parents[1]
if str(LLMRECOURSE_DIR) not in sys.path:
    sys.path.insert(0, str(LLMRECOURSE_DIR))

DINCO_DIR = LLMRECOURSE_DIR / "dinco"
if str(DINCO_DIR) not in sys.path:
    sys.path.insert(0, str(DINCO_DIR))

from telemetry_agent.dinco import triviaqa as dinco_base  # noqa: E402
from telemetry_agent.runners import _hotpot_utils as hotpot_utils  # noqa: E402
from telemetry_agent.planner import qwen_planner as planner_utils  # noqa: E402
from telemetry_agent.runners import _base_runner as base_runner  # noqa: E402


DEFAULT_DINCO_NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class OriginalClosedBookBeamDincoCalibrator:
    """Closed-book DInCo that follows the original TriviaQA beam-search formulation."""

    def __init__(
        self,
        *,
        qwen_model: Any,
        cache_dir: Optional[str] = None,
        n_sc_samples: int = 5,
        sc_match_threshold: float = 0.90,
        nli_model_name: str = DEFAULT_DINCO_NLI_MODEL_NAME,
        beam_max_new_tokens: int = 100,
        beam_length_penalty: float = 0.0,
        dinco_backend: Any = dinco_base,
        nli_tokenizer: Optional[Any] = None,
        nli_model: Optional[Any] = None,
    ) -> None:
        hotpot_utils.disable_broken_torchvision_for_transformers()

        self.qwen_model = qwen_model
        self.model = getattr(qwen_model, "model", None)
        self.tokenizer = getattr(qwen_model, "tokenizer", None)
        # `self.model` is None when ``qwen_model`` is a vLLM-backed
        # ``QwenVLLMDincoModel`` — in that case ``compute()`` routes through
        # the polymorphic ``qwen_model.*`` methods instead of HF ``model.generate``.
        # Tokenizer is still required for the NLI prompt construction in `_run_nli`.
        if self.tokenizer is None:
            raise ValueError(
                "OriginalClosedBookBeamDincoCalibrator requires a qwen_model with a "
                "loaded tokenizer (HF or vLLM). Got tokenizer=None."
            )

        self.n_sc_samples = max(1, int(n_sc_samples))
        self.sc_match_threshold = float(sc_match_threshold)
        self.nli_model_name = str(nli_model_name)
        self.beam_max_new_tokens = max(1, int(beam_max_new_tokens))
        self.beam_length_penalty = float(beam_length_penalty)
        self.backend = dinco_backend
        self.nli_tokenizer = nli_tokenizer
        self.nli_model = nli_model

        if self.nli_tokenizer is None:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, cache_dir=cache_dir)
        if self.nli_model is None:
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name,
                device_map="auto",
                cache_dir=cache_dir,
            )
        if hasattr(self.nli_model, "eval"):
            self.nli_model.eval()

    @staticmethod
    def _single_example_dataset(question: str) -> List[Dict[str, str]]:
        return [{"question": str(question)}]

    def _nli_device(self) -> torch.device:
        model_device = getattr(self.nli_model, "device", None)
        if model_device is not None:
            return torch.device(model_device)
        try:
            return next(self.nli_model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    def _run_nli(self, ds: Sequence[Dict[str, str]], beam_strs: Sequence[Sequence[str]]) -> torch.Tensor:
        beam_width = max((len(strs) for strs in beam_strs), default=1)
        nlis = -torch.ones(len(ds), beam_width, beam_width, 3)

        for ex_i, ex in enumerate(ds):
            if len(beam_strs[ex_i]) <= 1:
                continue

            premises: List[str] = []
            hypotheses: List[str] = []
            nli_mask = torch.zeros(beam_width, beam_width, dtype=torch.bool)
            for i in range(len(beam_strs[ex_i])):
                premises.extend(
                    [f"Question: {ex['question']}\nAnswer: {beam_strs[ex_i][i]}"] * (len(beam_strs[ex_i]) - 1)
                )
                hypotheses.extend(
                    [f"Answer: {beam_strs[ex_i][j]}" for j in range(len(beam_strs[ex_i])) if j != i]
                )
                nli_mask[i, : len(beam_strs[ex_i])] = True
                nli_mask[i, i] = False

            inputs = self.nli_tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
            ).to(self._nli_device())
            with torch.no_grad():
                probs = torch.softmax(self.nli_model(**inputs).logits, dim=-1).cpu()
            nlis[ex_i][torch.nonzero(nli_mask, as_tuple=True)] = probs

        return nlis

    def _run_sc_nli(
        self,
        ds: Sequence[Dict[str, str]],
        main_strs: Sequence[str],
        sampled_strs: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        entail_i = 0
        n_qst = len(main_strs)
        n_sample = max((len(strs) for strs in sampled_strs), default=1)
        nlis = torch.zeros(n_qst, n_sample)

        for ex_i, ex in enumerate(ds):
            main_str = self.backend.clean_str(main_strs[ex_i])
            premises: List[str] = []
            hypotheses: List[str] = []
            nli_mask = torch.ones(n_sample, dtype=torch.bool)

            for sample_i, sampled_str in enumerate(sampled_strs[ex_i]):
                sampled_str = self.backend.clean_str(sampled_str)
                if main_str == sampled_str:
                    nli_mask[sample_i] = False
                    continue
                ans_pair = [main_str, sampled_str]
                for perm in ((0, 1), (1, 0)):
                    premises.append(f"Question: {ex['question']}\nAnswer: {ans_pair[perm[0]]}")
                    hypotheses.append(f"Answer: {ans_pair[perm[1]]}")

            nlis[ex_i, ~nli_mask] = 1.0
            if not premises:
                continue

            inputs = self.nli_tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
            ).to(self._nli_device())
            with torch.no_grad():
                outputs = torch.softmax(self.nli_model(**inputs).logits, dim=-1)[:, entail_i].cpu()
            nlis[ex_i, nli_mask] = (outputs[0::2] + outputs[1::2]) / 2

        return nlis

    @staticmethod
    def _sampling_norm(s: str) -> str:
        """Normalize a string for sampling-DINCO dedupe + agreement comparison.

        HotpotQA-style: lowercase, strip articles (a/an/the), strip
        punctuation, collapse whitespace. Stronger than the original
        canary norm so lexical variants of the same factoid answer
        ("Einstein" / "Albert Einstein" / "Einstein, Albert") collapse to
        the same form rather than counting as distinct distractors.
        """
        import string as _string
        s = (s or "").split('\n')[0]
        s = s.replace('Answer:', '').strip()
        s = s.lower()
        s = ''.join(ch for ch in s if ch not in _string.punctuation)
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
        passages: Sequence[hotpot_utils.Passage],
        n_samples: int = 10,
    ) -> hotpot_utils.SamplingDincoResult:
        """Sampling-DINCO with graceful-degenerate, conditioned on retrieved passages.

        Differs from `compute()`:
        - Distractors come from stochastic sampling WITH passages (not closed-book beam).
        - When samples collapse onto the greedy answer, NVC degenerates to raw P(True)
          and `degenerate=True` is exposed.

        Validated in the dinco-beam-vs-sampling-hotpotqa canary (N=18 cases beam-DINCO
        would have dropped). Reuses `qwen_model.sample_answer_candidates` and
        `qwen_model.batch_yes_probability` which both accept `passages`, plus this
        class's existing `_run_nli` for the NLI matrix and `dinco_base.get_normalized_verbalized_confidence`
        for the NVC formula.
        """
        # 1. Stochastic samples conditioned on the retrieved passages
        raw_samples = self.qwen_model.sample_answer_candidates(
            question=question,
            passages=passages,
            n_sample=n_samples,
            max_new_tokens=self.beam_max_new_tokens,
        )
        # 2. Lexical clean + dedupe; greedy at index 0
        cleaned = self._dedupe_with_greedy_first(answer, raw_samples)
        if not cleaned:
            fallback = ""
            if hasattr(self.qwen_model, "clean_answer_for_dinco") and answer:
                fallback = self.qwen_model.clean_answer_for_dinco(answer)
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
            ds = self._single_example_dataset(question)
            ptrues_t = torch.tensor([ptrues_list], dtype=torch.float32)  # shape [1, K]
            nlis_t = self._run_nli(ds, [cleaned])  # shape [1, beam_width, beam_width, 3]
            nvcs = self.backend.get_normalized_verbalized_confidence(ptrues_t, nlis_t)
            sampling_dinco_conf = float(nvcs[0].item())
            valid_len = len(cleaned)
            nli_list = nlis_t[0, :valid_len, :valid_len].detach().cpu().tolist()
        else:
            sampling_dinco_conf = raw_verbal_ptrue
            nli_list = []
        return hotpot_utils.SamplingDincoResult(
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
        passages: Sequence[hotpot_utils.Passage],
        n_distractors: int,
    ) -> hotpot_utils.DincoResult:
        """Closed-book DINCO via beam search + self-consistency blend.

        Backend-agnostic: routes through ``qwen_model.beam_search_answer_candidates``,
        ``qwen_model.batch_yes_probability``, and ``qwen_model.sample_answer_candidates``,
        all of which exist on both ``QwenDincoModel`` (HF) and ``QwenVLLMDincoModel``
        (vLLM). Closed-book = called with ``passages=None``.

        DeBERTa NLI stays HF (small model, doesn't compete for vLLM memory).
        """
        del passages  # Closed-book: ignore any retrieved passages.

        ds = self._single_example_dataset(question)
        beam_width = max(1, int(n_distractors))

        # 1. Beam-search distractors (closed-book).
        beam_candidates, _beam_scores = self.qwen_model.beam_search_answer_candidates(
            question=question,
            passages=None,
            num_beams=beam_width,
            length_penalty=self.beam_length_penalty,
            max_new_tokens=self.beam_max_new_tokens,
        )
        candidates: List[str] = list(beam_candidates)
        if not candidates:
            fallback = ""
            if answer and hasattr(self.qwen_model, "clean_answer_for_dinco"):
                fallback = self.qwen_model.clean_answer_for_dinco(answer)
            if not fallback:
                fallback = self.backend.clean_str(str(answer or "")) or "insufficient evidence"
            candidates = [fallback]

        # 2. P(True) per candidate (closed-book — no passages).
        ptrues_list = self.qwen_model.batch_yes_probability(
            question=question,
            candidates=candidates,
            passages=None,
        )

        # 3. Pairwise NLI via DeBERTa (HF, both backends).
        beam_strs_batch = [candidates]
        nlis_t = self._run_nli(ds, beam_strs_batch)  # [1, beam_width, beam_width, 3]

        # 4. NVC formula (pure tensor math).
        ptrues_t = torch.tensor([ptrues_list], dtype=torch.float32)  # [1, K]
        # Pad ptrues to match beam_width if needed (NVC formula expects matched shapes).
        if ptrues_t.shape[1] < nlis_t.shape[1]:
            pad = torch.full((1, nlis_t.shape[1] - ptrues_t.shape[1]), -1.0)
            ptrues_t = torch.cat([ptrues_t, pad], dim=1)
        nvcs = self.backend.get_normalized_verbalized_confidence(ptrues_t, nlis_t)

        # 5. Self-consistency samples (closed-book).
        sampled_strs = self.qwen_model.sample_answer_candidates(
            question=question,
            passages=None,
            n_sample=self.n_sc_samples,
            max_new_tokens=self.beam_max_new_tokens,
        )

        # 6. SC NLI scoring against the main candidate.
        main_strs = [candidates[0]]
        sc_nlis = self._run_sc_nli(ds, main_strs, [sampled_strs])
        sc_augmented = torch.cat((sc_nlis, torch.ones(len(ds), 1)), dim=-1)
        sc_confs = torch.mean((sc_augmented > self.sc_match_threshold).float(), dim=-1)

        # 7. Blend.
        dinco_confs = (nvcs + sc_confs) / 2

        valid_len = len(candidates)
        return hotpot_utils.DincoResult(
            nvc=float(nvcs[0].item()),
            candidates=candidates,
            ptrues=[float(x) for x in ptrues_list],
            nli=nlis_t[0, :valid_len, :valid_len].detach().cpu().tolist(),
            sc_conf=float(sc_confs[0].item()),
            final_conf=float(dinco_confs[0].item()),
            sampled_generations=list(sampled_strs),
            sc_entailments=[float(x) for x in sc_nlis[0, : len(sampled_strs)].detach().cpu().tolist()],
        )


class GPT54ParityQwenPlannerRunner(base_runner.CalibratedPlannerMemoryRunner):
    _DEP_ID_RE = re.compile(r"\bsg\d+\b", flags=re.IGNORECASE)
    _COMPARISON_CUES = (
        "same",
        "different",
        "older",
        "younger",
        "before",
        "after",
        "earlier",
        "later",
        "higher",
        "lower",
        "more",
        "less",
        "both",
        "either",
        "nationality",
        "nationalities",
    )

    @classmethod
    def _contains_dependency_placeholders(cls, text: str) -> bool:
        return bool(cls._DEP_ID_RE.search(str(text or "")))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return hotpot_utils.normalize_answer(str(text or ""))

    @staticmethod
    def _extract_dependency_subject(record: Dict[str, Any]) -> str:
        for field in ("resolved_subquestion", "subquestion"):
            subquestion = str(record.get(field, "")).strip()
            if not subquestion:
                continue
            try:
                entities = hotpot_utils.QwenDincoModel._extract_question_entities(subquestion)
            except Exception:  # noqa: BLE001
                entities = []
            for entity in entities:
                text = str(entity or "").strip()
                if text and not re.fullmatch(r"sg\d+", text, flags=re.IGNORECASE):
                    return text
        return str(record.get("answer", "")).strip()

    @classmethod
    def _question_mentions_dependency_subjects(
        cls,
        question: str,
        dependency_records: Sequence[Dict[str, Any]],
    ) -> bool:
        norm_question = cls._normalize_text(question)
        subjects = [cls._normalize_text(cls._extract_dependency_subject(record)) for record in dependency_records]
        subjects = [subject for subject in subjects if subject]
        return bool(subjects) and all(subject in norm_question for subject in subjects)

    @classmethod
    def _looks_like_comparison_rewrite(cls, node: planner_utils.SubquestionNode, text: str) -> bool:
        if str(getattr(node, "purpose", "")).strip() == "comparison_fact":
            return True
        norm = f" {cls._normalize_text(text)} "
        return any(f" {cue} " in norm for cue in cls._COMPARISON_CUES)

    @staticmethod
    def _clean_rewritten_question(text: str, original_subquestion: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return original_subquestion
        if original_subquestion.endswith("?") and not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"
        return cleaned

    @classmethod
    def _substitute_dependency_placeholders(
        cls,
        text: str,
        dependency_records: Sequence[Dict[str, Any]],
    ) -> str:
        updated = str(text or "")
        for record in dependency_records:
            dep_id = str(record.get("id", "")).strip()
            if not dep_id:
                continue
            answer = str(record.get("answer", "")).strip()
            subject = cls._extract_dependency_subject(record)
            replacement = subject or answer
            answer_replacement = answer or replacement
            if answer_replacement:
                for pattern in (
                    rf"\bthe person identified in {re.escape(dep_id)}\b",
                    rf"\bperson identified in {re.escape(dep_id)}\b",
                    rf"\bthe value from {re.escape(dep_id)}\b",
                    rf"\bvalue from {re.escape(dep_id)}\b",
                    rf"\bthe answer from {re.escape(dep_id)}\b",
                    rf"\banswer from {re.escape(dep_id)}\b",
                    rf"\bthe entity from {re.escape(dep_id)}\b",
                    rf"\bentity from {re.escape(dep_id)}\b",
                ):
                    updated = re.sub(pattern, answer_replacement, updated, flags=re.IGNORECASE)
            if replacement:
                updated = re.sub(rf"\b{re.escape(dep_id)}\b", replacement, updated, flags=re.IGNORECASE)
        return updated

    @classmethod
    def _finalize_dependency_rewrite(
        cls,
        *,
        question: str,
        node: planner_utils.SubquestionNode,
        original_subquestion: str,
        qwen_rewritten_subquestion: str,
        dependency_records: Sequence[Dict[str, Any]],
    ) -> Tuple[str, bool, Optional[str]]:
        candidate = cls._clean_rewritten_question(qwen_rewritten_subquestion, original_subquestion)
        if not cls._contains_dependency_placeholders(candidate):
            return candidate, False, None

        if cls._looks_like_comparison_rewrite(node, candidate) and cls._question_mentions_dependency_subjects(
            question,
            dependency_records,
        ):
            return question.strip(), True, "comparison_original_question"

        substituted = cls._clean_rewritten_question(
            cls._substitute_dependency_placeholders(candidate, dependency_records),
            original_subquestion,
        )
        if not cls._contains_dependency_placeholders(substituted):
            if cls._looks_like_comparison_rewrite(node, substituted) and cls._question_mentions_dependency_subjects(
                question,
                dependency_records,
            ):
                return question.strip(), True, "comparison_original_question"
            if substituted != candidate:
                return substituted, True, "dependency_answer_substitution"
            return substituted, False, None

        if cls._looks_like_comparison_rewrite(node, original_subquestion) and cls._question_mentions_dependency_subjects(
            question,
            dependency_records,
        ):
            return question.strip(), True, "comparison_original_question"

        return original_subquestion, candidate != original_subquestion, "fallback_original_subquestion"

    def _resolve_dependency_subquestion(
        self,
        state: planner_utils.PipelineState,
        question: str,
        node: planner_utils.SubquestionNode,
    ) -> str:
        original_subquestion = str(node.subquestion).strip()
        if not node.depends_on:
            node.resolved_subquestion = original_subquestion
            return node.resolved_subquestion

        dependency_records = self._resolved_dependency_prompt_records(state=state, node=node)
        if not dependency_records:
            node.resolved_subquestion = original_subquestion
            return node.resolved_subquestion

        resolved_dependency_ids_json = planner_utils._safe_json_array_str(  # type: ignore[attr-defined]
            [
                str(record.get("id", "")).strip()
                for record in dependency_records
                if str(record.get("id", "")).strip()
            ]
        )
        resolved_dependency_context_json = planner_utils._safe_json_array_str(dependency_records)  # type: ignore[attr-defined]
        prompt = planner_utils.DEPENDENCY_REWRITE_PROMPT.format(
            question=question,
            subquestion=original_subquestion,
            resolved_dependency_ids_json=resolved_dependency_ids_json,
            resolved_dependency_context_json=resolved_dependency_context_json,
            operator_hints_json=planner_utils._safe_json_array_str(  # type: ignore[attr-defined]
                planner_utils._infer_operator_hints(original_subquestion or question)  # type: ignore[attr-defined]
            ),
            expected_answer_type_hint=planner_utils._infer_expected_answer_type_hint(  # type: ignore[attr-defined]
                original_subquestion or question
            ),
            slot_hints_json=planner_utils._safe_json_array_str(  # type: ignore[attr-defined]
                planner_utils._infer_slot_hints(original_subquestion or question)  # type: ignore[attr-defined]
            ),
        )

        qwen_rewritten_subquestion = original_subquestion
        try:
            parsed = self.planner.call_json(mode="dependency_rewrite", prompt=prompt)
            qwen_rewritten_subquestion = str(parsed.get("resolved_subquestion", "")).strip() or original_subquestion
        except Exception as exc:  # noqa: BLE001
            state.planning_trace.append(
                {
                    "event": "dependency_rewrite_error",
                    "node_id": node.id,
                    "original_subquestion": original_subquestion,
                    "error": str(exc),
                }
            )

        resolved_subquestion, fallback_used, fallback_reason = self._finalize_dependency_rewrite(
            question=question,
            node=node,
            original_subquestion=original_subquestion,
            qwen_rewritten_subquestion=qwen_rewritten_subquestion,
            dependency_records=dependency_records,
        )
        node.resolved_subquestion = resolved_subquestion
        state.planning_trace.append(
            {
                "event": "dependency_rewrite",
                "node_id": node.id,
                "original_subquestion": original_subquestion,
                "qwen_resolved_subquestion": qwen_rewritten_subquestion,
                "resolved_subquestion": node.resolved_subquestion,
                "dependency_ids": [
                    str(record.get("id", "")).strip()
                    for record in dependency_records
                    if str(record.get("id", "")).strip()
                ],
                "rewrite_changed": node.resolved_subquestion != original_subquestion,
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
            }
        )
        return node.resolved_subquestion

    def _run_subquestion(
        self,
        state: planner_utils.PipelineState,
        question: str,
        node: planner_utils.SubquestionNode,
        running_id_counter: int,
        full_passages: Sequence[hotpot_utils.Passage],
    ) -> int:
        del full_passages
        node.status = "running"
        execution_subquestion = self._resolve_dependency_subquestion(state=state, question=question, node=node)
        dependency_entries = self._dependency_memory_entries(state=state, node=node)
        dependency_passages = self._entries_to_passages(dependency_entries)
        has_dependency_memory = bool(dependency_passages)
        state.execution_order.append(node.id)

        pre_attempt: planner_utils.AttemptResult
        pre_route = "dependency_rewrite_question_only" if has_dependency_memory else "question_only"
        gate_score_pre: Optional[float]
        forced_retrieval_due_to_dinco_error = False
        dinco_failure_error: Optional[str] = None
        try:
            pre_attempt = self._question_only_attempt(execution_subquestion)
            gate_score_pre = self._gate_score(nvc=pre_attempt.nvc, dinco_conf=pre_attempt.dinco_conf)
            route_taken = base_runner.choose_route(self.routing_mode, gate_score_pre, self.gate_threshold)
        except Exception as exc:
            self._emit_printbad(
                question=question,
                node=node,
                execution_subquestion=execution_subquestion,
                dependency_entries=dependency_entries,
                pre_route=pre_route,
                exc=exc,
            )
            if not base_runner._is_retryable_dinco_azure_failure(exc):
                raise
            dinco_failure_error = str(exc)
            if self.retry_example_on_dinco_azure_failure and self._current_example_attempt == 0:
                raise base_runner._RetryExampleFromStart(
                    {
                        "event": "dinco_azure_failure_restart_example",
                        "question_id": self._current_question_id,
                        "node_id": node.id,
                        "resolved_subquestion": execution_subquestion,
                        "pre_route": pre_route,
                        "example_attempt_index": self._current_example_attempt,
                        "error": dinco_failure_error,
                    }
                ) from exc
            if not self.force_retrieval_on_repeat_dinco_azure_failure:
                raise
            forced_retrieval_due_to_dinco_error = True
            pre_attempt = self._make_dinco_azure_fallback_attempt(dinco_failure_error)
            gate_score_pre = None
            route_taken = "retrieve"
            state.planning_trace.append(
                {
                    "event": "dinco_azure_failure_force_retrieval",
                    "question_id": self._current_question_id,
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "pre_route": pre_route,
                    "example_attempt_index": self._current_example_attempt,
                    "error": dinco_failure_error,
                }
            )
        if not dependency_passages and self.root_subquestion_policy == "always_retrieve":
            route_taken = "retrieve"

        all_hits = (
            base_runner.search_passages(self.index, self._current_question_id, execution_subquestion, top_k=self.audit_top_k)
            if route_taken == "retrieve"
            else []
        )

        self._set_runtime(
            node,
            execution_subquestion=execution_subquestion,
            planner_retrieve_hint=bool(node.retrieve),
            pre_route=pre_route,
            route_taken=route_taken,
            pre_answer=pre_attempt.answer,
            pre_nvc=pre_attempt.nvc,
            pre_sc_conf=pre_attempt.sc_conf,
            pre_dinco_conf=pre_attempt.dinco_conf,
            gate_score_pre=gate_score_pre,
            dependency_memory_count=len(dependency_passages),
            dependency_memory_titles=[p.title for p in dependency_passages],
            dependency_memory_dinco_used=False,
            retrieved_chunk_ids_stage1=[],
            retrieved_titles_stage1=[],
            retrieved_scores_stage1=[],
            retrieved_chunk_ids_online=[],
            retrieved_titles_online=[],
            retrieved_scores_online=[],
            online_claims=[],
            online_g_mean=None,
            online_g_min=None,
            online_claim_supports=[],
            online_supported=None,
            grounding_skipped=None,
            grounding_mode=None,
            retry_used=False,
            strict_grounding_retry_used=False,
            strict_grounding_retry_triggered_by_weak_claim=False,
            strict_retry_answer="",
            strict_retry_claims=[],
            strict_retry_g_mean=None,
            strict_retry_g_min=None,
            strict_retry_claim_supports=[],
            root_closed_book_commit=False,
            dinco_azure_failure_forced_retrieval=forced_retrieval_due_to_dinco_error,
            dinco_azure_failure_error=dinco_failure_error,
            example_attempt_index=self._current_example_attempt,
        )

        state.planning_trace.append(
            {
                "event": "execute_subquestion",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "depends_on": list(node.depends_on),
                "planner_retrieve_hint": bool(node.retrieve),
                "pre_route": pre_route,
                "route_taken": route_taken,
                "gate_score_pre": gate_score_pre,
                "dependency_memory_count": len(dependency_passages),
                "rewritten_from_dependencies": bool(dependency_passages),
                "dependency_memory_dinco_used": False,
                "route_forced_by_evidence": False,
                "route_forced_by_dinco_azure_failure": forced_retrieval_due_to_dinco_error,
                "example_attempt_index": self._current_example_attempt,
            }
        )

        if route_taken == "skip":
            skip_pass = bool(
                pre_attempt.available
                and pre_attempt.dinco_conf is not None
                and pre_attempt.dinco_conf >= self.gate_threshold
                and hotpot_utils.normalize_answer(pre_attempt.answer)
                != hotpot_utils.normalize_answer("insufficient evidence")
            )
            allow_commit = True
            if has_dependency_memory:
                grounding_mode = "skipped_dependency_rewrite_closed_book"
                explanation = (
                    "Dependent subquestion committed from a high-confidence closed-book answer after dependency rewrite."
                )
                commit_source = "dependency_rewrite_closed_book_commit"
            else:
                grounding_mode = "skipped_closed_book"
                explanation = "Root subquestion committed from a high-confidence closed-book answer."
                commit_source = "closed_book_commit"
                if self.root_subquestion_policy == "skip_without_commit":
                    skip_pass = False
                allow_commit = self.root_subquestion_policy == "allow_closed_book_commit"
            if skip_pass and allow_commit:
                committed_attempt = planner_utils.AttemptResult(
                    answer=pre_attempt.answer,
                    raw_answer=pre_attempt.raw_answer,
                    nvc=pre_attempt.nvc,
                    sc_conf=pre_attempt.sc_conf,
                    dinco_conf=pre_attempt.dinco_conf,
                    source=commit_source,
                    support_claims=list(pre_attempt.support_claims),
                    explanation=explanation,
                    dinco_candidates=list(pre_attempt.dinco_candidates),
                    dinco_ptrues=list(pre_attempt.dinco_ptrues),
                    available=True,
                )
                self._set_runtime(
                    node,
                    online_claims=list(committed_attempt.support_claims),
                    grounding_skipped=True,
                    grounding_mode=grounding_mode,
                    root_closed_book_commit=not has_dependency_memory,
                )
                state.planning_trace.append(
                    {
                        "event": "subquestion_scored",
                        "node_id": node.id,
                        "resolved_subquestion": execution_subquestion,
                        "source": commit_source,
                        "answer": committed_attempt.answer,
                        "nvc": committed_attempt.nvc,
                        "sc_conf": committed_attempt.sc_conf,
                        "dinco_conf": committed_attempt.dinco_conf,
                        "g_mean": None,
                        "g_min": None,
                        "grounding_skipped": True,
                        "grounding_mode": grounding_mode,
                        "rewritten_from_dependencies": bool(dependency_passages),
                        "passed": True,
                    }
                )
                self._commit_success(node=node, attempt=committed_attempt, retrieved_titles=[])
                self._append_success_entry(state=state, question=question, node=node)
                return running_id_counter

            state.planning_trace.append(
                {
                    "event": "subquestion_scored",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "source": pre_attempt.source,
                    "answer": pre_attempt.answer,
                    "nvc": pre_attempt.nvc,
                    "sc_conf": pre_attempt.sc_conf,
                    "dinco_conf": pre_attempt.dinco_conf,
                    "rewritten_from_dependencies": bool(dependency_passages),
                    "passed": False,
                }
            )
            return self._decompose_node(
                state=state,
                question=question,
                node=node,
                running_id_counter=running_id_counter,
            )

        initial_hits = all_hits[: self.retrieval_top_k]
        stage1 = self._retrieval_attempt(
            subquestion=execution_subquestion,
            dependency_passages=dependency_passages,
            hits=initial_hits,
            fallback_answer=pre_attempt.answer,
        )

        self._set_runtime(
            node,
            retrieved_chunk_ids_stage1=[str(hit.row.get("chunk_id") or "") for hit in initial_hits],
            retrieved_titles_stage1=[str(hit.row.get("title") or "") for hit in initial_hits],
            retrieved_scores_stage1=[float(hit.score) for hit in initial_hits],
        )

        selected, retry_used, strict_retry = self._maybe_run_strict_grounding_retry(
            subquestion=execution_subquestion,
            dependency_passages=dependency_passages,
            hits=initial_hits,
            fallback_answer=pre_attempt.answer,
            first_attempt=stage1,
        )
        if strict_retry is not None:
            strict_attempt = strict_retry["attempt"]
            strict_support = strict_retry["support"]
            strict_supported = base_runner.is_supported(
                strict_support,
                self.support_mean_threshold,
                self.support_min_threshold,
            )
            self._set_runtime(
                node,
                strict_grounding_retry_used=True,
                strict_grounding_retry_triggered_by_weak_claim=True,
                strict_retry_answer=strict_attempt.answer,
                strict_retry_claims=list(strict_attempt.support_claims),
                strict_retry_g_mean=float(strict_support["g_mean"]),
                strict_retry_g_min=float(strict_support["g_min"]),
                strict_retry_claim_supports=list(strict_support["claim_supports"]),
            )
            state.planning_trace.append(
                {
                    "event": "retrieval_strict_grounding_retry",
                    "node_id": node.id,
                    "resolved_subquestion": execution_subquestion,
                    "trigger": "weak_hop_claim",
                    "previous_answer": stage1["attempt"].answer,
                    "previous_claims": list(stage1["attempt"].support_claims),
                    "previous_claim_supports": list(stage1["support"].get("claim_supports", [])),
                    "answer": strict_attempt.answer,
                    "g_mean": float(strict_support["g_mean"]),
                    "g_min": float(strict_support["g_min"]),
                    "claim_supports": list(strict_support["claim_supports"]),
                    "passed": bool(
                        strict_attempt.available
                        and strict_supported
                        and hotpot_utils.normalize_answer(strict_attempt.answer)
                        != hotpot_utils.normalize_answer("insufficient evidence")
                    ),
                }
            )

        selected_attempt = selected["attempt"]
        selected_support = selected["support"]
        online_supported = base_runner.is_supported(
            selected_support,
            self.support_mean_threshold,
            self.support_min_threshold,
        )
        passed = bool(
            selected_attempt.available
            and online_supported
            and hotpot_utils.normalize_answer(selected_attempt.answer)
            != hotpot_utils.normalize_answer("insufficient evidence")
        )
        runtime_g_mean = float(selected_support["g_mean"])
        runtime_g_min = float(selected_support["g_min"])
        runtime_claim_supports = list(selected_support["claim_supports"])
        grounding_skipped = False
        if retry_used:
            grounding_mode = (
                "retrieval_strict_grounding_retry_with_dependency_memory"
                if dependency_passages
                else "retrieval_strict_grounding_retry"
            )
        else:
            grounding_mode = "retrieval_with_dependency_memory" if dependency_passages else "retrieval_only_gate"

        self._set_runtime(
            node,
            retry_used=bool(retry_used),
            retrieved_chunk_ids_online=list(selected.get("evidence_chunk_ids", [])),
            retrieved_titles_online=list(selected.get("evidence_titles", [])),
            retrieved_scores_online=list(selected.get("evidence_scores", [])),
            online_claims=list(selected_attempt.support_claims),
            online_g_mean=runtime_g_mean,
            online_g_min=runtime_g_min,
            online_claim_supports=runtime_claim_supports,
            online_supported=bool(online_supported),
            grounding_skipped=grounding_skipped,
            grounding_mode=grounding_mode,
        )

        state.planning_trace.append(
            {
                "event": "subquestion_scored",
                "node_id": node.id,
                "resolved_subquestion": execution_subquestion,
                "source": selected_attempt.source,
                "answer": selected_attempt.answer,
                "nvc": selected_attempt.nvc,
                "sc_conf": selected_attempt.sc_conf,
                "dinco_conf": selected_attempt.dinco_conf,
                "g_mean": runtime_g_mean,
                "g_min": runtime_g_min,
                "grounding_mode": grounding_mode,
                "used_dinco": False,
                "retry_used": bool(retry_used),
                "passed": passed,
            }
        )

        if passed:
            repaired, running_id_counter = self._maybe_add_bridge_repair_child(
                state=state,
                question=question,
                node=node,
                execution_subquestion=execution_subquestion,
                attempt=selected_attempt,
                running_id_counter=running_id_counter,
            )
            if repaired:
                return running_id_counter
            self._commit_success(node=node, attempt=selected_attempt, retrieved_titles=selected_attempt.retrieved_titles)
            self._append_success_entry(state=state, question=question, node=node)
            return running_id_counter

        return self._decompose_node(
            state=state,
            question=question,
            node=node,
            running_id_counter=running_id_counter,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multihop calibrated retrieval on HotpotQA with local Qwen and the original "
            "closed-book beam-search DInCo formulation for question-only gating."
        )
    )
    parser.add_argument("--dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument(
        "--dataset_config",
        "--dataset_subset",
        dest="dataset_config",
        type=str,
        default="distractor",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=2000, help="<= 0 means full split")
    parser.add_argument(
        "--indexed_pool_limit",
        type=int,
        default=None,
        help=(
            "If set, first restrict the dataset to the first N examples before optional "
            "shuffle/limit. Use 2000 to sample from the indexed validation pool."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Defaults to false so the first validation examples are used.",
    )
    parser.add_argument(
        "--example_id",
        type=str,
        default=None,
        help="Run exactly one example id. Overrides --limit/--shuffle subset selection.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(BASE_DIR / "data" / "hotpotqa_distractor_validation_s0_n2000_chunks_bm25_index"),
    )
    parser.add_argument("--retrieval_top_k", type=int, default=8)
    parser.add_argument("--audit_top_k", type=int, default=8)
    parser.add_argument(
        "--retry_on_low_support",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry_extra_top_k", type=int, default=4)
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="dinco_gate",
        choices=["dinco_gate", "always_retrieve", "closed_book_only"],
    )
    parser.add_argument(
        "--gate_on",
        type=str,
        default="dinco",
        choices=["dinco", "nvc"],
    )
    parser.add_argument("--gate_threshold", type=float, default=0.80)
    parser.add_argument("--support_mean_threshold", type=float, default=0.70)
    parser.add_argument("--support_min_threshold", type=float, default=0.50)
    parser.add_argument(
        "--root_subquestion_policy",
        type=str,
        default="allow_closed_book_commit",
        choices=["always_retrieve", "allow_closed_book_commit", "skip_without_commit"],
    )
    parser.add_argument("--max_initial_subquestions", type=int, default=4)
    parser.add_argument("--max_subquestion_depth", type=int, default=2)
    parser.add_argument("--max_subquestion_nodes", type=int, default=12)
    parser.add_argument("--planner_max_new_tokens", type=int, default=800)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--n_distractors", type=int, default=5)
    parser.add_argument("--n_sc_samples", type=int, default=5)
    parser.add_argument("--sc_match_threshold", type=float, default=0.90)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--generator_device_map", type=str, default="auto")
    parser.add_argument(
        "--generator_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    parser.add_argument("--dinco_nli_model_name", type=str, default=DEFAULT_DINCO_NLI_MODEL_NAME)
    parser.add_argument("--dinco_beam_max_new_tokens", type=int, default=100)
    parser.add_argument("--dinco_beam_length_penalty", type=float, default=0.0)
    parser.add_argument("--minicheck_model_name", type=str, default="Bespoke-MiniCheck-7B")
    parser.add_argument(
        "--allow_minicheck_cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minicheck_cpu_fallback_model_name", type=str, default="roberta-large")
    parser.add_argument("--minicheck_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--minicheck_max_model_len", type=int, default=None)
    parser.add_argument(
        "--minicheck_gpu_memory_gb",
        type=float,
        default=25.0,
        help="Target MiniCheck GPU memory budget in GiB. Converted to vLLM gpu_memory_utilization at runtime.",
    )
    parser.add_argument(
        "--minicheck_gpu_memory_utilization",
        type=float,
        default=None,
        help="Optional explicit vLLM utilization ratio override. If unset, derived from --minicheck_gpu_memory_gb.",
    )
    parser.add_argument(
        "--noground",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated no-op. Retrieval now always uses MiniCheck grounding.",
    )
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument(
        "--printbad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the failing question/subquestion context if the DINCO pre-call raises.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.n_distractors < 1:
        raise ValueError("--n_distractors must be >= 1.")
    if args.n_sc_samples < 1:
        raise ValueError("--n_sc_samples must be >= 1.")
    if args.dinco_beam_max_new_tokens < 1:
        raise ValueError("--dinco_beam_max_new_tokens must be >= 1.")
    return args


def resolve_minicheck_gpu_memory_utilization(args: argparse.Namespace) -> Optional[float]:
    if args.minicheck_gpu_memory_utilization is not None:
        return float(args.minicheck_gpu_memory_utilization)
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    total_gib = float(torch.cuda.get_device_properties(device).total_memory) / float(1024**3)
    if total_gib <= 0:
        return None

    requested_ratio = float(args.minicheck_gpu_memory_gb) / total_gib
    return max(0.01, min(0.99, requested_ratio))


def default_output_paths(args: argparse.Namespace, limit: Optional[int]) -> Tuple[Path, Path]:
    subset_stem = base_runner.file_stem_for_subset(args.dataset_config, args.split, args.seed, limit)
    run_name = (
        f"{subset_stem}_multihop_{args.routing_mode}_{args.gate_on}_t{str(args.gate_threshold).replace('.', '_')}_"
        f"origbeam_{slugify_name(args.model_name.split('/')[-1])}"
    )
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else BASE_DIR / "results" / f"{run_name}.jsonl"
    summary_json = (
        Path(args.summary_json) if args.summary_json else BASE_DIR / "results" / f"{run_name}.summary.json"
    )
    return output_jsonl, summary_json


def build_summary(records: Sequence[Dict[str, Any]], args: argparse.Namespace, output_jsonl: Path) -> Dict[str, Any]:
    minicheck_gpu_memory_utilization = resolve_minicheck_gpu_memory_utilization(args)
    summary = base_runner.build_summary(records=records, args=args, output_jsonl=output_jsonl)
    summary.update(
        {
            "backend": "qwen32b_origbeam_closed_book_dinco",
            "model_name": args.model_name,
            "generator_device_map": args.generator_device_map,
            "generator_dtype": args.generator_dtype,
            "planner_max_new_tokens": args.planner_max_new_tokens,
            "max_retries": args.max_retries,
            "n_distractors": args.n_distractors,
            "n_sc_samples": args.n_sc_samples,
            "sc_match_threshold": args.sc_match_threshold,
            "dinco_nli_model_name": args.dinco_nli_model_name,
            "dinco_beam_max_new_tokens": args.dinco_beam_max_new_tokens,
            "dinco_beam_length_penalty": args.dinco_beam_length_penalty,
            "minicheck_model_name": args.minicheck_model_name,
            "allow_minicheck_cpu_fallback": args.allow_minicheck_cpu_fallback,
            "minicheck_cpu_fallback_model_name": args.minicheck_cpu_fallback_model_name,
            "minicheck_tensor_parallel_size": args.minicheck_tensor_parallel_size,
            "minicheck_max_model_len": args.minicheck_max_model_len,
            "minicheck_gpu_memory_gb": args.minicheck_gpu_memory_gb,
            "minicheck_gpu_memory_utilization": minicheck_gpu_memory_utilization,
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    hotpot_utils.seed_everything(args.seed)

    limit = normalize_limit(args.limit)
    output_jsonl, summary_json = default_output_paths(args, limit=limit)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    examples = base_runner.load_examples(args, limit=limit)
    index = BM25Index.load(Path(args.index_dir))

    if args.dry_run:
        qwen_model: Any = hotpot_utils.MockQwenModel()
        planner: Any = planner_utils.MockQwenPlannerModel()
        dinco: Any = hotpot_utils.MockDincoCalibrator()
        grounder: Any = hotpot_utils.MockMiniCheckGrounder()
    else:
        minicheck_gpu_memory_utilization = resolve_minicheck_gpu_memory_utilization(args)
        if minicheck_gpu_memory_utilization is None:
            minicheck_gpu_memory_utilization = 0.4
        qwen_model = hotpot_utils.QwenDincoModel(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            device_map=args.generator_device_map,
            dtype=args.generator_dtype,
        )
        planner = planner_utils.QwenPlannerModel(
            qwen_model=qwen_model,
            max_new_tokens=args.planner_max_new_tokens,
            max_retries=args.max_retries,
        )
        dinco = OriginalClosedBookBeamDincoCalibrator(
            qwen_model=qwen_model,
            cache_dir=args.cache_dir,
            n_sc_samples=args.n_sc_samples,
            sc_match_threshold=args.sc_match_threshold,
            nli_model_name=args.dinco_nli_model_name,
            beam_max_new_tokens=args.dinco_beam_max_new_tokens,
            beam_length_penalty=args.dinco_beam_length_penalty,
        )
        grounder = hotpot_utils.MiniCheckGrounder(
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.minicheck_tensor_parallel_size,
            max_model_len=args.minicheck_max_model_len,
            model_name=args.minicheck_model_name,
            allow_cpu_fallback=args.allow_minicheck_cpu_fallback,
            cpu_fallback_model_name=args.minicheck_cpu_fallback_model_name,
            gpu_memory_utilization=minicheck_gpu_memory_utilization,
        )

    runner = GPT54ParityQwenPlannerRunner(
        planner=planner,
        qwen_model=qwen_model,
        subquestion_qwen_model=qwen_model,
        dinco=dinco,
        grounder=grounder,
        index=index,
        gate_on=args.gate_on,
        gate_threshold=args.gate_threshold,
        support_mean_threshold=args.support_mean_threshold,
        support_min_threshold=args.support_min_threshold,
        routing_mode=args.routing_mode,
        retry_on_low_support=args.retry_on_low_support,
        retry_extra_top_k=args.retry_extra_top_k,
        audit_top_k=args.audit_top_k,
        root_subquestion_policy=args.root_subquestion_policy,
        max_initial_subquestions=args.max_initial_subquestions,
        max_subquestion_depth=args.max_subquestion_depth,
        max_subquestion_nodes=args.max_subquestion_nodes,
        retrieval_top_k=args.retrieval_top_k,
        n_distractors=args.n_distractors,
        printbad=args.printbad,
        noground=args.noground,
    )

    records: List[Dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as writer:
        for example in examples:
            row = runner.run_example(example)
            writer.write(json.dumps(row, ensure_ascii=True) + "\n")
            writer.flush()
            records.append(row)

    summary = build_summary(records, args=args, output_jsonl=output_jsonl)
    write_json(summary, summary_json)
    print(f"Wrote {len(records)} records to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
