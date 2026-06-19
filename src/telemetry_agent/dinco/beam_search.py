"""DINCO calibration on context-grounded HotpotQA short-answer QA.

Compares sampling-DINCO vs beam-search-DINCO vs raw-verbal-baseline at
matched compute on the same example IDs. vLLM-only backend.

Run:
    # Canary
    python dinco_hotpotqa_vllm.py --mode canary --out-dir results/canary

    # Full (after canary passes data-validator CLEAN)
    python dinco_hotpotqa_vllm.py --mode full --out-dir results/full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import yaml

# Expects ``HF_TOKEN`` to be pre-exported in the environment when the dataset
# or model artifacts are gated; ``assert_env`` will surface a clear error if
# the token is missing.

from .chat_prompt import build_chat_prompt
from telemetry_agent.dinco._controls import select_control_rows, find_swap_answer
from telemetry_agent.dinco._hotpot_eval import (
    em_score, f1_score, format_context,
    build_generator_messages, build_validator_messages,
)
from telemetry_agent.dinco.core import (
    GenResult,
    PTRUE_FALLBACK_NAN,
    build_yes_no_sets,
    compute_nvc,
    compute_pairwise_nli,
    dedupe_distractors,
    load_nli,
    vllm_beam_distractors,
    vllm_greedy,
    vllm_sample_distractors,
    vllm_validator_ptrue,
)
from telemetry_agent.dinco import _calibration as cal


HERE = Path(__file__).parent.resolve()
CONFIG_PATH_DEFAULT = HERE.parent.parent.parent / "notes" / "experiments" / "dinco-beam-vs-sampling-hotpotqa" / "experiment.yaml"

CONDITIONS = ["sampling_dinco", "beam_dinco", "raw_verbal_baseline"]


# ---------------------------------------------------------------------------
# Config + boot
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def assert_env() -> None:
    """Hard env asserts. Surfaces config bugs before any GPU is requested."""
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        raise RuntimeError(
            "VLLM_WORKER_MULTIPROC_METHOD must be 'spawn'; "
            f"got {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')!r}"
        )
    if os.environ.get("HF_DATASETS_TRUST_REMOTE_CODE") != "1":
        raise RuntimeError(
            "HF_DATASETS_TRUST_REMOTE_CODE must be '1' for hotpotqa/hotpot_qa loader"
        )
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing — partial uploads will fail")


def run_calibration_self_test() -> None:
    """Run test_calibration.py as a subprocess; abort if any test fails."""
    print("[boot] running calibration self-tests...", flush=True)
    rc = subprocess.run(
        [sys.executable, str(HERE / "test_calibration.py")],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    print(rc.stdout)
    if rc.returncode != 0:
        print(rc.stderr, file=sys.stderr)
        raise RuntimeError("Calibration self-tests failed; refusing to run on GPU.")


def loader_sanity_check() -> None:
    """Verify HotpotQA distractor split loads with the expected row count."""
    from datasets import load_dataset
    print("[boot] verifying HotpotQA loader...", flush=True)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    if len(ds) != 7405:
        raise RuntimeError(f"HotpotQA validation expected 7405 rows, got {len(ds)}")
    print(f"[boot] HotpotQA distractor validation: {len(ds)} rows OK", flush=True)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(ds, n_per_cell: int, seed: int,
                      min_yes_no: int = 3, min_multi_token: int = 2) -> List[int]:
    """Return a list of dataset indices, stratified by (type × level)."""
    rng = random.Random(seed)
    by_cell: Dict[Tuple[str, str], List[int]] = {}
    yes_no: List[int] = []
    multi_token: List[int] = []

    for i, ex in enumerate(ds):
        key = (ex["type"], ex["level"])
        by_cell.setdefault(key, []).append(i)
        ans = ex["answer"].lower().strip()
        if ans in ("yes", "no"):
            yes_no.append(i)
        if len(ex["answer"].split()) >= 3:
            multi_token.append(i)

    selected: List[int] = []
    seen: Set[int] = set()
    for key, idxs in by_cell.items():
        rng.shuffle(idxs)
        take = idxs[:n_per_cell]
        for i in take:
            if i not in seen:
                seen.add(i)
                selected.append(i)

    def _have_yn():
        return sum(1 for i in selected if ds[i]["answer"].lower().strip() in ("yes", "no"))

    def _have_mt():
        return sum(1 for i in selected if len(ds[i]["answer"].split()) >= 3)

    target_n = n_per_cell * len(by_cell)  # canonical sample size

    # Pad with yes/no and multi-token examples by REPLACING the last entry of
    # the most-overrepresented cell, never exceeding target_n.
    def _pad(pool: List[int], have_fn, min_count: int) -> None:
        rng.shuffle(pool)
        for i in pool:
            if have_fn() >= min_count:
                return
            if i in seen:
                continue
            # find a cell with >= n_per_cell entries to swap out
            counts: Dict[Tuple[str, str], int] = {}
            for j in selected:
                key = (ds[j]["type"], ds[j]["level"])
                counts[key] = counts.get(key, 0) + 1
            # pick the largest cell that does NOT contain this candidate's match
            largest = max(counts, key=lambda k: counts[k])
            for k_idx, j in enumerate(selected):
                if (ds[j]["type"], ds[j]["level"]) == largest:
                    seen.discard(j)
                    selected[k_idx] = i
                    seen.add(i)
                    break

    _pad(yes_no, _have_yn, min_yes_no)
    _pad(multi_token, _have_mt, min_multi_token)

    assert len(selected) == target_n, f"sample size drifted: {len(selected)} != {target_n}"

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def load_completed_ids(out_dir: Path) -> Set[str]:
    """A row is 'completed' iff it appears in ALL three condition JSONLs.

    We rebuild this set from the per-condition files on resume.
    """
    per_condition: Dict[str, Set[str]] = {c: set() for c in CONDITIONS}
    for cond in CONDITIONS:
        path = out_dir / f"{cond}.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            lines = f.readlines()
        # Drop a trailing line if it doesn't parse (mid-write crash).
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                per_condition[cond].add(row["id"])
            except json.JSONDecodeError:
                continue
    common = set.intersection(*per_condition.values()) if per_condition else set()
    return common


def append_jsonl_atomic(path: Path, row: Dict) -> None:
    line = json.dumps(row, ensure_ascii=False)
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Per-example processing
# ---------------------------------------------------------------------------

def process_example(
    *,
    ex: Dict,
    llm,
    tokenizer,
    nli_tok,
    nli_model,
    yes_set: Set[int],
    no_set: Set[int],
    cfg: dict,
    job_id: str,
    cluster: str,
    remove_gold: bool = False,
) -> Tuple[Optional[Dict[str, Dict]], Optional[str]]:
    """Run all 3 conditions on a single example.

    Returns (rows_dict, None) on success or (None, failure_reason) so callers
    can record a specific drop reason rather than a generic "condition_failure".
    """
    gen_cfg = cfg["config"]["generation"]
    sat_thresh = cfg["config"]["metrics"]["saturation_threshold"]

    if remove_gold:
        gold_titles = set(ex.get("supporting_facts", {}).get("title", []) or [])
        context_str = format_context(ex["context"], exclude_titles=gold_titles)
    else:
        context_str = format_context(ex["context"])
    question = ex["question"]
    gold = ex["answer"]

    # --- Greedy (computed once, reused) ---
    gen_msgs = build_generator_messages(question, context_str)
    gen_prompt = build_chat_prompt(tokenizer, gen_msgs)
    greedy = vllm_greedy(llm, gen_prompt, max_tokens=gen_cfg["max_tokens_answer"])
    if greedy.finish_reason == "length":
        return None, f"greedy_truncated: raw={greedy.raw_text[:80]!r}"
    greedy_text = greedy.text
    if not greedy_text:
        return None, f"greedy_empty: raw={greedy.raw_text[:80]!r}"

    # --- Raw verbal P(True) on greedy (computed once, reused) ---
    val_prompt = build_chat_prompt(
        tokenizer, build_validator_messages(question, context_str, greedy_text)
    )
    raw_ptrue_list = vllm_validator_ptrue(
        llm, [val_prompt], yes_set, no_set, logprobs_k=gen_cfg["validator_logprobs"]
    )
    raw_ptrue = raw_ptrue_list[0]
    if raw_ptrue != raw_ptrue:  # NaN check
        return None, f"raw_ptrue_nan: greedy={greedy_text!r}"

    em = em_score(greedy_text, gold)
    f1, _, _ = f1_score(greedy_text, gold)

    base_row = {
        "id": ex["id"],
        "question": question,
        "gold_answer": gold,
        "type": ex["type"],
        "level": ex["level"],
        "context_str": context_str,
        "greedy_answer": greedy_text,
        "greedy_raw": greedy.raw_text,
        "greedy_finish_reason": greedy.finish_reason,
        "raw_verbal_ptrue": raw_ptrue,
        "em": em,
        "f1": f1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cluster": cluster,
        "job_id": job_id,
    }

    rows: Dict[str, Dict] = {}

    # --- Condition: raw_verbal_baseline ---
    rows["raw_verbal_baseline"] = {
        **base_row,
        "condition": "raw_verbal_baseline",
        "distractors": None,
        "distractor_ptrues": None,
        "distractor_finish_reasons": None,
        "distractor_mean_ptrue": None,
        "dinco_confidence": raw_ptrue,
    }

    # --- Condition: sampling_dinco + beam_dinco ---
    distractor_sources = {
        "sampling_dinco": lambda: vllm_sample_distractors(
            llm, gen_prompt,
            n=gen_cfg["sampling"]["n"],
            temperature=gen_cfg["sampling"]["temperature"],
            top_p=gen_cfg["sampling"]["top_p"],
            max_tokens=gen_cfg["max_tokens_distractor"],
            seed=gen_cfg["sampling"]["seed"],
        ),
        "beam_dinco": lambda: vllm_beam_distractors(
            llm, gen_prompt,
            beam_width=gen_cfg["beam"]["beam_width"],
            max_tokens=gen_cfg["max_tokens_distractor"],
            length_penalty=gen_cfg["beam"]["length_penalty"],
        ),
    }

    from telemetry_agent.dinco.core import lexical_clean_str  # already imported above; explicit for normalize use

    def _norm(s: str) -> str:
        return lexical_clean_str(s).lower().replace('.', '').strip()

    greedy_norm = _norm(greedy_text)

    for cond_name, get_distractors in distractor_sources.items():
        gens: List[GenResult] = get_distractors()
        # Truncation check (real failure — keep dropping)
        truncated = [i for i, g in enumerate(gens) if g.finish_reason == "length"]
        if truncated:
            return None, f"{cond_name}_distractor_truncated: idxs={truncated}, sample_raw={gens[truncated[0]].raw_text[:80]!r}"

        # Agreement rate: fraction of generated samples that match the greedy
        # answer after normalization. Captures the self-consistency signal —
        # high agreement = model collapsed to one answer = high confidence.
        if gens:
            agreement_rate = sum(1 for g in gens if _norm(g.text) == greedy_norm) / len(gens)
        else:
            agreement_rate = 1.0

        # Lexical clean + dedupe; ensure the greedy is the main candidate (index 0)
        all_strs = [greedy_text] + [g.text for g in gens]
        all_scores = [0.0] + [g.cum_logprob if g.cum_logprob is not None else -1e6 for g in gens]
        all_scores[0] = max(all_scores) + 1.0
        cleaned, _ = dedupe_distractors(all_strs, all_scores)

        # GRACEFUL DEGENERATE: do NOT drop on too-few-distractors. When samples
        # collapse onto the greedy, NVC degenerates to raw P(True) — which is
        # the right answer because there are no real epistemic alternatives.
        n_unique_distractors = max(0, len(cleaned) - 1)  # excluding greedy at index 0
        degenerate = n_unique_distractors == 0

        # P(True) for each candidate
        val_prompts = [
            build_chat_prompt(tokenizer, build_validator_messages(question, context_str, c))
            for c in cleaned
        ]
        ptrues = vllm_validator_ptrue(
            llm, val_prompts, yes_set, no_set, logprobs_k=gen_cfg["validator_logprobs"]
        )
        nan_idx = [i for i, p in enumerate(ptrues) if p != p]
        if nan_idx:
            return None, f"{cond_name}_distractor_ptrue_nan: idxs={nan_idx}, candidates={[cleaned[i] for i in nan_idx]}"

        # NVC: only meaningful when there are >=1 distractors. Otherwise pass through.
        if n_unique_distractors >= 1:
            ptrues_t = torch.tensor(ptrues, dtype=torch.float32)
            nli = compute_pairwise_nli(nli_tok, nli_model, question, cleaned)
            dinco_conf = compute_nvc(ptrues_t, nli)
        else:
            dinco_conf = float(ptrues[0])  # raw P(True) on greedy — no normalization possible

        distractor_ptrues = ptrues[1:]
        distractor_finish_reasons = [g.finish_reason for g in gens]
        rows[cond_name] = {
            **base_row,
            "condition": cond_name,
            "distractors": cleaned[1:],  # exclude greedy
            "distractor_ptrues": distractor_ptrues,
            "distractor_finish_reasons": distractor_finish_reasons,
            "distractor_mean_ptrue": (
                float(sum(distractor_ptrues) / len(distractor_ptrues))
                if distractor_ptrues else None
            ),
            "n_unique_distractors": n_unique_distractors,
            "degenerate": degenerate,
            "agreement_rate": agreement_rate,
            "dinco_confidence": dinco_conf,
        }

    return rows, None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(jsonl_path: Path, sat_thresh: float) -> Dict:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {"n": 0}
    em = [r["em"] for r in rows]
    f1 = [r["f1"] for r in rows]
    conf = [r["dinco_confidence"] for r in rows]
    metrics = cal.all_metrics(em, conf, n_bins=15, saturation_threshold=sat_thresh)
    metrics["mean_em"] = float(sum(em) / len(em))
    metrics["mean_f1"] = float(sum(f1) / len(f1))
    # Distractor-mean saturation (W-6)
    dist_means = [r["distractor_mean_ptrue"] for r in rows if r["distractor_mean_ptrue"] is not None]
    metrics["distractor_mean_saturation"] = (
        cal.saturation_index(dist_means, sat_thresh) if dist_means else float("nan")
    )
    metrics["n_rows"] = len(rows)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["canary", "full"], required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", default=str(CONFIG_PATH_DEFAULT), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-override", type=int, default=None)
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="Optional path to a text file with one HotpotQA ID per "
                             "line. If set, bypasses stratified sampling and runs "
                             "exactly those examples (in file order).")
    parser.add_argument("--remove-gold", action="store_true", default=False,
                        help="Drop the 2 gold supporting paragraphs from the context "
                             "(simulates retrieval that missed the gold). Reads "
                             "supporting_facts.title from each example.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip subprocess HF uploads (for local dev).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    print(f"[boot] mode={args.mode} out_dir={args.out_dir}", flush=True)
    assert_env()
    run_calibration_self_test()
    loader_sanity_check()

    # --- Sampling ---
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)

    if args.mode == "canary":
        n_per_cell = 10  # 30 bridge + 30 comparison; 60 total
        flush_every = cfg["streaming"]["flush_every_examples_canary"]
    else:
        n_per_cell = (args.n_override or cfg["config"]["evaluation"]["n_samples"]) // 6
        flush_every = cfg["streaming"]["flush_every_examples"]

    if args.ids_file:
        wanted_ids = [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()]
        id_to_idx = {ds[i]["id"]: i for i in range(len(ds))}
        indices = [id_to_idx[i] for i in wanted_ids if i in id_to_idx]
        missing = [i for i in wanted_ids if i not in id_to_idx]
        print(f"[boot] --ids-file: {len(indices)}/{len(wanted_ids)} matched, "
              f"{len(missing)} missing", flush=True)
    else:
        indices = stratified_sample(ds, n_per_cell=n_per_cell, seed=args.seed)
        print(f"[boot] selected {len(indices)} examples", flush=True)

    # --- Resume ---
    completed = load_completed_ids(args.out_dir)
    print(f"[boot] resuming with {len(completed)} completed examples", flush=True)

    # --- Models ---
    print("[boot] loading vLLM engine...", flush=True)
    # (legacy path injection removed)
    from calibrated_retrieval_core import build_vllm_engine
    vllm_cfg = cfg["cluster"]["vllm"]
    llm, tokenizer = build_vllm_engine(
        model_name=cfg["config"]["models"][0],
        max_model_len=vllm_cfg["max_model_len"],
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
        enforce_eager=vllm_cfg["enforce_eager"],
    )

    print("[boot] loading NLI model...", flush=True)
    nli_tok, nli_model = load_nli()

    yes_set, no_set = build_yes_no_sets(tokenizer)
    print(f"[boot] yes_set={yes_set} no_set={no_set}", flush=True)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    cluster = os.environ.get("CLUSTER_NAME", "local")
    sat_thresh = cfg["config"]["metrics"]["saturation_threshold"]

    # --- Main loop ---
    n_processed = 0
    n_dropped = 0
    drops_log = args.out_dir / "drops.jsonl"
    t_start = time.time()

    for idx_i, ds_idx in enumerate(indices):
        ex = dict(ds[int(ds_idx)])
        if ex["id"] in completed:
            continue

        rows = None
        reason: Optional[str] = None
        try:
            rows, reason = process_example(
                ex=ex, llm=llm, tokenizer=tokenizer,
                nli_tok=nli_tok, nli_model=nli_model,
                yes_set=yes_set, no_set=no_set,
                cfg=cfg, job_id=job_id, cluster=cluster,
                remove_gold=args.remove_gold,
            )
        except Exception as e:
            import traceback
            reason = f"exception: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"

        if rows is None:
            n_dropped += 1
            with drops_log.open("a") as f:
                f.write(json.dumps({"id": ex["id"], "reason": reason or "unknown"}) + "\n")
            print(f"[loop] DROP {ex['id']}: {reason}", flush=True)
            continue

        # Atomic write across all 3 conditions
        for cond, row in rows.items():
            append_jsonl_atomic(args.out_dir / f"{cond}.jsonl", row)
        n_processed += 1

        # Partial flush every K examples
        if n_processed > 0 and n_processed % flush_every == 0:
            print(f"[loop] {n_processed} done, dropped {n_dropped}, "
                  f"elapsed {time.time() - t_start:.0f}s", flush=True)
            if not args.no_upload:
                upload_partials(args.out_dir, cfg, args.mode, job_id, cluster, partial=True)

    # --- Final aggregate + upload ---
    print("[final] aggregating metrics...", flush=True)
    summary = {
        "mode": args.mode,
        "n_processed": n_processed,
        "n_dropped": n_dropped,
        "n_examples_attempted": len(indices),
        "elapsed_seconds": time.time() - t_start,
        "per_condition": {},
    }
    for cond in CONDITIONS:
        path = args.out_dir / f"{cond}.jsonl"
        if path.exists():
            summary["per_condition"][cond] = aggregate_metrics(path, sat_thresh)
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # --- Validator-confound controls (canary only) ---
    if args.mode == "canary":
        run_validator_confound_controls(
            ds=ds, indices=indices,
            llm=llm, tokenizer=tokenizer,
            yes_set=yes_set, no_set=no_set,
            cfg=cfg, out_dir=args.out_dir,
        )

    if not args.no_upload:
        upload_partials(args.out_dir, cfg, args.mode, job_id, cluster, partial=False)

    return 0


def run_validator_confound_controls(*, ds, indices, llm, tokenizer, yes_set, no_set,
                                     cfg, out_dir: Path) -> None:
    """For each of up to 5 control categories, pick one matching row from the
    canary selection and run validator P(True) on (greedy, paraphrase(s), swapped).
    Writes canary_controls_results.jsonl. Logs WARNING if <3 categories matched.
    """
    print("[controls] running validator-confound controls...", flush=True)
    rows = [dict(ds[int(i)]) for i in indices]
    selected = select_control_rows(rows)
    if len(selected) < 3:
        print(f"[controls] WARNING: only {len(selected)} control categories matched the canary set",
              flush=True)
    out_path = out_dir / "canary_controls_results.jsonl"
    gen_cfg = cfg["config"]["generation"]

    for ex, name, paraphrases in selected:
        context_str = format_context(ex["context"])
        question = ex["question"]
        gold = ex["answer"]
        # Greedy first (for the "greedy" variant we need the model's actual answer)
        gen_prompt = build_chat_prompt(tokenizer, build_generator_messages(question, context_str))
        greedy = vllm_greedy(llm, gen_prompt, max_tokens=gen_cfg["max_tokens_answer"])
        if greedy.finish_reason == "length" or not greedy.text:
            continue
        swap = find_swap_answer(ex, rows) or "an unrelated answer"
        # Build candidate variants. The "greedy" variant uses the model's actual greedy.
        variants = [("greedy", greedy.text)]
        for i, p in enumerate(paraphrases):
            variants.append((f"paraphrase_{i}", p))
        variants.append(("swapped", swap))
        # Single batched validator call per row
        prompts = [
            build_chat_prompt(tokenizer, build_validator_messages(question, context_str, cand))
            for _, cand in variants
        ]
        ptrues = vllm_validator_ptrue(
            llm, prompts, yes_set, no_set, logprobs_k=gen_cfg["validator_logprobs"]
        )
        result = {
            "id": ex["id"],
            "category": name,
            "question": question,
            "gold_answer": gold,
            "type": ex["type"],
            "level": ex["level"],
            "variants": [
                {"kind": kind, "candidate": cand, "ptrue": p}
                for (kind, cand), p in zip(variants, ptrues)
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl_atomic(out_path, result)
        print(f"[controls] {name}: greedy={ptrues[0]:.3f} paraphrase={ptrues[1]:.3f} swapped={ptrues[-1]:.3f}",
              flush=True)
    print(f"[controls] wrote {out_path.name} with {len(selected)} rows", flush=True)


def upload_partials(out_dir: Path, cfg: dict, mode: str, job_id: str, cluster: str,
                    partial: bool) -> None:
    """Upload each per-condition JSONL via subprocess. Never imports
    huggingface_hub in the vLLM process."""
    slug = "dinco-beam-vs-sampling-hotpotqa"
    template = cfg["streaming"]["partial_dataset_name_template"]
    for cond in CONDITIONS:
        path = out_dir / f"{cond}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        if partial:
            name = template.format(condition=cond)
        else:
            tag = "canary-v1" if mode == "canary" else "v1"
            name = f"{slug}-{cond}-{tag}"
        meta = {
            "experiment_name": slug,
            "job_id": job_id,
            "cluster": cluster,
            "artifact_status": "partial" if partial else "final",
            "canary": mode == "canary",
            "model": cfg["config"]["models"][0],
            "condition": cond,
        }
        col_descs = {
            "greedy_answer": "Cleaned greedy answer (shared across conditions)",
            "raw_verbal_ptrue": "P(True) from validator on greedy answer (shared across conditions)",
            "distractors": "Distractor candidates (sampling or beam, post lexical-clean dedupe)",
            "distractor_ptrues": "P(True) per distractor (parallel to distractors); excludes greedy",
            "dinco_confidence": "Final DINCO NVC for the condition; equals raw_verbal_ptrue for raw_verbal_baseline",
            "em": "HotpotQA exact match (binary)",
            "f1": "HotpotQA token F1",
        }
        cmd = [
            sys.executable, str(HERE / "upload_helper.py"),
            "--jsonl", str(path),
            "--name", name,
            "--description", f"DINCO calibration on HotpotQA-distractor; condition={cond}; mode={mode}",
            "--metadata-json", json.dumps(meta),
            "--column-descriptions-json", json.dumps(col_descs),
            "--tags-json", json.dumps(["dinco", "calibration", "hotpotqa", "qwen32b", cond, mode]),
            "--private",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[upload] FAILED for {cond}: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
