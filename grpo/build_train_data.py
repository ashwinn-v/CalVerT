#!/usr/bin/env python3
"""Build a stratified HotpotQA-distractor TRAIN corpus for production GRPO.

Selects 10,000 questions from HotpotQA train split stratified by difficulty:
  5,000 hard + 2,500 medium + 2,500 easy
…plus a held-out validation slice (default 500 questions, stratified same way).

Outputs:
  1. <output_dir>/grpo_hotpot_10k_train.parquet  — training-prompt format
  2. <output_dir>/grpo_hotpot_10k_train.val.parquet  — held-out val (same schema)
  3. <chunks_jsonl>  — gold + distractor passages chunked, for BM25 indexing
                       (feed into telemetry_agent.retrieval.build_index)

Usage:
    python -m grpo.build_hotpot_train_10k \\
        --output_dir grpo/data/ \\
        --chunks_jsonl data/hotpotqa_train_10k_chunks.jsonl \\
        --val_n 500

Then build the BM25 index:
    python -m telemetry_agent.retrieval.build_index \\
        --input_jsonl data/hotpotqa_train_10k_chunks.jsonl \\
        --output_dir data/hotpotqa_train_10k_chunks_bm25_index
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the canary builder's prompt construction so train/eval prompts match.
from grpo.build_canary_data import _build_prompt


HOTPOT_DATASET = "hotpotqa/hotpot_qa"
HOTPOT_CONFIG = "distractor"
HOTPOT_SPLIT = "train"

DEFAULT_TRAIN_COUNTS = {"hard": 5000, "medium": 2500, "easy": 2500}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Where to write the train + val parquets.",
    )
    p.add_argument(
        "--chunks_jsonl", type=Path, required=True,
        help="Where to write the chunks JSONL for BM25 indexing.",
    )
    p.add_argument(
        "--train_hard", type=int, default=DEFAULT_TRAIN_COUNTS["hard"],
        help=f"Number of hard train questions (default {DEFAULT_TRAIN_COUNTS['hard']}).",
    )
    p.add_argument(
        "--train_medium", type=int, default=DEFAULT_TRAIN_COUNTS["medium"],
        help=f"Number of medium train questions (default {DEFAULT_TRAIN_COUNTS['medium']}).",
    )
    p.add_argument(
        "--train_easy", type=int, default=DEFAULT_TRAIN_COUNTS["easy"],
        help=f"Number of easy train questions (default {DEFAULT_TRAIN_COUNTS['easy']}).",
    )
    p.add_argument(
        "--val_n", type=int, default=500,
        help="Total validation questions, stratified same way (default 500).",
    )
    p.add_argument(
        "--chunk_words", type=int, default=120,
        help="Chunking window (words) for the BM25 corpus.",
    )
    p.add_argument(
        "--chunk_overlap_words", type=int, default=40,
        help="Chunk overlap (words).",
    )
    p.add_argument(
        "--prefer_paragraphs", action=argparse.BooleanOptionalAction, default=True,
        help="Keep paragraph chunks when possible before falling back to sliding windows.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stratified_sample(
    indexed_by_level: Dict[str, List[int]],
    counts: Dict[str, int],
    rng: random.Random,
) -> List[int]:
    """Sample `counts[level]` indices from each level's pool. Shuffles in place."""
    chosen: List[int] = []
    for level, n_target in counts.items():
        pool = indexed_by_level.get(level, [])
        if len(pool) < n_target:
            print(
                f"[build_hotpot_train_10k] WARN: level={level!r} has only "
                f"{len(pool)} examples, requested {n_target}; taking all.",
                file=sys.stderr,
            )
            chosen.extend(pool)
            continue
        rng.shuffle(pool)
        chosen.extend(pool[:n_target])
    return chosen


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"[build_hotpot_train_10k] loading {HOTPOT_DATASET}/{HOTPOT_CONFIG} split={HOTPOT_SPLIT}…", flush=True)
    from datasets import load_dataset
    ds = load_dataset(HOTPOT_DATASET, HOTPOT_CONFIG, split=HOTPOT_SPLIT)
    print(f"[build_hotpot_train_10k] loaded {len(ds)} train examples", flush=True)

    # Index by level. HotpotQA train has 'level' in {"easy","medium","hard"}.
    by_level: Dict[str, List[int]] = defaultdict(list)
    level_counter: Counter = Counter()
    for idx in range(len(ds)):
        level = str(ds[idx]["level"] or "unknown").lower().strip()
        by_level[level].append(idx)
        level_counter[level] += 1
    print(f"[build_hotpot_train_10k] level distribution: {dict(level_counter)}", flush=True)

    # ---- Stratified sample for TRAIN ----
    train_counts = {
        "hard": args.train_hard,
        "medium": args.train_medium,
        "easy": args.train_easy,
    }
    train_total = sum(train_counts.values())
    print(f"[build_hotpot_train_10k] sampling TRAIN: {train_counts} = {train_total}", flush=True)
    # Copy so we can shuffle without affecting val
    train_pool_by_level = {lvl: list(pool) for lvl, pool in by_level.items()}
    train_indices = stratified_sample(train_pool_by_level, train_counts, rng)
    train_index_set = set(train_indices)

    # ---- Stratified sample for VAL (disjoint from train) ----
    val_indices: List[int] = []
    if args.val_n > 0:
        # Same hard/medium/easy proportions: 50% hard, 25% medium, 25% easy
        val_counts = {
            "hard": args.val_n // 2,
            "medium": args.val_n // 4,
            "easy": args.val_n - (args.val_n // 2) - (args.val_n // 4),
        }
        print(f"[build_hotpot_train_10k] sampling VAL: {val_counts} = {sum(val_counts.values())}", flush=True)
        val_pool_by_level = {
            lvl: [idx for idx in pool if idx not in train_index_set]
            for lvl, pool in by_level.items()
        }
        val_indices = stratified_sample(val_pool_by_level, val_counts, rng)

    # ---- Write parquets ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_records = _rows_to_training_parquet(ds, train_indices, slot="train")
    val_records = _rows_to_training_parquet(ds, val_indices, slot="validation") if val_indices else []

    import pandas as pd
    train_path = args.output_dir / "grpo_hotpot_10k_train.parquet"
    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    print(f"[build_hotpot_train_10k] wrote {len(train_records)} train rows → {train_path}", flush=True)
    if val_records:
        val_path = args.output_dir / "grpo_hotpot_10k_train.val.parquet"
        pd.DataFrame(val_records).to_parquet(val_path, index=False)
        print(f"[build_hotpot_train_10k] wrote {len(val_records)} val rows → {val_path}", flush=True)

    # ---- Write chunks JSONL for BM25 indexing (union of train + val passages) ----
    # The BM25 index needs passages for BOTH train and val so agents can retrieve
    # the right pool at any time. Union them.
    print(f"[build_hotpot_train_10k] generating chunks for {len(train_indices)} train + {len(val_indices)} val questions…", flush=True)
    from telemetry_agent.runners._hotpot_evidence import iter_hotpotqa_chunk_rows_from_example

    args.chunks_jsonl.parent.mkdir(parents=True, exist_ok=True)
    all_indices = list(dict.fromkeys(list(train_indices) + list(val_indices)))  # de-dupe preserving order
    chunk_count = 0
    with args.chunks_jsonl.open("w") as f:
        for idx in all_indices:
            for row in iter_hotpotqa_chunk_rows_from_example(
                ds[idx],
                chunk_words=args.chunk_words,
                chunk_overlap_words=args.chunk_overlap_words,
                prefer_paragraphs=args.prefer_paragraphs,
            ):
                f.write(json.dumps(row) + "\n")
                chunk_count += 1
    print(f"[build_hotpot_train_10k] wrote {chunk_count} chunks → {args.chunks_jsonl}", flush=True)

    # Composition summary
    print("\n[build_hotpot_train_10k] train level composition (actual):")
    train_levels = Counter(str(ds[i]["level"]) for i in train_indices)
    for lvl, cnt in sorted(train_levels.items()):
        print(f"  {lvl}: {cnt}")
    if val_indices:
        print("[build_hotpot_train_10k] val level composition (actual):")
        val_levels = Counter(str(ds[i]["level"]) for i in val_indices)
        for lvl, cnt in sorted(val_levels.items()):
            print(f"  {lvl}: {cnt}")
    return 0


def _rows_to_training_parquet(ds, indices: List[int], slot: str) -> List[Dict[str, Any]]:
    """Wrap each selected HotpotQA row into the training-prompt parquet schema."""
    out: List[Dict[str, Any]] = []
    for idx in indices:
        ex = ds[idx]
        out.append({
            "prompt": _build_prompt(str(ex["question"])),
            "reward_model": {"ground_truth": str(ex["answer"])},
            "question_id": str(ex["id"]),
            "question": str(ex["question"]),
            "gold_answer": str(ex["answer"]),
            "dataset_source": "hotpot",
            "forced_max_turns": None,
            "slot": slot,
            "data_source": "hotpot",
            "level": str(ex.get("level") or "unknown"),
        })
    return out


if __name__ == "__main__":
    sys.exit(main())
