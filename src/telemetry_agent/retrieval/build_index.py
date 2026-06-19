#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from telemetry_agent._common import BASE_DIR, summarize_counts, write_json
from telemetry_agent.retrieval.bm25_index import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BM25 retrieval index over chunk JSONL.")
    parser.add_argument(
        "--chunks_jsonl",
        type=str,
        required=True,
        help="Path to chunk JSONL from a corpus builder such as build_triviaqa_rc_corpus.py or build_hotpotqa_corpus.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to data/<chunks_stem>_bm25_index",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="chunk_text",
        choices=["chunk_text", "chunk_body_text"],
    )
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=1.0)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def reorder_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("question_id") or ""),
            int(row.get("_source_line", 0)),
        ),
    )
    for row in rows:
        row.pop("_source_line", None)
    return rows


def default_output_dir(chunks_jsonl: Path) -> Path:
    stem = chunks_jsonl.stem
    return BASE_DIR / "data" / f"{stem}_bm25_index"


def main() -> None:
    args = parse_args()
    chunks_jsonl = Path(args.chunks_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(chunks_jsonl)

    rows = load_rows(chunks_jsonl)
    if not rows:
        raise ValueError(f"No chunk rows found in {chunks_jsonl}")
    rows = reorder_rows(rows)

    index = BM25Index.build(
        rows,
        text_field=args.text_field,
        min_df=args.min_df,
        max_df=args.max_df,
        k1=args.k1,
        b=args.b,
    )

    question_counts = {
        question_id: int(meta["count"]) for question_id, meta in index.question_offsets.items()
    }
    metadata = {
        "index_type": "bm25",
        "chunks_jsonl": str(chunks_jsonl),
        "text_field": args.text_field,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "k1": args.k1,
        "b": args.b,
    }
    index.save(output_dir, metadata=metadata)

    summary = {
        **metadata,
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "question_count": len(index.question_offsets),
        "vocab_size": len(index.vocabulary),
        "avg_docs_per_question": (len(rows) / len(index.question_offsets)) if index.question_offsets else 0.0,
        "question_doc_count_histogram": summarize_counts(question_counts.values()),
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Wrote BM25 index to {output_dir}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
