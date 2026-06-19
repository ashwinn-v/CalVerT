#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parent


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> int:
    ensure_parent_dir(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def slugify_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe or "unnamed"


def limit_suffix(limit: Optional[int]) -> str:
    return "all" if limit is None else f"n{limit}"


def normalize_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None or limit <= 0:
        return None
    return limit


def dict_of_lists_to_records(table: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(table, Mapping):
        return []

    list_lengths = [
        len(value)
        for value in table.values()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ]
    if not list_lengths:
        return []

    n_rows = max(list_lengths)
    records: List[Dict[str, Any]] = []
    for row_idx in range(n_rows):
        row: Dict[str, Any] = {}
        for key, values in table.items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                row[key] = values[row_idx] if row_idx < len(values) else None
            else:
                row[key] = values
        records.append(row)
    return records


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_paragraphs(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if len(parts) > 1:
        return parts

    line_parts = [line.strip() for line in text.splitlines() if line.strip()]
    if len(line_parts) > 1:
        return line_parts

    return [text]


def word_chunks(text: str, max_words: int, overlap_words: int) -> List[Dict[str, Any]]:
    words = normalize_whitespace(text).split()
    if not words:
        return []

    if max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words}")

    step = max(1, max_words - max(0, overlap_words))
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(
            {
                "text": " ".join(chunk_words),
                "start_word": start,
                "end_word": end,
                "word_count": len(chunk_words),
                "split_type": "sliding_window",
            }
        )
        if end >= len(words):
            break
    return chunks


def chunk_text(
    text: str,
    max_words: int = 120,
    overlap_words: int = 40,
    prefer_paragraphs: bool = True,
) -> List[Dict[str, Any]]:
    normalized = (text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []

    if not prefer_paragraphs:
        return word_chunks(normalized, max_words=max_words, overlap_words=overlap_words)

    paragraphs = split_paragraphs(normalized)
    if len(paragraphs) <= 1:
        return word_chunks(normalized, max_words=max_words, overlap_words=overlap_words)

    chunks: List[Dict[str, Any]] = []
    offset = 0
    for paragraph in paragraphs:
        paragraph_words = normalize_whitespace(paragraph).split()
        if not paragraph_words:
            continue

        if len(paragraph_words) <= max_words:
            chunks.append(
                {
                    "text": " ".join(paragraph_words),
                    "start_word": offset,
                    "end_word": offset + len(paragraph_words),
                    "word_count": len(paragraph_words),
                    "split_type": "paragraph",
                }
            )
        else:
            for piece in word_chunks(paragraph, max_words=max_words, overlap_words=overlap_words):
                piece = dict(piece)
                piece["start_word"] += offset
                piece["end_word"] += offset
                chunks.append(piece)
        offset += len(paragraph_words)

    return chunks


def maybe_select_subset(dataset: Any, limit: Optional[int], shuffle: bool, seed: int, start_index: int = 0) -> Any:
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if start_index or limit is not None:
        start = max(0, int(start_index or 0))
        end = len(dataset) if limit is None else min(start + int(limit), len(dataset))
        dataset = dataset.select(range(start, end))
    return dataset


def file_stem_for_subset(dataset_config: str, split: str, seed: int, limit: Optional[int]) -> str:
    return f"triviaqa_{slugify_name(dataset_config)}_{slugify_name(split)}_s{seed}_{limit_suffix(limit)}"


def summarize_counts(values: Sequence[Any]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for value in values:
        key = str(value)
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items(), key=lambda item: item[0]))


def load_hf_dataset_split(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    dataset_path: Optional[str] = None,
) -> Any:
    try:
        from datasets import load_dataset, load_from_disk
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            "Missing dependency 'datasets'. Run this script from the experiment environment."
        ) from exc

    if dataset_path:
        local_path = Path(dataset_path)
        split_path = local_path / split
        if split_path.exists() and split_path.is_dir():
            try:
                return load_from_disk(str(split_path))
            except OSError:
                return load_from_disk(str(split_path), keep_in_memory=True)
        try:
            dataset = load_from_disk(str(local_path))
        except OSError:
            dataset = load_from_disk(str(local_path), keep_in_memory=True)
    else:
        dataset = load_dataset(dataset_name, dataset_config)

    if split not in dataset:
        available = sorted(dataset.keys()) if hasattr(dataset, "keys") else []
        raise ValueError(f"Split '{split}' not found. Available splits: {available}")
    return dataset[split]
