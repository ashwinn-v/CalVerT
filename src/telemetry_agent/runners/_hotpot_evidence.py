#!/usr/bin/env python3
from __future__ import annotations

import re
import string
from typing import Any, Dict, Iterator, List, Mapping

from common import chunk_text, dict_of_lists_to_records, normalize_whitespace

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_hotpot_answer(text: str) -> str:
    lowered = (text or "").lower().translate(_PUNCT_TABLE)
    without_articles = _ARTICLES_RE.sub(" ", lowered)
    return normalize_whitespace(without_articles)


def supporting_sentence_ids_by_title(example: Mapping[str, Any]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for record in dict_of_lists_to_records(example.get("supporting_facts", {})):
        title = normalize_whitespace(str(record.get("title") or ""))
        if not title:
            continue
        try:
            sent_id = int(record.get("sent_id"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(title, []).append(sent_id)

    return {title: sorted(set(sentence_ids)) for title, sentence_ids in grouped.items()}


def iter_hotpotqa_doc_records(example: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    supporting_by_title = supporting_sentence_ids_by_title(example)
    context_records = dict_of_lists_to_records(example.get("context", {}))

    for doc_index, doc in enumerate(context_records):
        raw_title = normalize_whitespace(str(doc.get("title") or ""))
        title = raw_title or f"context_{doc_index}"
        raw_sentences = doc.get("sentences") or []
        if isinstance(raw_sentences, list):
            sentences = [
                text
                for text in (normalize_whitespace(str(sentence)) for sentence in raw_sentences)
                if text
            ]
        else:
            text = normalize_whitespace(str(raw_sentences))
            sentences = [text] if text else []

        supporting_sentence_ids = supporting_by_title.get(raw_title, [])
        yield {
            "doc_collection": "context",
            "doc_source": "hotpot_context",
            "doc_index": doc_index,
            "doc_id": f"{title}::{doc_index}",
            "filename": "",
            "title": title,
            "rank": None,
            "url": None,
            "description": None,
            "context_text": " ".join(sentences),
            "sentence_count": len(sentences),
            "supporting_sentence_ids": supporting_sentence_ids,
            "supporting_sentence_count": len(supporting_sentence_ids),
            "is_supporting_doc": bool(supporting_sentence_ids),
        }


def iter_hotpotqa_chunk_rows_from_example(
    example: Mapping[str, Any],
    *,
    chunk_words: int,
    chunk_overlap_words: int,
    prefer_paragraphs: bool,
) -> Iterator[Dict[str, Any]]:
    question_id = str(example.get("_id") or example.get("id") or example.get("question_id") or "")
    question = str(example.get("question") or "")
    answer_value = normalize_whitespace(str(example.get("answer") or ""))
    answer_aliases = [answer_value] if answer_value else []
    answer_normalized_aliases = [normalize_hotpot_answer(answer_value)] if answer_value else []
    question_type = str(example.get("type") or "")
    question_level = str(example.get("level") or "")
    supporting_titles = list(supporting_sentence_ids_by_title(example).keys())

    for doc in iter_hotpotqa_doc_records(example):
        context_text = normalize_whitespace(doc["context_text"])
        if not context_text:
            continue

        title = normalize_whitespace(doc["title"]) or "(untitled)"
        chunks = chunk_text(
            context_text,
            max_words=chunk_words,
            overlap_words=chunk_overlap_words,
            prefer_paragraphs=prefer_paragraphs,
        )
        for chunk_index, chunk in enumerate(chunks):
            body_text = normalize_whitespace(chunk["text"])
            if not body_text:
                continue

            chunk_id = f"{question_id}:{doc['doc_collection']}:{doc['doc_index']}:{chunk_index}"
            full_text = f"Title: {title}\n\n{body_text}"
            yield {
                "question_id": question_id,
                "question": question,
                "answer_value": answer_value,
                "answer_aliases": answer_aliases,
                "answer_normalized_aliases": answer_normalized_aliases,
                "question_type": question_type,
                "question_level": question_level,
                "supporting_titles": supporting_titles,
                "doc_collection": str(doc["doc_collection"]),
                "doc_source": str(doc["doc_source"]),
                "doc_index": int(doc["doc_index"]),
                "doc_id": str(doc["doc_id"]),
                "title": title,
                "filename": str(doc["filename"]),
                "rank": doc["rank"],
                "url": doc["url"],
                "description": doc["description"],
                "sentence_count": int(doc["sentence_count"]),
                "supporting_sentence_ids": list(doc["supporting_sentence_ids"]),
                "supporting_sentence_count": int(doc["supporting_sentence_count"]),
                "is_supporting_doc": bool(doc["is_supporting_doc"]),
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_start_word": int(chunk["start_word"]),
                "chunk_end_word": int(chunk["end_word"]),
                "chunk_word_count": int(chunk["word_count"]),
                "chunk_split_type": str(chunk["split_type"]),
                "chunk_text": full_text,
                "chunk_body_text": body_text,
            }
