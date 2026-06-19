#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def build_count_vectorizer(min_df: int = 1, max_df: float = 1.0) -> CountVectorizer:
    return CountVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
    )


@dataclass
class SearchHit:
    row_index: int
    score: float
    row: Dict[str, Any]


class BM25Index:
    def __init__(
        self,
        doc_term_matrix: sparse.csr_matrix,
        idf: np.ndarray,
        doc_len: np.ndarray,
        question_offsets: Dict[str, Dict[str, float]],
        rows: Sequence[Mapping[str, Any]],
        vocabulary: Mapping[str, int],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.doc_term_matrix = doc_term_matrix.tocsr()
        self.idf = np.asarray(idf, dtype=np.float32)
        self.doc_len = np.asarray(doc_len, dtype=np.float32)
        self.question_offsets = question_offsets
        self.rows = [dict(row) for row in rows]
        self.vocabulary = dict(vocabulary)
        self.k1 = float(k1)
        self.b = float(b)

    @classmethod
    def build(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        text_field: str = "chunk_text",
        min_df: int = 1,
        max_df: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        if not rows:
            raise ValueError("Cannot build BM25 index with no rows.")

        texts = [str(row.get(text_field) or "") for row in rows]
        vectorizer = build_count_vectorizer(min_df=min_df, max_df=max_df)
        doc_term_matrix = vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        doc_len = np.asarray(doc_term_matrix.sum(axis=1)).reshape(-1).astype(np.float32)
        df = np.asarray((doc_term_matrix > 0).sum(axis=0)).reshape(-1).astype(np.float32)
        n_docs = doc_term_matrix.shape[0]
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        question_offsets: Dict[str, Dict[str, float]] = {}
        start = 0
        while start < len(rows):
            question_id = str(rows[start].get("question_id") or "")
            end = start + 1
            while end < len(rows) and str(rows[end].get("question_id") or "") == question_id:
                end += 1
            q_doc_len = doc_len[start:end]
            avgdl = float(q_doc_len.mean()) if len(q_doc_len) else 0.0
            question_offsets[question_id] = {
                "start": start,
                "end": end,
                "count": end - start,
                "avgdl": avgdl,
            }
            start = end

        return cls(
            doc_term_matrix=doc_term_matrix,
            idf=idf,
            doc_len=doc_len,
            question_offsets=question_offsets,
            rows=rows,
            vocabulary=vectorizer.vocabulary_,
            k1=k1,
            b=b,
        )

    def save(self, output_dir: Path, metadata: Mapping[str, Any] | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(output_dir / "doc_term_matrix.npz", self.doc_term_matrix)
        np.save(output_dir / "idf.npy", self.idf)
        np.save(output_dir / "doc_len.npy", self.doc_len)

        with (output_dir / "vocabulary.json").open("w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, indent=2, ensure_ascii=True, sort_keys=True)

        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        index_meta = {
            "k1": self.k1,
            "b": self.b,
            "question_offsets": self.question_offsets,
            "row_count": len(self.rows),
            "vocab_size": len(self.vocabulary),
        }
        if metadata:
            index_meta.update(dict(metadata))
        with (output_dir / "index_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(index_meta, f, indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, output_dir: Path) -> "BM25Index":
        doc_term_matrix = sparse.load_npz(output_dir / "doc_term_matrix.npz").tocsr()
        idf = np.load(output_dir / "idf.npy")
        doc_len = np.load(output_dir / "doc_len.npy")

        with (output_dir / "vocabulary.json").open("r", encoding="utf-8") as f:
            vocabulary = json.load(f)
        with (output_dir / "index_metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        rows: List[Dict[str, Any]] = []
        with (output_dir / "rows.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        return cls(
            doc_term_matrix=doc_term_matrix,
            idf=idf,
            doc_len=doc_len,
            question_offsets=metadata["question_offsets"],
            rows=rows,
            vocabulary=vocabulary,
            k1=float(metadata.get("k1", 1.5)),
            b=float(metadata.get("b", 0.75)),
        )

    def score_question(self, question_id: str, query: str) -> np.ndarray:
        offset = self.question_offsets.get(question_id)
        if offset is None:
            raise KeyError(f"Question id '{question_id}' not found in index.")

        query_tokens = tokenize(query)
        if not query_tokens:
            size = int(offset["end"] - offset["start"])
            return np.zeros(size, dtype=np.float32)

        query_term_freqs: Dict[int, int] = {}
        for token in query_tokens:
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            query_term_freqs[term_id] = query_term_freqs.get(term_id, 0) + 1

        size = int(offset["end"] - offset["start"])
        scores = np.zeros(size, dtype=np.float32)
        if not query_term_freqs:
            return scores

        start = int(offset["start"])
        end = int(offset["end"])
        avgdl = float(offset["avgdl"]) if offset["avgdl"] else 1.0
        local_lengths = self.doc_len[start:end]

        for term_id, qtf in query_term_freqs.items():
            term_counts = self.doc_term_matrix[start:end, term_id].toarray().reshape(-1)
            if not np.any(term_counts):
                continue
            denom = term_counts + self.k1 * (1.0 - self.b + self.b * local_lengths / max(avgdl, 1e-6))
            scores += self.idf[term_id] * ((term_counts * (self.k1 + 1.0)) / np.maximum(denom, 1e-6)) * qtf

        return scores

    def search(self, question_id: str, query: str, top_k: int = 5) -> List[SearchHit]:
        offset = self.question_offsets.get(question_id)
        if offset is None:
            raise KeyError(f"Question id '{question_id}' not found in index.")

        scores = self.score_question(question_id, query)
        size = scores.shape[0]
        if size == 0 or top_k <= 0:
            return []

        top_k = min(top_k, size)
        local_indices = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        local_indices = local_indices[np.argsort(-scores[local_indices], kind="stable")]
        start = int(offset["start"])

        hits: List[SearchHit] = []
        for local_idx in local_indices:
            row_index = start + int(local_idx)
            hits.append(
                SearchHit(
                    row_index=row_index,
                    score=float(scores[local_idx]),
                    row=self.rows[row_index],
                )
            )
        return hits

    def score_global(self, query: str) -> np.ndarray:
        """Score the query against EVERY row in the index (ignoring offsets).

        Used by benchmarks whose qids aren't present in `question_offsets`
        (WiTQA, OverSearchQA). Assumes the index was built over a shared
        corpus (e.g., Wikipedia chunks) rather than per-question candidate
        pools. Returns a 1-D array of length `n_docs`.
        """
        n_docs = self.doc_term_matrix.shape[0]
        query_tokens = tokenize(query)
        if not query_tokens:
            return np.zeros(n_docs, dtype=np.float32)
        query_term_freqs: Dict[int, int] = {}
        for token in query_tokens:
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            query_term_freqs[term_id] = query_term_freqs.get(term_id, 0) + 1
        scores = np.zeros(n_docs, dtype=np.float32)
        if not query_term_freqs:
            return scores
        avgdl = float(self.doc_len.mean()) if self.doc_len.size else 1.0
        if avgdl <= 0:
            avgdl = 1.0
        for term_id, qtf in query_term_freqs.items():
            term_counts = self.doc_term_matrix[:, term_id].toarray().reshape(-1)
            if not np.any(term_counts):
                continue
            denom = term_counts + self.k1 * (
                1.0 - self.b + self.b * self.doc_len / avgdl
            )
            scores += self.idf[term_id] * (
                (term_counts * (self.k1 + 1.0)) / np.maximum(denom, 1e-6)
            ) * qtf
        return scores

    def search_global(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """Top-k over the full corpus, ignoring per-question offsets.

        Required for WiTQA and OverSearchQA, whose qids were never registered
        in the QAMPARI-built wikipedia_bm25 index. The index must have been
        built over a shared corpus (not per-question candidate pools) for the
        results to be meaningful.
        """
        if top_k <= 0:
            return []
        scores = self.score_global(query)
        size = scores.shape[0]
        if size == 0:
            return []
        top_k = min(top_k, size)
        local_indices = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        local_indices = local_indices[np.argsort(-scores[local_indices], kind="stable")]
        hits: List[SearchHit] = []
        for local_idx in local_indices:
            row_index = int(local_idx)
            hits.append(
                SearchHit(
                    row_index=row_index,
                    score=float(scores[local_idx]),
                    row=self.rows[row_index],
                )
            )
        return hits
