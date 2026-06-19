"""Smoke test for the BM25 index per-question search."""
from __future__ import annotations

from telemetry_agent.retrieval.bm25_index import BM25Index


def test_per_question_top_hit_matches_intent():
    rows = [
        {"chunk_id": "0", "question_id": "q1", "chunk_text": "The Eiffel Tower is in Paris."},
        {"chunk_id": "1", "question_id": "q1", "chunk_text": "Notre Dame is also in Paris."},
        {"chunk_id": "2", "question_id": "q2", "chunk_text": "The Great Wall is in China."},
    ]
    idx = BM25Index.build(rows=rows)
    hits = idx.search(question_id="q1", query="Eiffel Tower Paris", top_k=2)
    assert hits, "expected at least one BM25 hit for question q1"
    # The top hit for q1 should be the Eiffel Tower chunk.
    assert "Eiffel" in str(hits[0].row.get("chunk_text", ""))


def test_search_isolates_to_question():
    rows = [
        {"chunk_id": "0", "question_id": "q1", "chunk_text": "alpha bravo charlie"},
        {"chunk_id": "1", "question_id": "q2", "chunk_text": "alpha bravo charlie delta echo"},
    ]
    idx = BM25Index.build(rows=rows)
    hits = idx.search(question_id="q1", query="delta echo", top_k=5)
    # Only chunk 0 belongs to q1 even though chunk 1 has the better term overlap.
    assert all(h.row.get("question_id") == "q1" for h in hits)
