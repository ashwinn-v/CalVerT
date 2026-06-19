#!/usr/bin/env python3
"""
WiTQA data loading, schema adapter, popularity bucketing, empirical tool_needed
labeling, and alias-aware EM / Token-F1 scoring.

Schema: the real WiTQA TSV (megagonlabs/witqa, commit of record 2024-06) has
17 columns per row:

    subject (Q-code), predicate (P-code), object (Q-code), object_surfaceform,
    extracted_text, triple, sub_pred, triple_count, sub_pred_count,
    entity_count, expanded_object, subject_label, predicate_label,
    object_label, expanded_object_label, output_question, s_pop

Mapping to our canonical fields:
    output_question            -> question_text
    object_label  + object_surfaceform + expanded_object_label (list)
                               -> gold_answer (all joined on "|")
    subject_label              -> subject
    predicate_label            -> relation
    extracted_text             -> supporting_passage  (used for oracle)
    sub_pred_count             -> sr_count   (S-R count per paper)
    entity_count               -> s_count    (S count per paper)

`expanded_object_label` is a Python-style list embedded in a TSV cell (e.g.
`["Seton Ingersoll Miller", "Martin Flavin", ...]`). We parse it best-effort.

Gold answers are `|`-separated AFTER we assemble them — we never split on `,`
inside a gold string, so "Washington, D.C." survives.

Empirical `tool_needed`:
    Run Qwen3-8B closed-book 3x at T=0.7 over the full TSV once; cache labels to
    `data/witqa_toolneeded_labels.jsonl`. `tool_needed = not majority_correct`.
    Secondary label from paper: `paper_tool_needed = sr_count < 50`.

Popularity buckets (from `sr_count` = `sub_pred_count`):
    edges = [1, 10, 50, 100, 500, 1000, 5000, inf]
    names = ["1-9", "10-49", "50-99", "100-499", "500-999",
             "1000-4999", "5000+"]
"""
from __future__ import annotations

import ast
import csv
import json
import random
import re
import string
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---------------------------------------------------------------------------
# Answer normalization and alias matching (WiTQA-specific)
# ---------------------------------------------------------------------------

def normalize_witqa(s: str) -> str:
    """WiTQA-specific answer normalization.

    Order:
      1. NFKC unicode normalize (canonicalizes width / compatibility forms)
      2. lowercase
      3. strip punctuation
      4. remove English articles ("a", "an", "the")
      5. collapse whitespace

    Same recipe as QAMPARI `normalize_answer` + NFKC pre-step (Wikipedia entity
    aliases include non-ASCII glyph variants that should fold to ASCII).
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def split_aliases(answer_str: str) -> List[str]:
    """Split a WiTQA gold-answer field on `|` only.

    Commas inside an alias are preserved (e.g., "Washington, D.C.|Washington DC").
    Trimmed; empties dropped.
    """
    if answer_str is None:
        return []
    parts = [p.strip() for p in str(answer_str).split("|")]
    return [p for p in parts if p]


def compute_em(prediction: str, gold_answer: str) -> float:
    """Exact match with alias splitting.

    Returns 1.0 if the normalized prediction equals the normalized canonical
    gold answer OR any of its `|`-separated aliases. Else 0.0.
    """
    norm_pred = normalize_witqa(prediction)
    if not norm_pred:
        return 0.0
    for alias in split_aliases(gold_answer):
        if norm_pred == normalize_witqa(alias):
            return 1.0
    return 0.0


def _token_f1_one(pred_norm: str, gold_norm: str) -> float:
    pred_toks = pred_norm.split()
    gold_toks = gold_norm.split()
    if not pred_toks or not gold_toks:
        return 0.0
    common: Dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    n_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            n_same += 1
            common[t] -= 1
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_token_f1(prediction: str, gold_answer: str) -> float:
    """Max token-level F1 over the prediction vs each `|`-separated alias."""
    pred_norm = normalize_witqa(prediction)
    if not pred_norm:
        return 0.0
    best = 0.0
    for alias in split_aliases(gold_answer):
        gold_norm = normalize_witqa(alias)
        if not gold_norm:
            continue
        f1 = _token_f1_one(pred_norm, gold_norm)
        if f1 > best:
            best = f1
    return best


# ---------------------------------------------------------------------------
# Popularity bucketing
# ---------------------------------------------------------------------------

POPULARITY_BUCKET_EDGES: Tuple[float, ...] = (1, 10, 50, 100, 500, 1000, 5000, float("inf"))
POPULARITY_BUCKET_NAMES: Tuple[str, ...] = (
    "1-9", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000+",
)


def bucket_from_sr_count(sr: Any) -> str:
    """Return the popularity-bucket name for a given S-R_count value.

    Seven buckets per the WiTQA paper: [1, 10, 50, 100, 500, 1000, 5000, inf].
    Missing / non-numeric / zero values clamp to the "1-9" bucket (lowest).
    """
    try:
        v = float(sr)
    except (TypeError, ValueError):
        v = 1.0
    if v < POPULARITY_BUCKET_EDGES[0]:
        return POPULARITY_BUCKET_NAMES[0]
    for i in range(len(POPULARITY_BUCKET_NAMES)):
        lo = POPULARITY_BUCKET_EDGES[i]
        hi = POPULARITY_BUCKET_EDGES[i + 1]
        if lo <= v < hi:
            return POPULARITY_BUCKET_NAMES[i]
    return POPULARITY_BUCKET_NAMES[-1]


def paper_tool_needed(sr_count: Any, threshold: int = 50) -> bool:
    """WiTQA paper's S-R-count crossover heuristic.

    The paper reports retrieval helps below ~50 S-R co-occurrences and starts
    to hurt above; we record this as a secondary (paper-derived) label.
    """
    try:
        return float(sr_count) < threshold
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# TSV loader + schema adapter
# ---------------------------------------------------------------------------

WITQA_TSV_COLUMNS = (
    "output_question",
    "object_label",
    "object_surfaceform",
    "expanded_object_label",
    "subject_label",
    "predicate_label",
    "extracted_text",
    "sub_pred_count",
    "entity_count",
)


def _parse_expanded_list(raw: str) -> List[str]:
    """Parse the `expanded_object_label` cell — a Python-style list in a TSV cell.

    The TSV quotes the list with doubled double-quotes (CSV escaping), so by the
    time DictReader hands us the field it looks like `["A", "B", "C"]`. We
    try json.loads first, then ast.literal_eval, and finally split-on-comma as
    a last resort. Always returns a list (possibly empty) of clean strings.
    """
    if not raw:
        return []
    s = str(raw).strip()
    if not s:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [p for p in parts if p]

CANONICAL_FIELDS_WITQA = (
    "qid",
    "question_text",
    "gold_answer",
    "subject",
    "relation",
    "sr_count",
    "s_count",
    "popularity_bucket",
)


def load_witqa_tsv(
    path: str,
    *,
    n_examples: Optional[int] = None,
    seed: int = 42,
    stratify_by_bucket: bool = False,
) -> List[Dict[str, Any]]:
    """Load the WiTQA TSV and normalize every row to canonical schema.

    Args:
      path: filesystem path to the `.tsv`.
      n_examples: if set, subsample to this many rows.
      seed: RNG seed for subsampling.
      stratify_by_bucket: if True, sample equally across the 8 popularity buckets
          (last bucket takes the remainder if `n_examples` doesn't divide evenly).

    Every returned row has AT MINIMUM the canonical fields:
      qid, question_text, gold_answer, subject, relation, sr_count, s_count,
      popularity_bucket, and supporting_passage (may be empty string).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WiTQA TSV not found: {path}")

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for idx, raw in enumerate(reader):
            if raw is None:
                continue
            for col in WITQA_TSV_COLUMNS:
                if col not in raw:
                    raise ValueError(
                        f"WiTQA TSV missing expected column '{col}' — found "
                        f"columns: {sorted(raw.keys()) if raw else '[]'}"
                    )
            sr_raw = (raw.get("sub_pred_count") or "").strip()
            s_raw = (raw.get("entity_count") or "").strip()
            try:
                sr_count = int(sr_raw) if sr_raw else 0
            except ValueError:
                sr_count = 0
            try:
                s_count = int(s_raw) if s_raw else 0
            except ValueError:
                s_count = 0

            question = (raw.get("output_question") or "").strip()
            object_label = (raw.get("object_label") or "").strip()
            object_surface = (raw.get("object_surfaceform") or "").strip()
            expanded_aliases = _parse_expanded_list(raw.get("expanded_object_label") or "")
            subject = (raw.get("subject_label") or "").strip()
            relation = (raw.get("predicate_label") or "").strip()
            supporting = (raw.get("extracted_text") or "").strip()

            # Assemble gold_answer = "|"-joined dedup alias list. Order: canonical
            # Wikidata label, then passage surface form, then all expanded synonyms.
            alias_list: List[str] = []
            seen: set = set()
            for a in [object_label, object_surface] + expanded_aliases:
                a = a.strip()
                if a and a not in seen:
                    alias_list.append(a)
                    seen.add(a)
            gold_answer = "|".join(alias_list)

            row = {
                "qid": f"witqa_{idx:06d}",
                "question_text": question,
                "gold_answer": gold_answer,
                "subject": subject,
                "relation": relation,
                "supporting_passage": supporting,
                "sr_count": sr_count,
                "s_count": s_count,
                "popularity_bucket": bucket_from_sr_count(sr_count),
                "paper_tool_needed": paper_tool_needed(sr_count),
            }
            rows.append(row)

    if n_examples is None or n_examples >= len(rows):
        return rows

    rng = random.Random(seed)
    if not stratify_by_bucket:
        return rng.sample(rows, n_examples)

    # Stratified sample — equal per bucket, remainder goes to the most-populous.
    by_bucket: Dict[str, List[Dict[str, Any]]] = {name: [] for name in POPULARITY_BUCKET_NAMES}
    for r in rows:
        by_bucket[r["popularity_bucket"]].append(r)
    n_buckets = len(POPULARITY_BUCKET_NAMES)
    per_bucket = n_examples // n_buckets
    overflow = n_examples - per_bucket * n_buckets

    sampled: List[Dict[str, Any]] = []
    for name in POPULARITY_BUCKET_NAMES:
        pool = by_bucket[name]
        take = min(per_bucket, len(pool))
        sampled.extend(rng.sample(pool, take) if take < len(pool) else pool)
    if overflow > 0:
        remaining = [r for r in rows if r not in sampled]
        if remaining:
            take = min(overflow, len(remaining))
            sampled.extend(rng.sample(remaining, take))
    rng.shuffle(sampled)
    return sampled


def assert_witqa_schema(example: Dict[str, Any], *, idx: int = -1) -> None:
    """Fail-fast schema check — all canonical fields must be present, non-null."""
    missing = [f for f in CANONICAL_FIELDS_WITQA if example.get(f) in (None, "")]
    # supporting_passage may be empty string; `sr_count` may be 0. Allow both.
    missing = [f for f in missing if f not in ("sr_count", "s_count")]
    if missing:
        raise ValueError(
            f"WiTQA schema adapter failed at idx={idx} qid={example.get('qid')!r}: "
            f"missing/null fields {missing}. keys={sorted(example.keys())}"
        )


# ---------------------------------------------------------------------------
# Empirical tool_needed labels
# ---------------------------------------------------------------------------

def toolneeded_cache_path(data_dir: str) -> Path:
    return Path(data_dir) / "witqa_toolneeded_labels.jsonl"


def load_toolneeded_labels(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Return {qid: {"tool_needed": bool, "n_correct": int, ...}} or {} if missing."""
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("qid")
            if qid:
                out[str(qid)] = rec
    return out


def attach_toolneeded_labels(
    rows: Sequence[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a new list of rows with `tool_needed` + `toolneeded_n_correct`.

    Rows whose `qid` is missing from `labels` get `tool_needed=None` and are
    logged — the runner can warn-but-continue (or fail, per canary config).
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        label = labels.get(str(r.get("qid")))
        merged = dict(r)
        if label is None:
            merged["tool_needed"] = None
            merged["toolneeded_n_correct"] = None
        else:
            merged["tool_needed"] = bool(label.get("tool_needed"))
            merged["toolneeded_n_correct"] = int(label.get("n_correct", 0))
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# Closed-book prompt (used both by empirical labeler and the closed_book cond)
# ---------------------------------------------------------------------------

CLOSED_BOOK_SYSTEM = (
    "You answer factual questions. Give a short direct answer — one entity, "
    "name, or span. Do not add hedges, explanations, or disclaimers. If you "
    "are not sure, still give your best single-answer guess."
)


def build_closed_book_messages(question_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"Question: {question_text}\nAnswer:"},
    ]


# ---------------------------------------------------------------------------
# Per-bucket aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_by_bucket(
    records: Sequence[Dict[str, Any]],
    *,
    metric_field: str = "em",
) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-row result records by popularity bucket.

    Each record MUST have `popularity_bucket` and `metric_field`. Returns:

        {bucket_name: {"n": int, "mean": float}}
    """
    by_bucket: Dict[str, List[float]] = {name: [] for name in POPULARITY_BUCKET_NAMES}
    for r in records:
        b = r.get("popularity_bucket")
        if b in by_bucket:
            v = r.get(metric_field)
            if v is None:
                continue
            try:
                by_bucket[b].append(float(v))
            except (TypeError, ValueError):
                continue
    out: Dict[str, Dict[str, float]] = {}
    for b, vals in by_bucket.items():
        if not vals:
            out[b] = {"n": 0, "mean": 0.0}
        else:
            out[b] = {"n": len(vals), "mean": sum(vals) / len(vals)}
    return out


# ---------------------------------------------------------------------------
# Simple self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    # split_aliases
    assert split_aliases("Obama|Barack Obama") == ["Obama", "Barack Obama"]
    assert split_aliases("Washington, D.C.|Washington DC") == ["Washington, D.C.", "Washington DC"]
    assert split_aliases("") == []
    # _parse_expanded_list
    assert _parse_expanded_list('["A", "B"]') == ["A", "B"]
    assert _parse_expanded_list("['A','B']") == ["A", "B"]
    assert _parse_expanded_list("") == []
    assert _parse_expanded_list('[]') == []
    # EM with aliases
    assert compute_em("Obama", "Barack Obama|Obama") == 1.0
    assert compute_em("the obama", "Barack Obama|Obama") == 1.0  # articles removed
    assert compute_em("Biden", "Barack Obama|Obama") == 0.0
    # Token F1
    assert compute_token_f1("Barack Obama", "Obama") > 0.0
    assert compute_token_f1("", "Obama") == 0.0
    # Buckets — 7 buckets per paper, zero/missing clamps to lowest ("1-9")
    assert bucket_from_sr_count(0) == "1-9"
    assert bucket_from_sr_count(7) == "1-9"
    assert bucket_from_sr_count(49) == "10-49"
    assert bucket_from_sr_count(50) == "50-99"
    assert bucket_from_sr_count(9999) == "5000+"
    # paper label
    assert paper_tool_needed(7) is True
    assert paper_tool_needed(150) is False
    print("witqa_utils self-test OK")


if __name__ == "__main__":
    _selftest()
