"""HotpotQA scoring + prompt formatters.

`normalize_answer`, `f1_score`, `em_score` are ports of the official
`hotpot_evaluate_v1.py` functions. Do not substitute generic strip/lower —
those undercount EM by 5–10 points.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Official HotpotQA scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', t)

    def white_space_fix(t: str) -> str:
        return ' '.join(t.split())

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)

    def lower(t: str) -> str:
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> Tuple[float, float, float]:
    """Returns (f1, precision, recall)."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return (float(pred_tokens == gold_tokens), 0.0, 0.0)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return (0.0, 0.0, 0.0)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_context(context: Dict[str, List], exclude_titles: Optional[set] = None) -> str:
    """HotpotQA context dict → '{title}: {paragraph}\n\n...' string.

    `context` schema (HF distractor config):
      {"title": [t0, t1, ..., t9], "sentences": [[s0, s1, ...], [...], ...]}

    If `exclude_titles` is provided, paragraphs with those titles are dropped
    (used to simulate imperfect retrieval by removing gold passages).
    """
    exclude_titles = exclude_titles or set()
    titles = context['title']
    sentences = context['sentences']
    paras = []
    for title, sents in zip(titles, sentences):
        if title in exclude_titles:
            continue
        para = ' '.join(sents).strip()
        paras.append(f"{title}: {para}")
    return "\n\n".join(paras)


GENERATOR_PROMPT = """Read the following context paragraphs and answer the question with a concise factoid answer (a name, date, or short phrase). If the question is yes/no, answer "yes" or "no". Output only the answer, no explanation.

Context:
{context}

Question: {question}

Answer:"""


VALIDATOR_PROMPT = """Below is a question, supporting context, and a candidate answer. Determine whether the candidate answer is correct. Output only "Yes" or "No".

Context:
{context}

Question: {question}
Candidate answer: {candidate_answer}
Is the candidate answer correct?"""


def build_generator_messages(question: str, context_str: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": GENERATOR_PROMPT.format(question=question, context=context_str)}]


def build_validator_messages(question: str, context_str: str, candidate: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": VALIDATOR_PROMPT.format(
        question=question, context=context_str, candidate_answer=candidate)}]
