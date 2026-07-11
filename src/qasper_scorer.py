"""QASPER token-level F1 and Exact Match scorer.

Implements the official QASPER metric (Allen AI). For each question, the gold
answer is a list of acceptable answers — the score is the max over each gold
answer. F1 is computed at the token level after normalization (lowercase,
remove articles, remove punctuation, collapse whitespace), matching the
metric reported in the RAPTOR paper (Sarthi et al. 2024, Table 2).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass
class QasperMetrics:
    f1: float
    exact_match: float


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _f1(pred: str, gold: str) -> float:
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)

    # Handle "Unanswerable" / yes-no edge case: if both are empty, F1=1.
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _em(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


class QasperF1Scorer:
    """Compute F1 and EM against one or more acceptable gold answers."""

    def score(
        self, predicted_answer: str, gold_answers: list[str] | str
    ) -> QasperMetrics:
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        if not gold_answers:
            return QasperMetrics(f1=0.0, exact_match=0.0)
        f1 = max(_f1(predicted_answer, g) for g in gold_answers)
        em = max(_em(predicted_answer, g) for g in gold_answers)
        return QasperMetrics(f1=f1, exact_match=em)
