"""Tests for QASPER F1/EM scorer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qasper_scorer import QasperF1Scorer


def test_exact_match_after_normalization():
    s = QasperF1Scorer()
    m = s.score("The Transformer.", ["A transformer"])
    assert m.exact_match == 1.0
    assert m.f1 == 1.0


def test_partial_overlap_f1():
    s = QasperF1Scorer()
    m = s.score("attention is all you need", ["attention is all"])
    # pred has 5 toks, gold has 3 toks, 3 in common
    # precision = 3/5, recall = 3/3, f1 = 2*0.6*1/1.6 = 0.75
    assert abs(m.f1 - 0.75) < 1e-6
    assert m.exact_match == 0.0


def test_takes_max_over_multiple_golds():
    s = QasperF1Scorer()
    m = s.score("BERT", ["GPT", "BERT"])
    assert m.exact_match == 1.0


def test_empty_pred_zero_f1():
    s = QasperF1Scorer()
    assert s.score("", ["answer"]).f1 == 0.0


def test_both_empty_perfect():
    s = QasperF1Scorer()
    # Both empty -> F1 = 1.0 (matches QASPER convention for unanswerable)
    assert s.score("", [""]).f1 == 1.0
