"""Tests for the QASPER eval-set builder's answer-aggregation logic.

Network-free: no dataset download is involved, only the pure
`_collect_answers` helper that turns one question's raw QASPER answer
block into (chosen_answer, gold_answers, answer_type, spans).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "build_qasper_evalset.py"
)
_spec = importlib.util.spec_from_file_location("build_qasper_evalset", _SCRIPT_PATH)
build_qasper_evalset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_qasper_evalset)

_collect_answers = build_qasper_evalset._collect_answers


def _answer(*, unanswerable=False, yes_no=None, free_form="", spans=None):
    return {
        "unanswerable": unanswerable,
        "yes_no": yes_no,
        "free_form_answer": free_form,
        "extractive_spans": spans or [],
    }


def test_collect_answers_no_entries_returns_none():
    assert _collect_answers({"answer": []}) is None


def test_collect_answers_single_usable_answer():
    ans_block = {"answer": [_answer(free_form="Revenue grew 28%.")]}
    chosen, gold_answers, atype, spans = _collect_answers(ans_block)
    assert chosen == "Revenue grew 28%."
    assert gold_answers == ["Revenue grew 28%."]
    assert atype == "abstractive"


def test_collect_answers_multiple_annotators_all_collected():
    """Every usable annotator answer must end up in gold_answers, not just
    the first one (this is the behavior the fix restores)."""
    ans_block = {
        "answer": [
            _answer(free_form="Revenue grew 28% year over year."),
            _answer(free_form="A 28% YoY increase in revenue."),
        ]
    }
    chosen, gold_answers, atype, spans = _collect_answers(ans_block)
    assert chosen == "Revenue grew 28% year over year."
    assert gold_answers == [
        "Revenue grew 28% year over year.",
        "A 28% YoY increase in revenue.",
    ]


def test_collect_answers_deduplicates_identical_answers():
    ans_block = {
        "answer": [
            _answer(free_form="Yes."),
            _answer(free_form="Yes."),
        ]
    }
    _, gold_answers, _, _ = _collect_answers(ans_block)
    assert gold_answers == ["Yes."]


def test_collect_answers_all_unanswerable_falls_back():
    ans_block = {"answer": [_answer(unanswerable=True), _answer(unanswerable=True)]}
    chosen, gold_answers, atype, spans = _collect_answers(ans_block)
    assert chosen == "Unanswerable"
    assert gold_answers == ["Unanswerable"]
    assert atype == "unanswerable"


def test_collect_answers_skips_unanswerable_annotator_when_others_usable():
    ans_block = {
        "answer": [
            _answer(unanswerable=True),
            _answer(free_form="It is 42.3 million."),
        ]
    }
    chosen, gold_answers, atype, spans = _collect_answers(ans_block)
    assert chosen == "It is 42.3 million."
    assert gold_answers == ["It is 42.3 million."]


if __name__ == "__main__":
    test_collect_answers_no_entries_returns_none()
    test_collect_answers_single_usable_answer()
    test_collect_answers_multiple_annotators_all_collected()
    test_collect_answers_deduplicates_identical_answers()
    test_collect_answers_all_unanswerable_falls_back()
    test_collect_answers_skips_unanswerable_annotator_when_others_usable()
    print("All build_qasper_evalset tests passed!")
