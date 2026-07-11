"""Tests for the reproducibility manifest writer."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.manifest import write_manifest


class _FakeResult:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_manifest_basic_shape(tmp_path):
    results = [
        _FakeResult(
            config_name="paragraph__vector__k3",
            question_id="q1",
            gold_similarity=0.8,
            fact_recall=0.5,
            qasper_f1=-1.0,
            faithfulness=-1.0,
            total_latency_ms=1000.0,
            context_tokens=600,
        ),
        _FakeResult(
            config_name="paragraph__vector__k3",
            question_id="q2",
            gold_similarity=0.9,
            fact_recall=0.7,
            qasper_f1=0.42,
            faithfulness=0.9,
            total_latency_ms=1100.0,
            context_tokens=620,
        ),
    ]
    out = tmp_path / "manifest.json"
    write_manifest(
        results=results,
        output_path=str(out),
        config_metadata={"include_raptor": True},
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"
    assert "git" in data and "sha" in data["git"]
    assert "packages" in data
    assert data["config_metadata"]["include_raptor"] is True
    agg = data["aggregates"]
    assert agg["n_results"] == 2
    assert abs(agg["avg_gold_similarity"] - 0.85) < 1e-6
    # qasper_f1 only populated for one row -> average is 0.42
    assert abs(agg["avg_qasper_f1"] - 0.42) < 1e-6


def test_manifest_handles_empty_results(tmp_path):
    out = tmp_path / "manifest.json"
    write_manifest(results=[], output_path=str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["aggregates"] == {}
