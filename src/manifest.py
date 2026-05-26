"""Reproducibility manifest writer.

For each benchmark run, records: git SHA, Python version, platform, package
versions, run metadata, and aggregate stats. Lives alongside results.jsonl and
makes it possible to tell, months later, exactly what produced a numbers file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any


_INTERESTING_PACKAGES = (
    "chromadb", "sentence-transformers", "rank-bm25", "openai", "pyyaml",
    "matplotlib", "tiktoken", "numpy", "scikit-learn", "umap-learn", "datasets",
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover
        return versions
    for pkg in _INTERESTING_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _aggregate(results: list) -> dict[str, Any]:
    if not results:
        return {}
    n = len(results)

    def _avg(field: str) -> float:
        vals = [getattr(r, field, None) for r in results]
        vals = [v for v in vals if isinstance(v, (int, float)) and v != -1.0]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "n_results": n,
        "unique_configs": len({r.config_name for r in results}),
        "unique_questions": len({r.question_id for r in results}),
        "avg_gold_similarity": _avg("gold_similarity"),
        "avg_fact_recall": _avg("fact_recall"),
        "avg_qasper_f1": _avg("qasper_f1"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_total_latency_ms": _avg("total_latency_ms"),
        "avg_context_tokens": _avg("context_tokens"),
    }


def _requirements_hash(req_path: str = "requirements.txt") -> str | None:
    if not os.path.exists(req_path):
        return None
    with open(req_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_manifest(
    results: list,
    output_path: str,
    config_metadata: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    manifest = {
        "schema_version": "1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "sha": _git_sha(),
            "dirty": _git_dirty(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "requirements_sha256": _requirements_hash(),
        "config_metadata": config_metadata or {},
        "aggregates": _aggregate(results),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
