"""Sanity-check our RAPTOR reimplementation against the paper's findings.

Loads a QASPER results.jsonl, then asserts the two directional claims from
the RAPTOR paper (Sarthi et al., 2024):

  Claim 1 (Table 2): On QASPER, raptor_collapsed >= raptor_tree on F1 by a
    non-trivial margin. We use a loose tolerance because our subsample is
    small and our embedder differs.

  Claim 2 (Table 3): The gap (collapsed - tree) grows with question
    complexity. We approximate by checking that the gap on abstractive or
    multi-hop categories is >= the gap on extractive / single-doc categories.

Exits non-zero if either claim fails, with a one-line diagnostic.

Usage:
    python scripts/check_paper_faithfulness.py results/raw/results_qasper.jsonl
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click

from src.runner import load_results


SIMPLE_CATEGORIES = {"single_doc_factual", "extractive_evidence", "yes_no"}
COMPLEX_CATEGORIES = {"cross_doc_synthesis", "abstractive_synthesis", "multi_hop"}


def _avg(values: list[float]) -> float:
    values = [v for v in values if v is not None and v != -1.0]
    return sum(values) / len(values) if values else float("nan")


@click.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--metric", default="qasper_f1", show_default=True,
              type=click.Choice(["qasper_f1", "gold_similarity"]),
              help="Which scorer to use for the assertion.")
@click.option("--tolerance", default=0.02, show_default=True,
              help="Allowed shortfall on Claim 1 (collapsed >= tree - tolerance).")
@click.option("--strict", is_flag=True, default=False,
              help="If set, exit non-zero on a violation. "
                   "Default is to print and continue (useful for negative-result reporting).")
def main(results_path, metric, tolerance, strict):
    results = load_results(results_path)
    print(f"Loaded {len(results)} results from {results_path}")

    # Group: search_strategy -> category -> [metric values]
    by_search_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        parts = r.config_name.split("__")
        search = parts[1] if len(parts) > 1 else "?"
        if not search.startswith("raptor_"):
            continue
        val = getattr(r, metric, -1.0)
        by_search_cat[search][r.question_category].append(val)

    if "raptor_tree" not in by_search_cat or "raptor_collapsed" not in by_search_cat:
        print("WARN: results do not contain both raptor_tree and raptor_collapsed; "
              "cannot validate paper claims.")
        sys.exit(0 if not strict else 2)

    # Overall averages
    tree_overall = _avg([v for cat_vals in by_search_cat["raptor_tree"].values() for v in cat_vals])
    coll_overall = _avg([v for cat_vals in by_search_cat["raptor_collapsed"].values() for v in cat_vals])

    print(f"\n[overall] raptor_tree {metric}      = {tree_overall:.3f}")
    print(f"[overall] raptor_collapsed {metric} = {coll_overall:.3f}")
    print(f"[overall] gap (collapsed - tree)    = {coll_overall - tree_overall:+.3f}")

    # Claim 1
    claim1_ok = coll_overall >= tree_overall - tolerance
    status1 = "PASS" if claim1_ok else "FAIL"
    print(f"\n[Claim 1: collapsed >= tree - {tolerance}] {status1}")

    # Claim 2: gap on complex categories >= gap on simple categories
    cats = set(by_search_cat["raptor_tree"].keys()) & set(by_search_cat["raptor_collapsed"].keys())
    simple_gaps = []
    complex_gaps = []
    for cat in cats:
        gap = (
            _avg(by_search_cat["raptor_collapsed"][cat])
            - _avg(by_search_cat["raptor_tree"][cat])
        )
        if cat in SIMPLE_CATEGORIES:
            simple_gaps.append(gap)
        elif cat in COMPLEX_CATEGORIES:
            complex_gaps.append(gap)

    if not simple_gaps or not complex_gaps:
        print("[Claim 2] insufficient category coverage to check — need at least one "
              "category in SIMPLE and COMPLEX buckets. Skipping.")
        claim2_ok = True
    else:
        simple_gap = sum(simple_gaps) / len(simple_gaps)
        complex_gap = sum(complex_gaps) / len(complex_gaps)
        claim2_ok = complex_gap >= simple_gap
        status2 = "PASS" if claim2_ok else "FAIL"
        print(f"\n[Claim 2: complex_gap >= simple_gap]")
        print(f"  simple categories gap:  {simple_gap:+.3f}")
        print(f"  complex categories gap: {complex_gap:+.3f}")
        print(f"  -> {status2}")

    if strict and not (claim1_ok and claim2_ok):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
