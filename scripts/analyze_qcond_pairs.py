"""Paired per-question comparison of qcond v2 against collapsed search.

Pairs rows by (question_id, top_k) across two results files produced with
the same generator and protocol, then reports per-category and per-k mean
deltas, win/tie/loss counts, and a paired-bootstrap 95% CI on the overall
mean delta. Free — analyzes existing results, no LLM calls.

Usage:
  python scripts/analyze_qcond_pairs.py results/raw/results_qcond2_qasper.jsonl \
      results/raw/results_qasper.jsonl --metric qasper_f1
"""

import json
import random
from collections import defaultdict

import click


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _key(row):
    return (row["question_id"], row["config_name"].rsplit("__k", 1)[1])


def _bootstrap_ci(deltas, n_resamples=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
        for _ in range(n_resamples)
    )
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


@click.command()
@click.argument("qcond_results", type=click.Path(exists=True))
@click.argument("reference_results", type=click.Path(exists=True))
@click.option("--metric", default="gold_similarity", show_default=True)
@click.option("--reference-prefix", default="raptor_100__raptor_collapsed",
              show_default=True, help="Config-name prefix of the comparator.")
def main(qcond_results, reference_results, metric, reference_prefix):
    """Report paired qcond-minus-reference deltas on METRIC."""
    qcond = {_key(r): r for r in _load(qcond_results)
             if "qcond" in r["config_name"]}
    ref = {_key(r): r for r in _load(reference_results)
           if r["config_name"].startswith(reference_prefix)}

    by_cat, by_k, all_deltas = defaultdict(list), defaultdict(list), []
    for k, row in qcond.items():
        if k not in ref:
            continue
        delta = row[metric] - ref[k][metric]
        by_cat[row["question_category"]].append(delta)
        by_k[k[1]].append(delta)
        all_deltas.append(delta)

    def wtl(deltas):
        wins = sum(1 for d in deltas if d > 1e-9)
        losses = sum(1 for d in deltas if d < -1e-9)
        return f"{wins}/{len(deltas) - wins - losses}/{losses}"

    print(f"qcond minus {reference_prefix} on {metric} "
          f"({len(all_deltas)} paired rows)\n")
    for cat in sorted(by_cat):
        ds = by_cat[cat]
        print(f"  {cat:24} n={len(ds):3}  mean={sum(ds)/len(ds):+.4f}  "
              f"win/tie/loss={wtl(ds)}")
    for k in sorted(by_k, key=int):
        ds = by_k[k]
        print(f"  k={k:<23} n={len(ds):3}  mean={sum(ds)/len(ds):+.4f}")
    lo, hi = _bootstrap_ci(all_deltas)
    print(f"\n  OVERALL mean={sum(all_deltas)/len(all_deltas):+.4f}  "
          f"95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]  "
          f"win/tie/loss={wtl(all_deltas)}")


if __name__ == "__main__":
    main()
