"""Free (no-LLM) calibration sweep for raptor_qcond hyperparameters.

Retrieval quality against gold sources is deterministic, so qcond's
mechanisms can be calibrated without paying for answer generation: for
every hyperparameter combination we retrieve on the SYNTHETIC eval set
only (QASPER is held out as the one-shot test set), score context
precision/recall with the standard RetrievalScorer, and track the token
budget. The winning setting becomes QCondConfig's defaults.

Requires the synthetic RAPTOR tree to be cached (any prior benchmark run
with --include-raptor); the tree is loaded from data/raptor_cache/ and
no LLM calls are made.
"""

import csv
import itertools
import os
import sys

import click
import tiktoken
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    CORPUS_DIR, EVAL_SET_PATH, RESULTS_DIR, SearchStrategy, LLM_PRESETS,
)
from src.raptor.tree_builder import RaptorBuildConfig, RaptorTreeBuilder
from src.raptor.tree_retriever import QCondConfig, RaptorRetriever
from src.scorers import RetrievalScorer

TOP_KS = (3, 5, 10)
# Termination-shaping mechanisms (the README's candidate fixes)...
TAU_LEVEL_SCALES = (0.0, 0.05)
LEAF_BIASES = (0.0, 0.05)
EXPAND = (False, True)
# ...plus the pre-existing beam-width knobs, which diagnostics showed to be
# the binding constraint: v1 explores <=8 of ~150 leaves and returns only
# 3-8 candidates at k=10, so recall saturates regardless of termination
# tuning. Calibrating them is part of the same protocol.
K_BRANCHES = (2, 3, 5)
TAU_FOCUSES = (0.6, 0.85)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _retrieval_row(retriever, strategy, cfg_label, questions, scorer, top_k):
    """Mean precision/recall/tokens for one retriever setting at one k."""
    precs, recs, toks = [], [], []
    for q in questions:
        result = retriever.retrieve(q["question"], strategy=strategy, top_k=top_k)
        m = scorer.score(
            retrieved_chunks=result.chunks,
            gold_source_ids=q["gold_source_ids"],
            distractor_ids=q.get("distractors"),
        )
        precs.append(m.context_precision)
        recs.append(m.context_recall)
        toks.append(sum(len(_tokenizer.encode(c.text)) for c in result.chunks))
    n = len(questions)
    return {
        "setting": cfg_label,
        "top_k": top_k,
        "recall": sum(recs) / n,
        "precision": sum(precs) / n,
        "tokens": sum(toks) / n,
    }


@click.command()
@click.option("--summarizer", default="anthropic",
              type=click.Choice(list(LLM_PRESETS)), show_default=True,
              help="Preset whose model name keys the cached tree to load.")
@click.option("--out", default=os.path.join(RESULTS_DIR, "qcond_sweep.csv"),
              show_default=True)
def main(summarizer, out):
    """Sweep qcond hyperparameters on the synthetic eval set (no LLM calls)."""
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    print("Loading cached RAPTOR tree (no LLM calls expected)...")
    builder = RaptorTreeBuilder(
        build_config=RaptorBuildConfig(), llm_config=LLM_PRESETS[summarizer]
    )
    index = builder.build(CORPUS_DIR)
    scorer = RetrievalScorer()

    rows = []
    # Reference: the paper's collapsed mode at each k.
    ref = RaptorRetriever(index)
    for top_k in TOP_KS:
        rows.append(_retrieval_row(
            ref, SearchStrategy.RAPTOR_COLLAPSED, "REF_collapsed",
            questions, scorer, top_k,
        ))

    combos = list(itertools.product(
        TAU_LEVEL_SCALES, LEAF_BIASES, EXPAND, K_BRANCHES, TAU_FOCUSES
    ))
    for i, (scale, bias, expand, k_branch, tau_focus) in enumerate(combos, 1):
        label = (f"scale={scale}_bias={bias}_expand={'on' if expand else 'off'}"
                 f"_kb={k_branch}_focus={tau_focus}")
        if (scale, bias, expand, k_branch, tau_focus) == (0.0, 0.0, False, 2, 0.6):
            label += "_(v1)"
        retriever = RaptorRetriever(
            index,
            qcond_config=QCondConfig(
                tau_level_scale=scale,
                leaf_bias=bias,
                expand_terminal_leaves=expand,
                k_branch=k_branch,
                tau_focus=tau_focus,
            ),
        )
        for top_k in TOP_KS:
            rows.append(_retrieval_row(
                retriever, SearchStrategy.RAPTOR_QCOND, label,
                questions, scorer, top_k,
            ))
        print(f"  [{i}/{len(combos)}] {label}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}\n")

    # Leaderboard per k: recall-first among settings within the collapsed
    # token budget, tie-broken by precision.
    for top_k in TOP_KS:
        ref_row = next(r for r in rows
                       if r["setting"] == "REF_collapsed" and r["top_k"] == top_k)
        budget = ref_row["tokens"]
        candidates = [r for r in rows
                      if r["top_k"] == top_k and r["setting"] != "REF_collapsed"]
        in_budget = [r for r in candidates if r["tokens"] <= budget] or candidates
        in_budget.sort(key=lambda r: (-r["recall"], -r["precision"]))
        print(f"--- k={top_k} (collapsed ref: recall={ref_row['recall']:.3f} "
              f"prec={ref_row['precision']:.3f} tokens={budget:.0f}) ---")
        for r in in_budget[:5]:
            print(f"  {r['setting']:38} recall={r['recall']:.3f} "
                  f"prec={r['precision']:.3f} tokens={r['tokens']:.0f}")


if __name__ == "__main__":
    main()
