# Scale-up experiment: 250 papers, 13,437 leaves

**Question.** Does qcond's parity with collapsed search (observed at 30 papers)
improve at scale, where beam traversal should have a structural advantage
(H1)? And do the three termination mechanisms — inert on shallow trees —
start mattering on deeper ones (H2)?

**Protocol.** 250 QASPER papers, 1 question each, split seeded into 100 dev /
150 test. Dev: free retrieval-only calibration (`scripts/sweep_qcond.py`,
grid over all five qcond knobs; leaderboard in
[sweep_250dev.csv](sweep_250dev.csv)). Test: one shot, 11 configs × 150
questions, generator `claude-haiku-4-5`, qcond run with the dev-calibrated
setting (`k_branch=5, tau_focus=0.85` via `--qcond-config`). Experiment cost
≈ $3.40.

## Finding 1 (H2): RAPTOR trees do not get deep

13,437 leaves clustered to **336 → 14 nodes: depth 2**. BIC-selected GMM
clustering yields ~40:1 fanout, so tree depth grows like log₄₀(N) — a
depth-5 tree would need ~10⁸ leaves. "Deep trees" barely exist at any
practical corpus size, so level-dependent termination mechanisms have almost
nothing to condition on: all termination settings tied exactly on dev, again.
H2 is not just unconfirmed — the regime it targets is unreachable.

## Finding 2 (H1): the gap to collapsed reopens at scale — refuted

Test set (150 held-out questions, best per family):

| config | F1 | sim | ctx recall | tokens |
|---|---|---|---|---|
| raptor_collapsed k10 | **0.056** | 0.267 | 0.493 | 1,048 |
| fixed_512 vector k10 | 0.054 | 0.260 | 0.440 | 4,914 |
| fixed_512 hybrid k5 | 0.054 | 0.279 | 0.433 | 2,457 |
| raptor_qcond k3 (best qcond) | 0.052 | 0.252 | 0.347 | 300 |
| raptor_qcond k10 | 0.049 | 0.261 | 0.420 | 1,007 |
| raptor_tree k10 | 0.049 | 0.264 | 0.587 | 2,219 |

Paired per-question deltas (qcond − collapsed, 450 pairs,
`scripts/analyze_qcond_pairs.py`):

- **F1: −0.0039, 95% CI [−0.0076, −0.0004]** — excludes zero. Collapsed is
  significantly better at 250 papers; the 30-paper parity does not persist.
- Similarity: −0.0047, CI [−0.0117, +0.0022] — same direction, not
  individually significant.

The scale trend across corpora is monotone against qcond: significantly
behind on the tiny synthetic corpus, parity at 30 papers, significantly
behind again at 250. Parity was the best case, observed only where the
candidate pool was small enough for a widened beam to cover most of it.

## Finding 3: the honest remaining value is search efficiency

From the instrumented dev sweep: qcond scores **~3,700 of 13,787 nodes per
query (27%)** for 87% of collapsed's recall (0.380 vs 0.437). Collapsed
scores every node by construction. For embedding dot-products this saving is
free anyway (one matrix multiply), so it buys nothing *in this benchmark* —
but per-node scoring is not always cheap: cross-encoder rerankers are
standard in production RAG, and agentic systems score candidates with LLM
calls. In those regimes a 73% reduction in scorer invocations for a ~7%
relative quality cost is a real trade, with qcond acting as the scoping
stage in front of the expensive scorer. Untested here (and the natural
follow-up): whether that beats the production-standard ANN-shortlist +
rerank pipeline at matched quality.

Also notable for RAPTOR overall: collapsed@k10 matches or beats the flat
baselines here while using **2.3–4.7× fewer context tokens** — the
token-efficiency finding from the 30-paper run strengthens at scale.

## Conclusion

The contribution's final form is a calibrated negative result with a complete
mechanism story: query-conditional traversal cannot beat collapsed search on
answer quality at any tested scale (matched budgets), because (a) RAPTOR
trees are too shallow for adaptive descent to have room to be clever, and
(b) whatever recall a wide beam recovers, a flat search over the same nodes
recovers more. Its measurable advantage — 73% less scoring work — only pays
where node scoring is expensive. Both hypotheses were pre-registered,
dev/test-split, and are reported as they landed.

Raw rows: `results_qasper250_merged.jsonl`, stitched from two runs — the first
was killed at row 1,245 by a network outage (row-level checkpointing kept all
completed rows; its partial `qcond_k10` config was fully rerun in the second).
Only the second run's manifest exists (`manifest_scaleup.json` — the first
crashed before its end-of-run manifest write); both ran the same commit,
machine, and settings.
