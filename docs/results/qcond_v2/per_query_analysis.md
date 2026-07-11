# Per-query paired analysis: qcond v2 vs collapsed

Pairs each (question, k) row of qcond v2 with the collapsed-search row from the
same protocol (same generator, corpus, questions), so every delta is a
controlled within-question comparison. 95% CIs are paired bootstrap
(10,000 resamples). Reproduce with `scripts/analyze_qcond_pairs.py`.

## QASPER (held-out; 180 paired rows)

| slice | n | mean ΔF1 (qcond−collapsed) | win/tie/loss |
|---|---|---|---|
| abstractive_synthesis | 57 | −0.0050 | 25/1/31 |
| extractive_evidence | 54 | **+0.0070** | 26/7/21 |
| unanswerable | 33 | 0.0000 | 0/33/0 |
| yes_no | 36 | −0.0000 | 8/22/6 |
| **overall** | 180 | **+0.0005, 95% CI [−0.0036, +0.0048]** | 59/63/58 |

On gold similarity the pattern is the same: overall CI [−0.0109, +0.0084],
extractive +0.0095 vs abstractive −0.0122.

**Reading:** parity on the held-out set is statistically confirmed — the CI is
tight around zero and win/loss counts are balanced. The category split is
directionally consistent with the mechanism (qcond's beam commits to leaves, so
it does relatively better on extractive questions and relatively worse on
abstractive ones, where collapsed can surface summary nodes), but at n≈55 per
category the split is within noise — it is a hypothesis for the scale-up
experiment, not a finding.

## Synthetic (75 paired rows)

| metric | overall mean Δ | 95% CI | win/tie/loss |
|---|---|---|---|
| gold_similarity | −0.0192 | [−0.0389, −0.0003] | 27/0/48 |
| fact_recall | −0.0831 | [−0.1496, −0.0169] | 6/46/23 |

**Reading:** on the short-document corpus, collapsed remains **significantly
better** at the answer level — both CIs exclude zero — even though qcond v2
matches its retrieval-level context recall (0.940 at k=10). The gap is
concentrated in fact recall at k=10 (Δ −0.196): with the whole corpus only 1–2
pages per document, collapsed's mixed-abstraction pool packs more exact figures
into the context than qcond's leaf-heavy scopes. Parity is a QASPER
(long-document) result, not a general one.

## Implications for next steps

1. No usable routing signal at this sample size — per-category deltas on QASPER
   are ±0.01 with overlapping distributions. A query-type router is not
   justified by this data.
2. The extractive-vs-abstractive split and the short-vs-long-document split
   both point the same direction: qcond's relative strength grows with document
   length and specificity of the question. The decisive experiment remains the
   corpus scale-up (deeper trees), where beam search has a structural advantage
   to exploit and the currently-inert termination mechanisms get levels to act
   on.
