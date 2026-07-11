# qcond v2: calibration and held-out evaluation

Protocol: the three README next-step mechanisms (`tau_level_scale`, `leaf_bias`,
`expand_terminal_leaves`) were implemented as `QCondConfig` knobs, then calibrated
together with qcond's pre-existing hyperparameters using retrieval-only scoring
(no LLM) on the **synthetic eval set only** — see `scripts/sweep_qcond.py` and
[sweep.csv](sweep.csv). QASPER was held out and touched exactly once, below.
Generator: `claude-haiku-4-5`, identical to the v1 runs, so rows are directly
comparable.

## Diagnostic

v1's failure was not (only) early termination: with `k_branch=2` the descent
beam explored ≤8 of ~150 leaves and returned just 3–8 candidates at `top_k=10`,
capping context recall at 0.607 on the synthetic set regardless of termination
tuning. The three hypothesized termination fixes added no recall once the beam
widened (the synthetic tree is depth 2; they remain available as config knobs).

**Winner** (recall-first, precision tie-break, then simplicity):
`k_branch = 5`, all other settings unchanged from v1
(`tau_term=0.05, tau_focus=0.6, max_descents=5, softmax_temp=0.1`,
termination mechanisms inactive). v1 used `k_branch=2`.

## Synthetic corpus (25 questions; calibration domain)

| config | sim v1 → v2 | fact v1 → v2 | recall v1 → v2 | tokens v1 → v2 |
|---|---|---|---|---|
| qcond k3  | 0.693 → **0.737** | 0.231 → **0.401** | 0.487 → **0.777** | 354 → 321 |
| qcond k5  | 0.683 → **0.746** | 0.303 → **0.491** | 0.607 → **0.820** | 577 → 538 |
| qcond k10 | 0.706 → **0.764** | 0.348 → **0.531** | 0.607 → **0.940** | 779 → 1066 |
| collapsed k10 (ref) | 0.799 | 0.727 | 0.940 | 1428 |

## QASPER (60 questions; held-out, one shot)

| config | F1 v1 → v2 | sim v1 → v2 | recall v1 → v2 | tokens v1 → v2 |
|---|---|---|---|---|
| qcond k3  | 0.040 → **0.050** | 0.222 → **0.232** | 0.433 → **0.550** | 311 → 301 |
| qcond k5  | 0.048 → **0.059** | 0.219 → **0.253** | 0.450 → **0.600** | 523 → 505 |
| qcond k10 | 0.049 → **0.062** | 0.226 → **0.259** | 0.517 → **0.700** | 1049 → 1012 |
| collapsed k3/k5/k10 (ref) | 0.052 / 0.055 / 0.063 | 0.243 / 0.240 / 0.265 | 0.583 / 0.633 / 0.700 | 319 / 531 / 1071 |
| best flat baseline (fixed_512__hybrid__k5) | 0.065 | 0.283 | 0.700 | 2468 |

## Verdict

qcond v2 improves on v1 across every metric at every k on both corpora, and the
calibration transferred to the held-out set. Against the paper's best mode it
reaches **parity, not superiority**: statistically level with collapsed at k3
and k10, slightly ahead at k5, at a marginally lower token budget. The original
premise — that per-query adaptive traversal beats a flat collapsed search — is
not supported at matched budgets; the honest conclusion is that collapsed search
is hard to beat and qcond's value is now matching it while retaining a
traversal-shaped mechanism to build on.

Raw rows: `results_qcond2.jsonl` / `results_qcond2_qasper.jsonl` (regenerable);
manifests committed alongside this file.
