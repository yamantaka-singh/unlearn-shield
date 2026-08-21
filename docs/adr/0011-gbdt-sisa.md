# 0011 — SISA over gradient-boosted trees

**Status:** offline engine accepted, 2026-08-21
**Scope:** `engine/gbdt.py` only. Not wired into the gateway, worker, or
manifest pipeline — see "What is not built".

## Context

`roadmap-assessment.md` recorded GBDT support as the strongest deferred item:
most production tabular fraud models are boosted trees, so a SISA
implementation that only handles MLPs addresses a minority of its target.

It also predicted the fit would be *better* than the MLP path. That prediction
was checked before any module was written.

## Verified before building

**Truncation is an exact rollback.** `booster[0:n]` produces predictions
bit-identical to training only `n` rounds (`atol=0, rtol=0`). Incremental
boosting via `xgb_model=` is likewise bit-identical to a straight-through run.
This is the claim the whole engine rests on; if it were approximate, the
unlearning guarantee would be approximate.

**A rebuild equals a clean retrain, byte-for-byte.** Purge the target, keep the
trees from slices before theirs, boost forward — and the result is
byte-identical to training from scratch on the retained data
(`test_rebuild_equals_a_clean_retrain_on_retained_data`). This holds because
slices are subject-aligned (ADR 0005), so the kept trees provably never saw the
target. "Byte-identical to never having trained on them" is a materially
stronger statement than the "behaves similarly" that gradient-ascent unlearning
offers.

## Decision

Build the offline tree engine. Three properties fall out of trees that the MLP
path has to work for:

| | MLP path | GBDT path |
|---|---|---|
| Reaching the rollback point | load `shard{k}_slice{i}.pt` | slice the tree list — free |
| Checkpoint files per shard | 5 slices + 1 preprocessor | **1 booster** |
| Fitted per-shard statistics | scaler mean/std (ADR 0004) | none |

Measured rebuild cost by the target's slice, against a full build:

| Target in | Cost | Trees kept |
|---|---:|---:|
| slice 0 | 72% | 0 |
| slice 2 | 50% | 40 |
| slice 4 | **20%** | 80 |

That monotone gradient is the entire point of slicing, and it appears cleanly
because dropping trees is exact.

Feature handling needs no scaler at all: trees split on thresholds, so any
monotone rescaling yields the same tree. Features are raw numerics plus a
fixed-vocabulary one-hot, neither fitted to anyone's data —
`test_features_of_one_row_do_not_depend_on_any_other_row` asserts the property
a fitted scaler would break.

## A correction: the `base_score` leak was over-claimed

An earlier draft of this ADR — and of the code comments — asserted that
XGBoost's fitted `base_score` carried erased subjects' influence into rebuilt
models, citing a measured 0.3100255 inherited against 0.29768768 for a clean
model.

That measurement was real but did not come from this engine. It trained on all
data at once, then purged 20% and continued. This engine does not do that.

Measured in the actual code path, all three models — pre-purge, rebuilt, and
clean-retrained — report the same `base_score`. XGBoost estimates it on the
*first* `train()` call, which here sees **slice 0 only**. So a subject in slice
≥ 1 is absent from the data the estimate is drawn from, and a subject in slice 0
makes `rollback()` return `None`, restarting training and redrawing the estimate
from retained data. Both paths are already clean.

`base_score` is still pinned to `0.5`, because that safety is *incidental* — it
depends on when XGBoost happens to estimate, and breaks if training ever starts
mid-slice or if XGBoost changes that timing. A constant is fitted to nobody and
costs nothing, which turns a contingent property into a structural one.

The test was downgraded to match: it asserts the pin as a config property and
confirms the pin reaches the model. A behavioural assertion would **pass with
the pin removed**, which is a test that looks like a guard and guards nothing.
The reasoning is recorded in the test's own docstring so a future reader does
not "strengthen" it back into something vacuous.

## What is not built

The gateway, worker, manifest, and SMT paths are untouched. This is the offline
engine only, mirroring how Phase 2 staged the MLP path before Phase 4 wired it
up. Serving needs its own decision: `inference/batched_ensemble.py` is
PyTorch-specific (`stack_module_state` + `vmap`) and has no tree analogue, so a
GBDT ensemble needs a different serving path rather than a parameter to the
existing one.

LightGBM is not built. It is the same shape — `init_model=` for continuation,
and the same truncation property — so the second implementation is mechanical
once the first is proven. Building both before either was needed would have
been speculative.

## Consequences

- New dependency: `xgboost==2.1.3`, pinned like everything else because
  determinism is a property of the whole environment (ADR 0003).
- `nthread=1` for the same reason torch's threads are pinned: thread count
  changes float reduction order, and a manifest is only auditable if a rebuild
  reproduces.
- Two engines now exist with no shared interface. That is deliberate — an
  abstract base class over one-and-a-half implementations would be the
  speculative structure this project keeps declining. Extract one when the
  serving path genuinely needs to dispatch on model type.
