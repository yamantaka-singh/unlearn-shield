# 0012 — Validated against real PaySim, not just the synthetic generator

**Status:** accepted, 2026-08-22

Every prior phase was verified against `data/synth.py` — a generator this
project controls the statistics of. This is the first pass against the real,
public dataset the plan was written around: PaySim, the "primary source" named
in `docs/implementation-plan.md`. Two real bugs were found and fixed. One
apparent finding — the scariest one — was investigated and retracted once its
actual cause turned out to be a mistake in the validation script itself, not
the product. All three are recorded, because the retraction is as load-bearing
as the fixes: it is the difference between "sharding costs real accuracy" and
"a test harness pattern is error-prone," and those call for very different
responses.

## Getting the data

Kaggle requires authenticated login; this session has none. `huggingface.co/datasets/theman10/paysim`
mirrors the same file (MIT-licensed, ungated) — confirmed genuine by file size
(470 MB, matching the known PaySim export exactly) and by column layout
(`step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud`,
matching the published schema). Validation ran against the first 1.5M real
rows (24% of the 6.3M-row file) — the download was stopped once findings
stabilized across two check sizes (815k and 1.5M rows, same conclusion both
times), not because the full file would have said something different.

## Finding 1: `engine/slicer.py` crashed on an empty shard

`assign_slices` did `cumulative[-1]` on `record_counts[order]` with no guard
for zero subjects. Every existing test used 120+ subjects across 5 shards, so
an empty shard never came up by chance. A schema-accurate real-PaySim fixture
with 5 subjects across the real default of 5 shards hit it immediately:

```
IndexError: index -1 is out of bounds for axis 0 with size 0
```

Hash-based shard assignment (`engine/sharder.py`) leaving a shard empty isn't
a real-data quirk specifically — it's the ordinary behaviour of hashing few
subjects into several buckets, and it will happen to a new tenant's first few
customers or a small staging seed the same way it happened to a 5-row fixture.
`engine/train.py::build()` loops `range(NUM_SHARDS)` unconditionally, so
nothing upstream protects this.

**Fixed:** `assign_slices` returns `np.empty(0, dtype=np.int64)` for empty
input. `test_empty_shard_returns_empty_not_a_crash` guards it.

## Finding 2: hot-shard concentration silently does nothing on real data

`engine/train.py::build()` had `max_step: int = 720` as a hardcoded default,
used regardless of whether `raw` was the synthetic generator (which produces
exactly that range) or real data (which doesn't). `churn_score.py` computes
`recency = last_step / max_step`. A 150k-row real sample spans steps 1–153 —
recency tops out around 0.21, and `0.65 × 0.21 + 0.35 × noise` can never reach
`HOT_THRESHOLD` (0.6) no matter what the noise term does. Every subject landed
in a cold shard:

```
subjects per shard (before fix): {2: 50375, 3: 49757, 4: 49863}   # 0, 1 empty
subjects per shard (after fix):  {0: 22679, 1: 22639, 2: 35152, 3: 34610, 4: 34915}
```

The entire premise of hot/cold sharding — concentrate likely deletions into a
few shards — produced nothing, with no error and no warning. This is exactly
the failure shape this project is otherwise careful about: a silent gap
between what the design claims and what it does.

**Fixed:** `max_step` defaults to `None`; `build()` derives it from
`raw["step"].max()` whenever real data is supplied, and only falls back to 720
for the synthetic generator. Pass `max_step` explicitly to override.

## Finding 3, retracted: apparent severe accuracy loss from sharding

Measuring GBDT accuracy on real data (0.108% fraud prevalence, next section)
produced per-shard AUCs of 0.49–0.69, against an unsharded model on the same
rows scoring 0.9989. The natural read: SISA fragments an already-tiny fraud
class across 5 shards and further across 5 training slices, and at real
prevalence there simply isn't enough signal per shard to learn the pattern.
That would have been a genuine, serious, previously unquantified cost of the
whole architecture — the synthetic generator's engineered ~3.4% fraud rate
(34× real prevalence) had masked it in every prior test.

It is not what was happening. Root cause: the validation script set
`gbdt.SHARD_DIR` but not `engine.train.SHARD_DIR`. `engine.gbdt.load_shard` is
`engine.train`'s own function object (imported via `from engine.train import
... load_shard`), and a function resolves free variables through the module it
was *defined* in — so `load_shard()` always reads through `engine.train`'s
binding regardless of who calls it. `engine.gbdt.save_booster` and
`booster_path`, by contrast, are defined in `engine/gbdt.py` and read *its*
binding. One `gbdt.build()` call therefore read shard data through one
module's `SHARD_DIR` and wrote the trained booster through another's. With
only `gbdt.SHARD_DIR` patched, the booster trained on stale leftover demo data
from the untouched default directory, then got saved to the correct real-data
path — and was later scored against the real data it was never trained on.

Confirmed by controlled comparison on the actual shard files:

```
saved (misconfigured) booster: AUC 0.6897, 100 trees
freshly retrained booster:     AUC 0.9983, 100 trees, same shard, same params
digests equal: False
```

Fixed the script (patch both bindings) and re-measured. Real result:

```
shard 0: 22680 rows, 21 fraud, in-sample AUC 0.9998
shard 1: 22640 rows, 21 fraud, in-sample AUC 0.9998
shard 2: 35153 rows, 39 fraud, in-sample AUC 0.9993
shard 3: 34611 rows, 30 fraud, in-sample AUC 0.9997
shard 4: 34916 rows, 51 fraud, in-sample AUC 0.9983
```

Every shard matches the unsharded baseline. Real fraud prevalence does not
meaningfully cost accuracy here, even with as few as 21 positive examples in a
shard — this fraud signal (a transaction that fully drains the origin balance)
is close to a deterministic rule, and a handful of examples is enough for a
boosted tree to find it. The synthetic generator's inflated fraud rate turns
out not to have been hiding a real problem after all.

### What this cost: three separate real confusions, one session

The `SHARD_DIR`-per-module pattern (every module that does `from config.settings
import SHARD_DIR` gets its own frozen copy) caused three distinct mistakes
while producing this ADR: an empty-shard repro that only patched
`engine.train` (harmless, since that particular bug didn't depend on which
module), a "348 rows" false alarm from an unrelated stale-directory read in a
completely different script, and this one — the most expensive, because it
looked like a real, severe product defect for the length of a full diagnostic
session before resolving into a validation-script bug.

**Fixed at the point that actually recurred**, not with a production refactor:
`tests/conftest.py::override_shard_dir()` patches `engine.train`,
`engine.gbdt`, and `engine.rebuild` together, documents exactly why patching
one alone is insufficient, and is guarded by
`test_override_shard_dir_helper_patches_every_module_consistently` — which was
verified to fail when `engine.train`'s binding is the one omitted, reproducing
this exact incident on demand. A full refactor to a single shared config
object was considered and set aside: production always sets these constants
via environment variables before any import happens, which the per-module
binding pattern handles correctly; the fragility is specific to tests and
ad-hoc scripts that monkeypatch mid-process, so the fix belongs there.

## What held up, unmodified

**Subject cardinality.** `data/synth.py` invents a geometric multi-record
distribution per subject specifically to stress-test ADR 0005's
subject-aligned slicing. Real PaySim does not have that shape at all:

```
815k real rows:  815,431 / 815,588 nameOrig ids (99.98%) have exactly 1 record, max 2
1.5M real rows:  1,518,780 / 1,519,319 nameOrig ids (99.96%) have exactly 1 record, max 3
```

PaySim models anonymous one-shot mobile-money transfers between randomly
generated customer identifiers, not a persistent account history — this is a
property of the simulator's design, not a defect in this project's assumption.
Subject-aligned slicing does not misbehave here: with one record per subject,
`min_slice_idx == max_slice_idx` trivially, and the rare 2–3-record subjects
that do occur are still handled correctly (verified directly — a real subject
with `record_count=6` was erased and its certificate verified in the
end-to-end run below). The design degrades gracefully to record-level
granularity rather than doing anything wrong; the multi-record scenario it
exists to fix simply doesn't arise in this particular dataset. A real
deployment's own transaction log — repeat customers with years of history,
rather than PaySim's one-shot transfers — is exactly the shape ADR 0005 was
written for, and PaySim was never going to be able to confirm or deny that;
only a real customer ledger can.

**End-to-end erasure, on real data.** Built 5 real shards from 150k real rows,
trained, picked a real subject with `min_slice_idx=1` (6 real records),
erased them, and verified the resulting certificate:

```
target shard=3 min_slice_idx=1 record_count=6
rebuild: 0.05s, slices_retrained=[1, 2, 3, 4], rows_purged=6
VERIFIED
  signature valid (Ed25519)
  absence proof valid against dataset_root 75146eb6ba23d40b... [sparse Merkle (smt-256)]
  subject 026ca4b731052c17... absent from shard 3 at model_version shard3-24a46b455c77
```

**Real fraud rate**: 0.108% (162/150,000) — close to PaySim's documented
~0.13%, and 34× rarer than `data/synth.py`'s engineered ~3.4%. Worth knowing
for anyone tuning `DISAGREEMENT_THRESHOLD` or the eval harness against real
traffic rather than the synthetic default.

## What's new

- `data/prepare.py` — real PaySim ingest, one pass over the CSV via stdlib
  `csv`, no pandas. Produces the exact schema `data/synth.py` does, so
  `engine.train.build(raw=...)` accepts either with no other change.
- `scripts/check_paysim_structure.py` — the subject-cardinality check above,
  runnable against any PaySim export.
- `tests/unit/test_prepare.py` — a tiny fixture in PaySim's real column layout
  (the real 470MB file can't be vendored, but the layout is public and
  stable), including a schema-drift guard: an unrecognised `type` value raises
  rather than silently miscoding.
- `engine/train.py::build(raw=...)` — the seam that let all of the above run
  through the existing pipeline unchanged past the first line.

## Consequences

- `ADR 0005`'s ordering assumption (churn-ascending slices, subjects most
  likely to be deleted in the last slice) is unaffected by any of this — it's
  a design for a real customer ledger, and PaySim's one-shot-transfer shape
  means it is neither confirmed nor contradicted here. That validation needs a
  real transaction log, not PaySim.
- Anyone building a demo or benchmark against a real-data subsample must pass
  or derive `max_step` correctly (now automatic when `raw` is supplied) —
  otherwise hot-shard concentration silently does nothing, with no error.
- The `SHARD_DIR`-per-module fragility is mitigated for future tests, not
  eliminated for future ad-hoc scripts. Anyone writing a manual verification
  script against `engine.gbdt` and `engine.train` together should import
  `tests.conftest.override_shard_dir` rather than patch modules by hand.
