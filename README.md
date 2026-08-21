# UnlearnShield

Machine unlearning for tabular fraud models. When a record has to come out of a
trained model — a poisoned batch, a fraud ring's history, a revoked consent —
UnlearnShield rebuilds only the affected shard and emits a signed manifest that
an auditor can check without access to the training system.

It uses SISA (sharded, isolated, sliced training), so removal is structural
rather than statistical: the retrained shard is a function of its retained data,
not a model nudged away from the deleted rows by gradient ascent.

**Status: Phase 2 of 7.** The determinism harness and the offline unlearning
engine are built and green — a shard can be built, trained slice by slice, and a
subject excised end to end from the CLI. The manifest, the API, and the
dashboard are not. See
[docs/implementation-plan.md](docs/implementation-plan.md) for the full build
plan and [docs/plan-corrections.md](docs/plan-corrections.md) for the defects
found in it and what changed.

## What is actually here

```
config/determinism.py    the harness everything downstream depends on
config/settings.py       env-driven config, subject_ref HMAC
data/synth.py            PaySim-shaped generator (real CSV drops in unchanged)
data/churn_score.py      PLACEHOLDER churn signal, clearly marked as invented
engine/sharder.py        hot/cold shard assignment, frozen at ingest
engine/slicer.py         subject-aligned slices, churn-ordered
engine/preprocessing.py  per-shard constants, fit on slice 0 only
engine/model.py          fixed MLP, identical across shards
engine/train.py          build + incremental slice training with checkpoints
engine/rebuild.py        purge, roll back, retrain forward
db/schema.sql            full DDL, including tables later phases populate
tests/unit/              29 tests
docs/adr/                decision records
```

## Build a corpus and erase someone

```bash
export PYTHONHASHSEED=0
.venv/bin/python -m engine.train --build      # partition + train 5 shards
.venv/bin/python -m engine.rebuild --subject C0000042
```

The rebuild reports which slices it retrained. A subject in slice 4 costs one
slice; a subject in slice 0 costs the whole shard. That spread is the point.

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q
```

`PYTHONHASHSEED` can only be set before the interpreter starts, so the harness
refuses to run without it rather than silently producing weights nobody can
reproduce. Docker and CI set it for you.

On the pinned image, which is where the determinism claim has to hold:

```bash
docker compose run --rm determinism python -m pytest tests/unit -q
```

## Why determinism comes first

The product is a manifest saying a shard was retrained without a subject's data.
Nothing about a weight tensor reveals whether that is true. The only affordable
check is to re-run the rebuild and compare digests — which works only if a
rebuild is a pure function of its inputs.

Seeding does not get you there. Float addition is not associative, and OpenMP
splits reductions by thread count, so the same shard on a 4-core runner and a
16-core one yields different normalisation constants and different weights.
[ADR 0003](docs/adr/0003-cpu-only-determinism.md) has the measured numbers.

Determinism is scoped to a `code_digest`, not asserted across versions. A torch
upgrade is allowed to change weights. It is not allowed to change them for a
fixed image digest. Otherwise every dependency bump reads as a compliance
incident.

## Two things the plan got wrong

Both were found while building Phase 2, and both fail silently — the rebuild
completes, the weights change, every check passes, and the subject is still in
the model.

**Record-level slicing buys nothing here.** SISA slices by record because its
deletion unit is one data point. Ours is a subject who owns many records, so
their rows scatter across slices and the rollback point is the minimum among
them — which for a 30-record subject against 5 slices is slice 0, every time.
Slices are cut at subject boundaries instead ([ADR 0005](docs/adr/0005-subject-aligned-slices.md)).

**Refitting preprocessing on rebuild reintroduces the leak it was meant to
prevent.** Refit on the shard's remaining data, then resume from
`checkpoint_{i-1}`, and you resume from weights trained under scaling constants
fit on the subject you are deleting. Fitting on slice 0 only closes it
([ADR 0004](docs/adr/0004-per-shard-preprocessing.md)).

## What the manifest proves, and what it does not

The Merkle absence proof shows a `subject_ref` is missing from the shard's
retained record set. It does **not** bind `result_weights` to `dataset_root` —
a pipeline could purge the record, publish a clean root, and ship the previous
checkpoint. The signature would still verify, because a signature over a false
claim is still a valid signature.

Weight provenance rests on two other things: the `code_digest` recorded in the
manifest, and re-running a sample of completed jobs and comparing digests.
Sample selection is derived from `HMAC(erasure_id, audit_key)` rather than
chosen by the worker, so an operator cannot steer the sample away from jobs it
would rather not have re-run, and an auditor holding `audit_key` can confirm the
sample was not gamed.

## Scope boundary

UnlearnShield erases a subject from the **model**. It does not erase them from
your data lake, your feature store, or your backups. Those nodes of the deletion
graph sit behind interfaces here and are owned by the upstream system. A
compliance story that only covers the model is incomplete, and this is the part
it covers.

## Licence

MIT.
