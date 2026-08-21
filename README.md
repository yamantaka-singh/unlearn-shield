# UnlearnShield

Machine unlearning for tabular fraud models. When a record has to come out of a
trained model — a poisoned batch, a fraud ring's history, a revoked consent —
UnlearnShield rebuilds only the affected shard and emits a signed manifest that
an auditor can check without access to the training system.

It uses SISA (sharded, isolated, sliced training), so removal is structural
rather than statistical: the retrained shard is a function of its retained data,
not a model nudged away from the deleted rows by gradient ascent.

**Status: Phase 1 of 7.** The determinism harness is built and green. The
unlearning engine, the manifest, the API, and the dashboard are not. See
[docs/implementation-plan.md](docs/implementation-plan.md) for the full build
plan and [docs/plan-corrections.md](docs/plan-corrections.md) for the defects
found in it and what changed.

## What is actually here

```
config/determinism.py   the harness everything downstream depends on
config/settings.py      env-driven config
db/schema.sql           full DDL, including tables later phases populate
scripts/spot_check_determinism.py
tests/unit/             determinism tests, including the negative controls
docs/adr/               decision records
```

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
