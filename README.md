# UnlearnShield

Machine unlearning for tabular fraud models. When a record has to come out of a
trained model — a poisoned batch, a fraud ring's history, a revoked consent —
UnlearnShield rebuilds only the affected shard and emits a signed manifest that
an auditor can check without access to the training system.

It uses SISA (sharded, isolated, sliced training), so removal is structural
rather than statistical: the retrained shard is a function of its retained data,
not a model nudged away from the deleted rows by gradient ascent.

**Status: Phase 4 of 7.** Determinism harness, offline unlearning engine, and
verification layer are built. The gateway and worker are now built too — an
erasure request goes in over HTTP, gets processed asynchronously by a
Postgres-queued worker, and comes back out as a certificate a standalone
verifier accepts, with `/v1/predict` reflecting the change. The dashboard is
not built. See [docs/implementation-plan.md](docs/implementation-plan.md) for
the full build plan and [docs/plan-corrections.md](docs/plan-corrections.md)
for defects found in it.

## What is actually here

```
config/determinism.py    the harness everything downstream depends on
config/settings.py       env-driven config, subject_ref HMAC, auth token map
data/synth.py             PaySim-shaped generator (real CSV drops in unchanged)
data/churn_score.py       PLACEHOLDER churn signal, clearly marked as invented
engine/sharder.py         hot/cold shard assignment, frozen at ingest
engine/slicer.py          subject-aligned slices, churn-ordered
engine/preprocessing.py   per-shard constants, fit on slice 0 only
engine/model.py           fixed MLP, identical across shards
engine/train.py           build + incremental slice training with checkpoints
engine/rebuild.py         purge, roll back, retrain forward, emit a certificate
verify/merkle.py          RFC 6962 tree, non-inclusion proofs
verify/manifest.py        canonical serialisation
verify/sign.py            Ed25519
verify/verifier_cli.py    standalone auditor tool, imports nothing from engine/
gateway/                  stateless FastAPI app -- never trains
  auth.py                 bearer-token -> scope map, no user accounts
  idempotency.py          INSERT ... ON CONFLICT, no check-then-insert race
  routes/erasure.py       202 intake, status, certificate, attest
  routes/predict.py       sequential shard ensemble, model_version in every response
  routes/models.py        current model_version, manifest lookup by version
worker/                   separate deployment, no ingress -- never serves
  queue.py                claim-then-commit, lease + reaper
  jobs.py                 batch rebuild per shard, promote, spot-check sampling
scripts/load_routing.py   loads engine/'s offline routing.json into Postgres
db/schema.sql             full DDL
tests/unit/               91 tests, no DB required
tests/integration/        18 tests, real Postgres required (skip cleanly without it)
tests/e2e/                1 test: full lifecycle over real HTTP + real Postgres
docs/adr/                 decision records
```

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt

# Unit tests need no database
PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q
```

`PYTHONHASHSEED` can only be set before the interpreter starts, so the harness
refuses to run without it. Docker and CI set it for you.

## Build a corpus, erase someone, serve a prediction

```bash
docker compose up -d postgres   # port 55432 on the host -- 5432 is often taken
                                 # by a native Postgres install; the internal
                                 # container-to-container URL still uses 5432

export DATABASE_URL="postgresql://unlearnshield:unlearnshield@localhost:55432/unlearnshield"
export PYTHONHASHSEED=0
.venv/bin/python -m verify.sign                     # dev keypair, once
export UNLEARNSHIELD_SIGNING_KEY=$(cat .signing_key)

.venv/bin/python -m engine.train --build            # partition + train 5 shards
.venv/bin/python -m scripts.load_routing            # load routing + baseline into Postgres

.venv/bin/python -m uvicorn gateway.main:app --port 8000 &
.venv/bin/python -m worker.main                     # separate process, polls the queue

curl -X POST localhost:8000/v1/erasure \
  -H "Authorization: Bearer dev-token" -H "Idempotency-Key: demo-1" \
  -H "content-type: application/json" \
  -d '{"subject_id": "C0000042", "reason": "fraud_excision"}'
# {"erasure_id": "...", "status": "queued", "sla_deadline": "..."}
```

The worker picks it up, retrains only the affected shard, writes a signed
manifest, and promotes a new `model_version`. `GET /v1/erasure/{id}/certificate`
returns it; `verify/verifier_cli.py` accepts it with no access to any of the
above. `POST /v1/predict` reflects the new `model_version` immediately after.

Run everything against the pinned image instead:

```bash
docker compose run --rm determinism python -m pytest tests/unit -q
```

## Why determinism comes first

The product is a manifest saying a shard was retrained without a subject's
data. Nothing about a weight tensor reveals whether that is true. The only
affordable check is to re-run the rebuild and compare digests — which works
only if a rebuild is a pure function of its inputs.

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
claim is still a valid signature. The verifier prints this on every successful
run rather than letting an auditor infer more than was shown.

Weight provenance rests on two things beyond the signature: `code_digest`, and
sampling completed jobs for re-run. Sample selection is derived from
`HMAC(erasure_id, audit_key)` rather than chosen by the worker, so an operator
cannot steer the sample away from jobs it would rather not have re-run.
**The re-run itself is not wired yet** — see Known gaps below.

## The proof leaks two neighbours

A sorted-Merkle non-inclusion proof works by naming the target's immediate
neighbours in sort order, so every certificate discloses two other subjects'
HMAC refs. They're pseudonymous, but an auditor collecting many certificates
accumulates a growing slice of the shard's population. Inherent to the
construction; a sparse Merkle tree avoids it at 256-level proof cost, and isn't
built. [ADR 0002](docs/adr/0002-manifest-over-hash-equality.md) has the
reasoning.

## Known gaps, stated rather than hidden

Two things surfaced only by running the gateway and worker against a real
Postgres instance, not by code review or unit tests, and are worth knowing
about before this touches real data:

**The reproducibility spot-check selects a sample but doesn't re-run it yet.**
`worker/jobs.py::_should_spot_check` picks jobs via `HMAC(erasure_id,
audit_key)` and that part is real and tested. Actually re-running a sampled job
and comparing digests needs the pre-purge shard state, which nothing currently
snapshots. Writing a fabricated "matched" row without a real second rebuild
would corrupt the one table Phase 7's drift alert depends on, so nothing is
written instead of writing something false.

**Offline mutation (shard purge, retrain, `routing.json`) is not transactional
with Postgres.** If the worker's DB bookkeeping fails after a rebuild has
already happened, the job needs manual reconciliation, not a blind retry — a
retry would call into a routing table that no longer lists the subject and
fail with a confusing error that hides the real, already-resolved problem. The
worker fails loudly and immediately with an explicit message when this happens,
rather than leaving the job silently stuck until its lease expires. See
[ADR 0006](docs/adr/0006-content-addressed-checkpoints-and-claim-then-commit.md)
for the full account, including a second defect (silently overwritten
checkpoint files) found and fixed the same way.

## Scope boundary

UnlearnShield erases a subject from the **model**. It does not erase them from
your data lake, your feature store, or your backups. Those nodes of the
deletion graph sit behind interfaces here and are owned by the upstream
system. A compliance story that only covers the model is incomplete, and this
is the part it covers.

## Testing

```bash
PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q          # no DB needed
docker compose up -d postgres
DATABASE_URL=postgresql://unlearnshield:unlearnshield@localhost:55432/unlearnshield \
  .venv/bin/python -m pytest tests/integration tests/e2e -q        # real Postgres
```

Integration and e2e tests skip cleanly (not fail) when Postgres is
unreachable, so `pytest tests/unit` stays fast and hermetic in any
environment, and the fuller suite runs wherever a real Postgres exists.

## Licence

MIT.
