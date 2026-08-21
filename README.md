<div align="center">

# UnlearnShield

**Provable machine unlearning for tabular fraud models.**

Remove a subject from a trained model and get back a signed certificate
an auditor can verify — without access to your training system, your
database, or your weights.

[![tests](https://img.shields.io/badge/tests-144_passing-2ea043?style=flat-square)](#testing)
[![p99](https://img.shields.io/badge/predict_p99-9.8ms-2ea043?style=flat-square)](#serving-latency)
[![proof leak](https://img.shields.io/badge/subjects_leaked_per_proof-0-2ea043?style=flat-square)](#absence-proofs-that-name-nobody)
[![python](https://img.shields.io/badge/python-3.11-3776ab?style=flat-square)](https://www.python.org)
[![licence](https://img.shields.io/badge/licence-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## The problem

GDPR Article 17 and DPDP say a subject can demand erasure. Fraud teams need to
excise a poisoned batch or a ring's history. Both mean the same thing
technically: **remove a subject's influence from a trained model, and be able
to show you did.**

Gradient-ascent "unlearning" nudges a model away from deleted rows. It gives a
probabilistic guarantee, and an auditor cannot check it at all. *Probably
forgot* is not an answer.

## The approach

SISA — sharded, isolated, sliced training. Removal is **structural**: the
retrained shard is a function of its retained data, full stop.

```
   subjects ──┬── shard 0 ──┬─ slice 0 ─┬─ slice 1 ─┬─ slice 2 ─┬─ slice 3 ─┬─ slice 4
              │             │           │           │           │           │
              │            ckpt        ckpt        ckpt        ckpt        ckpt
              ├── shard 1 ── …
              ├── shard 2 ── …          ▲                                    ▲
              ├── shard 3 ── …          │                                    │
              └── shard 4 ── …     erase here                          erase here
                                   = retrain 4/5                       = retrain 1/5
                                     of the shard                        of the shard

   erasure request ──▶ purge rows ──▶ roll back to checkpoint ──▶ retrain forward
                                                                        │
                                         signed certificate ◀───────────┘
```

Shards are assigned by churn likelihood, and slices are ordered so the subjects
most likely to be erased land in the **last** slice, where rollback is
cheapest. Erasing a slice-4 subject retrains one fifth of one shard. That
spread is the entire point.

---

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt

PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q     # 125 tests, no database
```

`PYTHONHASHSEED` must be set before the interpreter starts, so the determinism
harness refuses to run without it. Docker and CI set it for you.

<details>
<summary><b>Full stack — build a model, erase someone, verify the certificate</b></summary>

```bash
docker compose up -d postgres     # host port 55432; 5432 is usually taken by a
                                  # native install. Container-to-container stays 5432.

export DATABASE_URL="postgresql://unlearnshield:unlearnshield@localhost:55432/unlearnshield"
export PYTHONHASHSEED=0
.venv/bin/python -m verify.sign                       # dev keypair, once
export UNLEARNSHIELD_SIGNING_KEY=$(cat .signing_key)

.venv/bin/python -m engine.train --build              # partition + train 5 shards
.venv/bin/python -m scripts.load_routing              # routing + baseline → Postgres

.venv/bin/python -m uvicorn gateway.main:app --port 8000 &
.venv/bin/python -m worker.main &                     # separate process, polls the queue

curl -X POST localhost:8000/v1/erasure \
  -H "Authorization: Bearer dev-token" \
  -H "Idempotency-Key: demo-1" \
  -H "content-type: application/json" \
  -d '{"subject_id": "C0000042", "reason": "fraud_excision"}'
```

```json
{ "erasure_id": "1967c3c3-…", "status": "queued", "sla_deadline": "2026-09-20T10:23:56Z" }
```

The worker retrains only the affected shard, writes a signed certificate, and
promotes a new `model_version`. Then:

```bash
.venv/bin/python -m verify.verifier_cli cert.json
```

```
  ok  signature valid (Ed25519)
  ok  absence proof valid against dataset_root 933d27c9b543f3f7… [sparse Merkle (smt-256)]
  ok  subject 42818a91740bb31c… absent from shard 1 at model_version shard1-6e186637f7da
VERIFIED
```

</details>

---

## What makes it hold up

### Absence proofs that name nobody

A sorted Merkle tree proves absence by handing over the target's two
neighbours — so every certificate discloses two other subjects' identifiers,
and an auditor collecting certificates slowly assembles a population census.

`verify/smt.py` uses a sparse Merkle tree instead. The `subject_ref` *is* the
path through 2²⁵⁶ positions; absence means the leaf at your own position is
empty. Siblings are subtree hashes — they commit to other subjects without
naming them.

| | sorted Merkle | sparse Merkle |
|---|---|---|
| Subject refs disclosed per proof | **2** | **0** |
| Proof size (400-subject shard) | 2 leaves + paths | 6–9 siblings |
| Verification | 0.2 ms | 0.23 ms |

Both numbers are asserted in `test_proof_names_no_other_subject`, not claimed
here. → [ADR 0007](docs/adr/0007-sparse-merkle-and-serving-latency.md)

### Deterministic retraining

An auditor verifies a rebuild by re-running it and comparing digests — which
only works if a rebuild is a pure function of its inputs. Seeding does not get
you there. Float addition is not associative and OpenMP splits reductions by
thread count, so the same shard on 4 cores and 16 cores yields different
weights.

Thread counts are pinned, `PYTHONHASHSEED` is enforced, the base image is
digest-pinned, and every dependency is `==`. Determinism is scoped to a
`code_digest` — a torch upgrade may change weights; it may not change them for
a fixed image. → [ADR 0003](docs/adr/0003-cpu-only-determinism.md)

### Serving latency

Phase 5 assumed the serving cost was ensembling across shards and prescribed
batching. Measurement disagreed:

| Per `/v1/predict` | Before | After |
|---|---:|---:|
| Fresh `psycopg2.connect` | 6.22 ms | pooled |
| Load 5 checkpoints from disk | 2.63 ms | cached |
| **Ensemble forward passes** | **0.15 ms** | batched |
| **p50 / p99** | **13.3 / 45.9 ms** | **3.5 / 9.8 ms** |

The forward pass was 1.7% of the request. Optimising it — via ONNX, a Rust
gateway, whatever — would have left 8.85 ms of connection setup and file I/O
untouched. → [ADR 0007](docs/adr/0007-sparse-merkle-and-serving-latency.md)

The ensemble cache is keyed on the promoted version's checkpoint hashes, so a
rebuild changes the key and the stale entry becomes unreachable. That is a
correctness decision: a hand-invalidated cache eventually misses one, and the
failure mode is a model that keeps scoring with erased data in it while every
job row says `done`.

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │  config/determinism.py       │
                        │  everything downstream        │
                        │  depends on this holding      │
                        └───────────────┬──────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐            ┌────────────────────┐          ┌──────────────────┐
│   engine/     │            │     gateway/       │          │     verify/      │
│  offline, no  │            │  stateless HTTP    │          │  zero imports    │
│  network      │            │  never trains      │          │  from engine/,   │
│               │            │                    │          │  gateway/, or db │
│  sharder      │            │  auth  idempotency │          │                  │
│  slicer       │            │  routes/           │          │  smt   merkle    │
│  preprocessing│            └─────────┬──────────┘          │  manifest  sign  │
│  train        │                      │                     │  verifier_cli    │
│  rebuild ─────┼──────────┐           ▼                     └────────▲─────────┘
└───────────────┘          │  ┌─────────────────┐                     │
                           │  │   Postgres      │                     │
        ┌──────────────────┘  │  queue + audit  │      certificate ───┘
        ▼                     └────────▲────────┘
┌───────────────┐                      │
│   worker/     │──────────────────────┘
│  no ingress   │   claim-then-commit, lease + reaper
│  never serves │   batches jobs per shard: one retrain, many erasures
└───────────────┘
```

`verify/` is isolated on purpose, and a test enforces it by copying the
directory somewhere bare and running it there. **A verifier that needs the
training system to run is not a proof** — it is the operator asserting their
own compliance and shipping a script that agrees.

---

## What the certificate proves — and what it doesn't

**Proves.** The subject is absent from the record set whose root the
certificate names, and the certificate was signed by the holder of the private
key.

**Does not prove.** That `result_weights` was trained on that record set.
Nothing binds the two. An operator could purge the record, publish a clean
root, and ship the previous checkpoint — the signature would still verify,
because a signature over a false claim is a valid signature.

That gap is closed by `code_digest` plus re-running a sample of completed
rebuilds. Sample selection is `HMAC(erasure_id, audit_key)` rather than the
worker's choice, so an operator cannot steer it away from jobs it would rather
not have re-run.

**The verifier prints this limitation on every successful run.** An auditor
should not have to read the source to learn what they were not shown.

---

## Known gaps

Stated rather than buried. Both surfaced from running the system, not from
reading it.

**The reproducibility spot-check selects but does not yet re-run.** Sample
selection is real and tested; the second rebuild needs pre-purge shard state
that nothing snapshots. Writing a fabricated `matched=true` row would corrupt
the one table a drift alert depends on, so nothing is written instead of
something false.

**Offline mutation is not transactional with Postgres.** A DB failure after a
successful rebuild needs manual reconciliation, not a retry — the retry would
call into a routing table that no longer lists the subject and fail with an
error that hides the real problem. The worker fails loudly and immediately and
says so. → [ADR 0006](docs/adr/0006-content-addressed-checkpoints-and-claim-then-commit.md)

**Scope.** UnlearnShield erases a subject from the **model**. Not from your
data lake, feature store, or backups — those are upstream and owned elsewhere.
A compliance story covering only the model is incomplete; this is the part it
covers.

---

## Roadmap

| Phase | Status | |
|---|:---:|---|
| 0 — Bootstrap | done | Repo, schema, digest-pinned image, CI |
| 1 — Determinism | done | Thread pinning, hash-seed enforcement, spot-check harness |
| 2 — Engine | done | Shard, slice, per-shard preprocessing, train, rebuild |
| 3 — Verification | done | Sparse Merkle proofs, canonical manifests, Ed25519, standalone verifier |
| 4 — Gateway & worker | done | FastAPI, Postgres queue with leases, idempotent intake |
| 5 — Serving | done | Connection pooling, ensemble cache, batched inference |
| 6 — Ops dashboard | next | Queue depth, SLA countdown, certificate viewer |
| 7 — Hardening | | Spot-check re-run, load ceiling, incident runbooks |

Evaluated and deliberately **not** built, with reasoning in
[docs/roadmap-assessment.md](docs/roadmap-assessment.md): GBDT/XGBoost SISA
(genuinely valuable, largest real item — deferred, not dismissed), ZK-SNARK
proofs, graph/vector unlearning, dynamic re-sharding, Kafka and warehouse CDC
connectors, ONNX and Rust inference.

---

## Testing

```bash
PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q                    # 125, no database

docker compose up -d postgres
DATABASE_URL=postgresql://unlearnshield:unlearnshield@localhost:55432/unlearnshield \
  .venv/bin/python -m pytest tests/integration tests/e2e -q                  # 19, real Postgres
```

Integration and e2e tests **skip** rather than fail when Postgres is
unreachable, so the unit suite stays fast and hermetic anywhere while the full
suite runs wherever a database exists.

| Suite | Guards |
|---|---|
| `test_determinism` | Thread pinning, hash-seed enforcement, digest reproducibility |
| `test_smt` | Absence proofs, forged and dropped siblings, the zero-leak claim |
| `test_merkle` | Legacy sorted-tree proofs — certificates outlive their issuing code |
| `test_batched_ensemble` | Batched output equals sequential to 1e-6; cache keying |
| `test_rebuild` | Purge, rollback, certificate emission, one-slice-per-subject invariant |
| `test_preprocessing_isolation` | Per-shard constants; no cross-shard statistical leak |
| `test_verifier_isolation` | `verify/` runs in a bare directory with no training system |
| `test_worker_queue` | `SKIP LOCKED` claiming, lease expiry, reaper |
| `test_e2e_erasure` | Enqueue → worker → certificate → verify → predict reflects it |

---

<div align="center">

**[Implementation plan](docs/implementation-plan.md)** ·
**[Plan corrections](docs/plan-corrections.md)** ·
**[Decision records](docs/adr/)** ·
**[Roadmap assessment](docs/roadmap-assessment.md)**

MIT

</div>
