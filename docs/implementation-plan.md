# UnlearnShield — Implementation Plan (Production, Not Hackathon)

## How to use this document

This is a phased build plan for a real system, meant to sit in context for Claude Code across sessions. Each phase is self-contained: goal, concrete file-level tasks, data/API contracts where relevant, a verification checklist, and anti-patterns to avoid — the last section exists because most of them are mistakes this exact project already made once, in an earlier design pass, and had to walk back. Phases are ordered so each one is buildable and testable before the next depends on it. Do not skip ahead to the gateway before the determinism harness (Phase 1) is green — everything downstream assumes it holds.

---

## Assumptions & Defaults

Stated up front so they're easy to challenge or override rather than silently baked in:

1. **Tabular models, CPU-trainable, minutes not hours per shard.** If the real target is transformer-scale, SISA is the wrong framework and Phase 1 will surface that immediately — treat a failed CPU-determinism check as a stop signal, not a bug to work around.
2. **Deletion volume: tens to low hundreds per day**, not tens of thousands. Above ~10k/day the sharding strategy needs to be adaptive rather than static (out of scope here — flag if volume approaches this).
3. **No real upstream integrations yet.** The SQL/feature-store/object-storage nodes of the deletion DAG are stubbed behind interfaces, not built. UnlearnShield owns the model layer only.
4. **Fraud / poisoned-data excision is the lead use case**, with GDPR/DPDP compliance as the secondary framing. This affects nothing structurally, but it affects which failure modes get test coverage first (contamination excision before consent-revocation polish).

---

## Repository Layout

```
unlearnshield/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── config/
│   ├── settings.py            # env-driven config, one place
│   └── determinism.py         # the harness from Phase 1 — imported everywhere training happens
├── db/
│   ├── schema.sql              # DDL, single source of truth
│   └── migrations/             # numbered .sql files, applied in order
├── data/
│   ├── prepare.py              # PaySim ingest → cleaned parquet
│   └── churn_score.py          # synthetic churn-likelihood feature (see Phase 2 note)
├── engine/                     # offline, no network — Phase 2
│   ├── sharder.py
│   ├── slicer.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── rebuild.py
├── verify/                     # Phase 3
│   ├── merkle.py
│   ├── manifest.py
│   ├── sign.py
│   └── verifier_cli.py         # zero imports from engine/ or gateway/ — see Phase 3 anti-patterns
├── gateway/                    # Phase 4 — stateless, never trains
│   ├── main.py
│   ├── schemas.py
│   ├── auth.py
│   ├── idempotency.py
│   └── routes/
│       ├── predict.py
│       ├── erasure.py
│       └── models.py
├── worker/                     # Phase 4 — separate deployment, no ingress
│   ├── main.py
│   ├── queue.py
│   └── jobs.py
├── inference/                  # Phase 5
│   └── batched_ensemble.py
├── dashboard/                  # Phase 6
│   └── app.py
├── scripts/
│   ├── train_baseline.py
│   ├── run_worker.py
│   └── spot_check_determinism.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/adr/
    ├── 0001-sisa-over-gradient-unlearning.md
    ├── 0002-manifest-over-hash-equality.md
    ├── 0003-cpu-only-determinism.md
    ├── 0004-per-shard-preprocessing.md
    └── 0005-postgres-queue-over-redis.md
```

---

## Tech Stack & Why (ADR Summary)

| Layer | Choice | Why | Rejected |
|---|---|---|---|
| Model | PyTorch MLP, CPU | Determinism harness holds on CPU without a fight | GPU (throughput, but non-determinism costs more than it saves at this volume) |
| Unlearning | SISA (sharded, isolated, sliced) | Exact/structural, not statistical | Gradient-ascent / influence-function unlearning (only probabilistic guarantees) |
| Verification | Signed manifest + Merkle absence proof | Provable without a second training run | Hash-equality against a clean-slate retrain (circular — costs more than the operation it verifies) |
| DB | Postgres, raw SQL (psycopg2) | One dependency does routing table + manifests + job queue; no ORM overhead for ~6 tables | SQLAlchemy ORM (unneeded abstraction for this schema size) |
| Queue | Postgres table, `FOR UPDATE SKIP LOCKED` | No new infra; correct at this volume | Redis + RQ (extra moving part for no real gain under ~1k jobs/day) |
| API | FastAPI + Pydantic | Typed schemas, async, minimal ceremony | Flask (no native validation), gRPC (no client needs it yet) |
| Signing | PyNaCl (Ed25519) | Standard, small, audited | Rolling our own |
| Dashboard | Streamlit | Internal ops tool, not the product | Building a real frontend for something ops-only |
| Deploy | Docker Compose (dev/staging), plain containers behind a load balancer (prod) | Two services (gateway, worker) plus Postgres doesn't need an orchestrator yet | Kubernetes (justified later if replica counts demand it, not before) |

Full rationale for each lives in `docs/adr/`.

---

## Phase 0 — Repo & Infra Bootstrap

**Goal:** empty but runnable skeleton; CI green on nothing.

**Tasks:**
- Scaffold the directory tree above.
- `requirements.txt`: `torch` (CPU wheel), `fastapi`, `uvicorn[standard]`, `pydantic`, `psycopg2-binary`, `pynacl`, `pandas`, `scikit-learn`, `streamlit`.
- `db/schema.sql` — write all DDL now, even for tables later phases populate (see below). One migration file per phase is fine, but the full schema should exist and be readable from day one.
- `docker-compose.yml`: `postgres`, `gateway`, `worker` services. Gateway and worker share one image, different entrypoints.
- `Dockerfile`: pin the base image digest (`python:3.11-slim@sha256:...`), pin every package version in `requirements.txt` with `==`. This is not optional — determinism in Phase 1 depends on a reproducible environment, and floating versions silently break it months later.
- `.github/workflows/ci.yml`: install deps, run `pytest tests/unit`, run `python scripts/spot_check_determinism.py --ci` (stub returns pass until Phase 1 exists).

**DB schema (write now, all tables):**
```sql
-- routing: hashed subject ref, never the raw ID
CREATE TABLE subject_shard_map (
    subject_ref   TEXT PRIMARY KEY,      -- HMAC-SHA256(subject_id, tenant_key)
    shard         INT NOT NULL,
    slice_idx     INT NOT NULL,
    tenant_id     TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE erasure_jobs (
    erasure_id    UUID PRIMARY KEY,
    subject_ref   TEXT NOT NULL,
    reason        TEXT NOT NULL CHECK (reason IN ('consent_revocation','fraud_excision')),
    shard         INT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued','processing','done','failed')),
    idempotency_key TEXT UNIQUE,
    sla_deadline  TIMESTAMPTZ NOT NULL,
    requested_by  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at  TIMESTAMPTZ
);

CREATE TABLE erasure_manifests (
    erasure_id     UUID PRIMARY KEY REFERENCES erasure_jobs(erasure_id),
    shard          INT NOT NULL,
    resumed_from   TEXT NOT NULL,   -- checkpoint hash
    dataset_root   TEXT NOT NULL,   -- merkle root, post-purge
    absence_proof  JSONB NOT NULL,
    code_digest    TEXT NOT NULL,
    config_digest  TEXT NOT NULL,
    result_weights TEXT NOT NULL,   -- sha256 of new checkpoint
    model_version  TEXT NOT NULL,
    signature      TEXT NOT NULL,
    manifest_json  JSONB NOT NULL,  -- canonical form that was signed
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_versions (
    model_version  TEXT PRIMARY KEY,
    shard_checkpoints JSONB NOT NULL,  -- {"0": "sha256:...", "1": "sha256:...", ...}
    eval_set_version  TEXT NOT NULL,
    promoted_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE checkpoints (
    checkpoint_hash TEXT PRIMARY KEY,
    shard          INT NOT NULL,
    slice_idx      INT NOT NULL,
    file_path      TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Verification checklist:**
- `docker compose up` brings up all three services with no crash loop.
- `pytest tests/unit` runs (even at zero tests, exit 0).
- CI workflow triggers on push and passes.

**Definition of done:** empty skeleton, schema applied, CI green.

---

## Phase 1 — Determinism Harness

**Goal:** prove bit-identical retraining before anything else depends on it. This is the load-bearing phase — if it doesn't hold, Phase 3's manifest is unfalsifiable in name only.

**Tasks:**
- `config/determinism.py`:
```python
import os, torch

def enforce_determinism(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cpu")   # GPU not supported until a separate ADR revisits this
```
- Every training entrypoint (`engine/train.py`, `engine/rebuild.py`) calls this first, before touching data.
- DataLoader: `num_workers=0`. If throughput later demands workers, each must get an explicit `worker_init_fn` seeding — don't add workers without it.
- `scripts/spot_check_determinism.py`: trains a small shard twice from identical inputs, hashes both `state_dict`s, asserts equality. This is the script CI runs, and the same one Phase 4's worker runs nightly on ~1% of real rebuilds (see Phase 4).

**Verification checklist:**
- `pytest tests/unit/test_determinism.py`: train twice, `sha256(state_dict bytes)` matches, on this repo's pinned Docker image specifically (not "on my laptop").
- Confirm the check fails loudly if `enforce_determinism` is skipped — write a negative test that trains without it and asserts the hashes usually differ (won't always, but should differ often enough to catch a regression; run 5x and require at least one mismatch).

**Anti-patterns to avoid:**
- Do not assume `torch.manual_seed()` alone is enough — it was the exact mistake caught last design pass. cuDNN algorithm selection, TF32, and DataLoader worker seeding all bypass it independently.
- Do not let this phase quietly target GPU "for speed" — that decision needs its own ADR and its own determinism strategy, not a shortcut here.

**Definition of done:** two independent training runs of the same shard produce byte-identical weights, verified by an automated test, on the pinned container.

---

## Phase 2 — Offline Unlearning Engine

**Goal:** sharding, slicing, per-shard preprocessing, training, and rebuild — all as a CLI, no API yet.

### 2a. Sharding (`engine/sharder.py`)

Random assignment scatters deletions evenly across every shard, which means nearly every request triggers a rebuild somewhere — the whole point of sharding is defeated. Assign by expected deletion likelihood instead: concentrate high-churn users into a small number of "hot" shards, leave long-tenure users in "cold" shards that rarely rebuild.

```python
def assign_shard(user_id: str, churn_score: float, num_shards=5, hot_shards=2, hot_threshold=0.6) -> int:
    if churn_score >= hot_threshold:
        return stable_hash(user_id) % hot_shards
    return hot_shards + (stable_hash(user_id) % (num_shards - hot_shards))
```

`churn_score` doesn't exist in PaySim — synthesize it in `data/churn_score.py` as a stand-in for real consent-type/tenure signals (e.g., derived from account-creation recency in the synthetic data), and document clearly in the module docstring that this is a placeholder for a real churn/consent-type feature a production deployment would supply.

### 2b. Slicing (`engine/slicer.py`)

Within a shard, order records by recency ascending — oldest first, most recent last — so records likely to be deleted soon land in late slices, where rollback is cheap. Uniform ordering gives ~2x savings on average; recency ordering is what makes rollback usually skip most of the shard's training time.

### 2c. Per-shard preprocessing (`engine/preprocessing.py`)

**This is the single most important correctness rule in the whole system:** fit scalers, encoders, and any vocabulary exclusively on one shard's data, never globally. A preprocessor fit on the full dataset before sharding bakes every user's values into every shard's normalization statistics — so a "deleted" user's numbers survive in every other shard's feature scale even after their row is purged from their own shard. The isolation guarantee in `engine/train.py` is false if this rule is broken anywhere.

- Fit once per shard, at shard-build time, on that shard's full data.
- On rebuild (Phase 2e), refit on the shard's remaining data after the purge — this is the one time refitting happens outside initial build, and it's cheap because it's one shard, not five.
- Store the fitted preprocessor alongside each shard's checkpoint (pickle is fine here — it's ours, not signed, not user-facing).

### 2d. Model & training (`engine/model.py`, `engine/train.py`)

- Fixed MLP architecture, identical across all shards (required for Phase 5's batched ensemble).
- `train.py`: for each shard, for each slice in order, extend the training set, train `K` epochs, checkpoint. Call `enforce_determinism(seed)` first, always.
- Checkpoint naming: `checkpoints/shard{k}_slice{i}.pt`, hash recorded in the `checkpoints` table from Phase 0's schema.

### 2e. Excision & rebuild (`engine/rebuild.py`)

Given a `subject_ref`:
1. Look up `(shard, slice_idx)` from `subject_shard_map`.
2. Mark the subject's rows purged in that shard's slice dataset.
3. Refit preprocessing on the shard's remaining data (2c).
4. Roll back to `checkpoint_{shard}_{slice_idx - 1}` (or fresh init if `slice_idx == 0`).
5. Retrain forward through the remaining slices, purged rows excluded, deterministic settings enforced.
6. Save the new checkpoint; this becomes an input to Phase 3's manifest.

**Verification checklist:**
- `pytest tests/unit/test_sharder.py`: assert hot-shard concentration — with a skewed churn distribution, >70% of high-churn synthetic users land in the 2 hot shards.
- `pytest tests/unit/test_preprocessing_isolation.py`: fit preprocessing per-shard, confirm shard A's scaler statistics are numerically unaffected by a value change in shard B's data.
- `pytest tests/unit/test_rebuild.py`: rebuild a shard with a known target purged, confirm the target's rows are absent from every slice ≥ the rollback point, and confirm the rebuild used `enforce_determinism`.
- End-to-end CLI smoke test: `python -m engine.train`, then `python -m engine.rebuild --subject <id>`, both exit 0 and produce a new checkpoint hash different from the original.

**Anti-patterns to avoid:**
- Fitting any preprocessing step globally "just to get a baseline model running faster" — this is the leak, and it's easy to reintroduce under time pressure.
- A learned aggregation head (stacking model, attention-weighted ensemble) anywhere near shard outputs. The isolation guarantee requires the combining function to be fixed (e.g., simple averaging), never trained on cross-shard data. If a future ticket proposes a smarter ensemble, that ticket needs a new ADR before it touches this code.
- Retrofitting SISA onto data that wasn't sharded from the start. It's a training-time commitment; there's no "add sharding later" path for an existing monolithic model.

**Definition of done:** a shard can be built, an excision request can be processed end-to-end via CLI, and the target's data is verifiably absent from every downstream training slice, all deterministically.

---

## Phase 3 — Verification: Manifest, Merkle Proofs, Signing

**Goal:** turn "we retrained the shard" into a portable, independently checkable claim — without training a second model to prove it.

### 3a. Merkle absence proof (`verify/merkle.py`)

Build a Merkle tree over the shard's **retained** record set, leaves sorted by `HMAC-SHA256(record_id, tenant_key)`. To prove a subject is absent:
1. Find the subject's would-be sorted position.
2. Produce inclusion proofs for its immediate predecessor and successor leaves in the tree.
3. The proof is: "these two specific leaves are both present and adjacent in sorted order, with the target's hash falling strictly between them and matching neither." A verifier can check this from the proof alone, without the full retained dataset — that's the property that makes it portable.

This is the standard sorted-Merkle-tree non-inclusion pattern (the same technique behind Certificate Transparency logs) — implement from `hashlib`, no external Merkle library needed for a tree this size.

### 3b. Manifest (`verify/manifest.py`)

```json
{
  "subject_ref": "HMAC-SHA256(subject_id, tenant_key)",
  "shard": 2,
  "resumed_from": "sha256:checkpoint_2_3",
  "dataset_root": "sha256:merkle_root_shard2_postpurge",
  "absence_proof": { "predecessor": "...", "successor": "...", "path": ["..."] },
  "code_digest": "sha256:container_image",
  "config_digest": "sha256:training_config",
  "result_weights": "sha256:model_2_v47",
  "model_version": "v47",
  "purged_at": "2026-08-21T10:00:00Z",
  "completed_at": "2026-08-21T10:04:12Z"
}
```
Serialize with sorted keys (canonical JSON) before signing — signature verification is meaningless if two semantically-identical JSON blobs hash differently due to key order.

### 3c. Signing (`verify/sign.py`)

Ed25519 via PyNaCl. Private key lives outside the repo (env var or secrets manager, never committed); public key is checked into `verify/` so the standalone verifier can ship with it.

### 3d. Standalone verifier (`verify/verifier_cli.py`)

Given a manifest JSON and the public key: verify the signature, then verify the absence proof against `dataset_root` in the manifest. **This file must not import anything from `engine/`, `gateway/`, or the database.** If the verifier needs access to the training system to run, it stops being a proof and becomes an assertion — the whole value of this component is that an auditor can run it standalone.

**Verification checklist:**
- `pytest tests/unit/test_merkle.py`: build a tree, produce an absence proof for a genuinely absent record, confirm the verifier accepts it; produce a false absence proof for a record that's actually present, confirm the verifier rejects it.
- `pytest tests/unit/test_manifest.py`: round-trip a manifest through canonical serialization, sign, verify; tamper with one field post-signing, confirm verification fails.
- Run `verify/verifier_cli.py` in a fresh Python environment with zero access to `engine/` or the database — confirm it still works against a manifest produced elsewhere.

**Anti-patterns to avoid:**
- Reintroducing hash-equality-against-a-clean-slate-retrain as "extra assurance." It was rejected specifically because it costs a full second training run per deletion — don't let it creep back in as an "optional" verification mode; if stronger evidence is wanted, the reproducibility spot-check in Phase 4 (re-running ~1% of jobs) is the right tool, not a per-request shadow retrain.
- Letting `verifier_cli.py` accumulate a dependency on the gateway's DB models "just to look up one thing" — that's the exact coupling this file exists to avoid.

**Definition of done:** any manifest produced by `engine/rebuild.py` can be independently verified by a standalone script with no access to the training system, and tampering with any manifest field is detected.

---

## Phase 4 — Gateway & Worker

**Goal:** an API surface that never trains, and a worker that never serves.

### 4a. Routes (`gateway/routes/`)

```
POST   /v1/predict
POST   /v1/erasure                    -> 202
GET    /v1/erasure/{erasure_id}
GET    /v1/erasure/{erasure_id}/certificate
POST   /v1/erasure/attest
GET    /v1/models/current
GET    /v1/models/{version}/manifest
```

- **`POST /v1/erasure` returns 202, never 200**, body `{erasure_id, status: "queued", shard, sla_deadline}`. It enqueues a row in `erasure_jobs`; it does not wait on a rebuild. A synchronous design here will time out under real load and gives the demo a latency number the production system can't deliver.
- **`Idempotency-Key` header required.** Look it up in `erasure_jobs.idempotency_key` before inserting; a replay returns the original `erasure_id` instead of enqueuing a duplicate rebuild.
- **Subject IDs never appear in a URL path or query string** — they land in ingress logs, APM traces, and CDN logs, creating fresh copies of the exact identifier being erased. `POST /v1/erasure/attest` takes the subject in the request body for this reason; `erasure_id` is an opaque UUID with no derivation from the subject.
- **`POST /v1/predict` returns `model_version` in every response.** Without it, "which model scored this applicant, and had subject X been erased yet?" is unanswerable later, and that question shows up in disputes.

### 4b. Auth scopes (`gateway/auth.py`)

- `predict:invoke` — serving path.
- `erasure:write` — intake (service-to-service; end users hit a consent manager, which calls this).
- `erasure:attest` — auditor-facing read.

### 4c. Worker (`worker/`)

- Polls `erasure_jobs` with `SELECT ... FOR UPDATE SKIP LOCKED` — this is the entire queue; no Redis, no broker.
- Groups queued jobs by shard, processes as a windowed batch rebuild per shard rather than one rebuild per job — one nightly rebuild of a hot shard can discharge dozens of queued requests at once.
- Trigger conditions: a scheduled sweep (e.g., every N hours) **and** a forced trigger if any job's `sla_deadline` is within a configurable buffer.
- After each rebuild: writes the manifest (Phase 3), updates `erasure_jobs.status`, updates `model_versions` if this rebuild produces a new promotable model.
- **Reproducibility spot-check**: on ~1% of completed jobs, re-run the rebuild and compare weight hashes; alert on drift. This is where the "prove bit-identical" guarantee from Phase 1 earns its keep in production, without paying for it on every single job.

**Verification checklist:**
- `pytest tests/integration/test_gateway_routes.py`: erasure request returns 202 with correct shape; predict includes `model_version`.
- `pytest tests/integration/test_idempotency.py`: two requests with the same `Idempotency-Key` produce one job.
- `pytest tests/e2e/test_e2e_erasure.py`: enqueue → worker processes → manifest exists → `verifier_cli.py` confirms absence → `/predict` reflects the change → `/v1/erasure/{id}` reports `done`.
- Load a URL containing a raw subject ID into a log aggregator search and confirm it returns nothing — i.e., confirm no code path put one there.

**Anti-patterns to avoid:**
- Any code path where an HTTP request thread blocks on `engine.rebuild`. If a route handler imports from `engine/`, that's a sign the boundary broke.
- Storing raw subject IDs anywhere in `erasure_jobs` or logs — only `subject_ref` (the HMAC), consistent with the routing table.
- A gateway replica performing training. Gateway is horizontally scaled and stateless; worker is not.

**Definition of done:** an erasure request submitted through the API is durably queued, processed asynchronously, produces a verifiable manifest, and is reflected in subsequent predictions — all without a subject ID ever touching a URL, and without any HTTP request waiting on a training run.

---

## Phase 5 — Serving Realism

**Goal:** the parts that don't show up in a demo but decide whether this survives a platform team's review.

- **Batched ensemble inference (`inference/batched_ensemble.py`)**: since all shard sub-models share one architecture (Phase 2d), stack their parameters and run one batched forward pass instead of five sequential ones. Benchmark before/after; this is the single most likely rejection reason if skipped, since naive sequential ensembling costs roughly `S×` latency.
- **Eval-set versioning**: the validation set contains real users, some of whom get erased. An un-versioned eval set means an AUC drop after a rebuild could be the eval set changing, not the model regressing — indistinguishable without versioning. Store `eval_set_version` alongside `model_version` (already in the Phase 0 schema); regenerate the eval set's purge-state alongside every shard rebuild that touches it.
- **Alert-and-proceed rebuild gate**: normal MLOps blocks promotion on a metrics regression; compliance requires the erasure to land regardless of accuracy impact. Implement the gate as alert-and-proceed — log a flagged regression, page the on-call/risk channel, do not block the rebuild. SISA's minority-class degradation (documented in the ADRs) means this will fire for real on fraud-label shards; write the runbook now, not after the first page.

**Verification checklist:**
- Benchmark script comparing sequential vs. batched ensemble inference latency at a realistic batch size; batched wins by roughly `S×` minus overhead.
- `pytest tests/unit/test_eval_versioning.py`: confirm an eval-set version bump is recorded on every rebuild that purges an eval-set member.
- Simulate a rebuild that regresses minority-class accuracy past threshold; confirm it completes and alerts, and does not block.

**Definition of done:** inference latency is production-viable under the ensemble, metric regressions are attributable to a specific cause, and rebuild gating never silently blocks a legally required erasure.

---

## Phase 6 — Dashboard (Ops Tool, Not the Product)

**Goal:** internal visibility, not a customer-facing surface.

- Streamlit app (`dashboard/app.py`) reading directly from Postgres (read-only role): queue depth per shard, SLA countdown per pending job, manifest viewer, accuracy-delta chart per rebuild.
- No write actions from the dashboard beyond a manual "force rebuild now" button that inserts into `erasure_jobs` through the same path the API uses — no direct DB writes from the UI layer.

**Definition of done:** an operator can see queue health and every erasure's status/manifest without touching SQL directly.

---

## Phase 7 — Hardening

**Goal:** the unglamorous pass before this touches real data.

- Security review of the auth scopes and the idempotency-key handling.
- Load test the worker's batch-rebuild path at 2–3x the assumed volume (Assumption 2) to find the point where the Postgres queue stops being sufficient — know the ceiling, don't discover it in production.
- Confirm container digest pinning survives a `docker compose build --no-cache` (catches silently-floated base image tags).
- Re-run the full determinism spot-check suite (Phase 4) at increased sample rate for one week before any real cutover, then drop back to steady-state ~1%.
- Write the incident runbook for: queue backlog exceeding SLA, determinism drift alert, minority-class regression alert.

**Definition of done:** the system has a known breaking point, a known recovery procedure for each alert type, and has run its own reproducibility check against itself for a week without drift.

---

## Testing Strategy Summary

| Layer | Tool | What it catches |
|---|---|---|
| Unit | pytest | Sharding logic, preprocessing isolation, Merkle proofs, manifest signing |
| Integration | pytest + docker-compose test stack | Route contracts, idempotency, queue behavior |
| E2E | pytest, full stack | Enqueue → rebuild → manifest → verify → predict reflects change |
| Determinism | dedicated harness + nightly spot-check | Silent non-determinism creep (GPU flags, library upgrades, DataLoader changes) |
| Load | locust or k6, Phase 7 only | Queue throughput ceiling |

---

## Milestones (rough, 1–2 engineers)

| Phase | Estimate |
|---|---|
| 0 — Bootstrap | 2–3 days |
| 1 — Determinism | 3–5 days (do not compress this one) |
| 2 — Engine | 1.5–2 weeks |
| 3 — Verification | 1–1.5 weeks |
| 4 — Gateway & Worker | 1.5–2 weeks |
| 5 — Serving realism | 1–1.5 weeks |
| 6 — Dashboard | 2–3 days |
| 7 — Hardening | 1–2 weeks |
| **Total** | **~9–12 weeks** |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GPU pressure later for throughput | Revisit as its own ADR with a real determinism strategy — don't bolt it onto Phase 1's CPU-only assumption |
| SISA accuracy loss concentrated on fraud/minority class | Surfaced deliberately in Phase 5's alert-and-proceed gate, not hidden |
| Deletion volume exceeds Postgres-queue comfort zone | Load-tested in Phase 7 before it's a surprise; Redis/RQ migration path stays open if needed |
| Merkle absence-proof bugs (subtle, high-consequence) | Explicit adversarial test in Phase 3 (false-proof rejection), not just happy-path coverage |
| DPDP enforcement timeline slips further | Lead positioning stays fraud/model-risk (Assumption 4); compliance framing is upside, not the load-bearing pitch |

---

## Open Questions to Resolve During Build, Not Before

- Real churn/consent-type signal to replace the synthetic `churn_score` stand-in once there's a real data source.
- Multi-tenant key management for the HMAC/signing keys, once this serves more than one fintech customer.
- Whether `erasure:attest` should be rate-limited per subject_ref to prevent enumeration probing.

These don't block starting Phase 0 — they block scaling past the first real deployment.
