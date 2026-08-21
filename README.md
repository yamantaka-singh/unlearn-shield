<div align="center">

# 🛡️ UnlearnShield

**Structural machine unlearning for tabular fraud models**

When a record has to come out of a trained model — a poisoned batch, a fraud ring's history, revoked consent — UnlearnShield rebuilds only the affected shard and emits a signed manifest an auditor can verify without access to the training system.

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-29_passing-22c55e?style=for-the-badge)](#testing)

</div>

---

## The Problem

Regulations (GDPR, DPDP) and operational hygiene (fraud ring excision, poisoned data cleanup) demand the ability to **remove a subject's influence from a trained model**. Gradient-ascent "unlearning" only gives probabilistic guarantees — the model is nudged away from deleted rows, not rebuilt without them. For an auditor, "*probably* forgot" is not an answer.

## The Approach

UnlearnShield uses **SISA** (Sharded, Isolated, Sliced, Aggregated training). Removal is **structural**, not statistical:

```
┌──────────────────────────────────────────────────────────────────┐
│                         Training Data                            │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│ Shard 0  │ Shard 1  │ Shard 2  │ Shard 3  │ Shard 4              │
│ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │ ┌──────┐             │
│ │Slice0│ │ │Slice0│ │ │Slice0│ │ │Slice0│ │ │Slice0│             │
│ │Slice1│ │ │Slice1│ │ │Slice1│ │ │Slice1│ │ │Slice1│             │
│ │Slice2│ │ │Slice2│ │ │Slice2│ │ │Slice2│ │ │Slice2│             │
│ │Slice3│ │ │Slice3│ │ │Slice3│ │ │Slice3│ │ │Slice3│             │
│ │Slice4│ │ │Slice4│ │ │Slice4│ │ │Slice4│ │ │Slice4│             │
│ └──────┘ │ └──────┘ │ └──────┘ │ └──────┘ │ └──────┘             │
└──────────┴──────────┴──────────┴──────────┴──────────────────────┘
                                ↓ Subject erasure request
                    ┌───────────────────────┐
                    │ 1. Identify shard      │
                    │ 2. Purge subject rows  │
                    │ 3. Roll back to ckpt   │
                    │ 4. Retrain forward     │
                    │ 5. Sign manifest       │
                    └───────────────────────┘
```

A subject in slice 4 costs one slice of retraining. A subject in slice 0 costs the whole shard. High-churn subjects are placed in later slices so erasure is cheap. That spread is the point.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Deterministic Retraining** | Bit-identical weights across runs on a pinned image. An auditor re-runs the rebuild and compares digests — no second full training run needed. |
| **Merkle Absence Proofs** | RFC 6962 sorted Merkle tree proves a `subject_ref` is *not* in the retained set. Domain-separated hashing, no odd-node duplication (avoids CVE-2012-2459). |
| **Signed Manifests** | Ed25519 signatures over canonical JSON bind `subject_ref`, `dataset_root`, `code_digest`, `config_digest`, and `result_weights` into one verifiable document. |
| **Subject-Aligned Slices** | Slices cut at subject boundaries, not record boundaries. A 30-record subject doesn't scatter across all slices, forcing a rollback to slice 0 every time. |
| **HMAC Subject References** | Raw subject IDs never leave ingest. All downstream references are `HMAC-SHA256(subject_id, tenant_key)` — not a bare hash trivially reversible by enumeration. |
| **Churn-Ordered Placement** | High-churn subjects land in later slices where erasure is cheapest. Slice ordering is frozen at ingest so checkpoints stay valid. |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────┐
                    │           config/                 │
                    │  determinism.py  ←  everything    │
                    │  settings.py        depends on    │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌───────────┐ ┌──────────────────┐
     │    engine/      │ │   data/   │ │     verify/       │
     │  sharder.py     │ │ synth.py  │ │  merkle.py        │
     │  slicer.py      │ │ churn_    │ │  manifest.py      │
     │  preprocessing  │ │ score.py  │ │  sign.py          │
     │  model.py       │ └───────────┘ │  verifier_cli.py  │
     │  train.py       │               └──────────────────┘
     │  rebuild.py     │                 ▲  no imports from
     └────────────────┘                 │  engine/ or db/
              │                         │
              └────── manifest ─────────┘
```

> **`verify/` is deliberately isolated** — it has zero imports from `engine/`, `gateway/`, or the database. An auditor can run the verifier CLI without access to the training system.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install & Run

```bash
# Clone
git clone https://github.com/yamantaka-singh/unlearn-shield.git
cd unlearn-shield

# Environment
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt

# Run tests (PYTHONHASHSEED is required — the harness refuses to start without it)
PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q
```

### Build & Erase

```bash
# Partition data into 5 shards and train
export PYTHONHASHSEED=0
.venv/bin/python -m engine.train --build

# Erase a subject — rebuilds only the affected shard
.venv/bin/python -m engine.rebuild --subject C0000042
```

The rebuild reports which slices were retrained and outputs the signed manifest.

### Docker (Determinism-Verified)

The determinism guarantee holds on the pinned Docker image, not on arbitrary hosts:

```bash
docker compose run --rm determinism python -m pytest tests/unit -q
```

---

## 📁 Project Structure

```
unlearn-shield/
├── config/
│   ├── determinism.py         # Pins seeds, thread counts, PYTHONHASHSEED check
│   └── settings.py            # Env-driven config, HMAC subject_ref
├── data/
│   ├── synth.py               # PaySim-shaped synthetic generator
│   └── churn_score.py         # Churn signal (placeholder, clearly marked)
├── engine/
│   ├── sharder.py             # Hot/cold shard assignment, frozen at ingest
│   ├── slicer.py              # Subject-aligned slices, churn-ordered
│   ├── preprocessing.py       # Per-shard constants, fit on slice 0 only
│   ├── model.py               # Fixed MLP, identical across shards
│   ├── train.py               # Build + incremental slice training with checkpoints
│   └── rebuild.py             # Purge → rollback → retrain → manifest
├── verify/
│   ├── merkle.py              # RFC 6962 sorted Merkle tree + absence proofs
│   ├── manifest.py            # Canonical JSON serialisation, no engine imports
│   ├── sign.py                # Ed25519 signing
│   └── verifier_cli.py        # Standalone auditor tool
├── db/
│   ├── schema.sql             # Full DDL (routing, jobs, manifests, audit log)
│   └── migrations/            # Numbered SQL migrations
├── tests/unit/                # 29 tests covering determinism, merkle, rebuild, etc.
├── docs/
│   ├── implementation-plan.md # 7-phase build plan
│   ├── plan-corrections.md    # Defects found and what changed
│   └── adr/                   # Architecture Decision Records
├── Dockerfile                 # Digest-pinned base image
├── docker-compose.yml         # Postgres + determinism service
└── requirements.txt           # Pinned with == (determinism depends on it)
```

---

## 🔬 Why Determinism Comes First

The product is a manifest saying a shard was retrained without a subject's data. Nothing about a weight tensor reveals whether that is true. The only affordable audit is to **re-run the rebuild and compare digests** — which works only if a rebuild is a pure function of its inputs.

Seeding alone doesn't get you there:

- **Float addition is not associative** — OpenMP splits reductions by thread count, so the same shard on 4 cores vs. 16 cores yields different normalisation constants and different weights.
- **Thread counts are pinned to 1** — not for performance, but because `OMP_NUM_THREADS=1` is the only value that makes the output a function of the data alone.
- **Base image is digest-pinned** — `python:3.11-slim@sha256:...`, not `:latest`. A BLAS update changes float reduction order.
- **Dependencies are pinned with `==`** — a floated minor version silently breaks determinism months later.

Determinism is scoped to a `code_digest`, not asserted across versions. A torch upgrade is allowed to change weights — it is not allowed to change them for a fixed image digest.

> **See:** [ADR 0003 — CPU-Only Determinism](docs/adr/0003-cpu-only-determinism.md) for measured numbers.

---

## 🔐 What the Manifest Proves (and What It Doesn't)

**Proves:**
- A `subject_ref` is absent from the shard's retained record set (Merkle absence proof)
- The training code and config that produced the weights (via `code_digest` and `config_digest`)
- The exact weights that resulted (via `result_weights` digest)

**Does not prove:**
- That `result_weights` were actually produced from `dataset_root` — a pipeline could purge the record, publish a clean root, and ship the *previous* checkpoint. The signature would still verify, because a signature over a false claim is still a valid signature.

Weight provenance rests on two things beyond the signature:
1. **`code_digest`** recorded in the manifest
2. **Spot-check re-runs** — sample selection is `HMAC(erasure_id, audit_key)`, not chosen by the worker, so an operator cannot steer the sample away from jobs it would rather not have re-run

---

## ⚠️ Scope Boundary

UnlearnShield erases a subject from the **model**. It does not erase them from your data lake, feature store, or backups. Those sit behind interfaces and are owned by the upstream system. A compliance story that only covers the model is incomplete — this is the part it covers.

---

## 📝 Lessons Learned

Two design bugs were found during Phase 2 that would have caused erasure to silently fail:

**1. Record-level slicing buys nothing here**
SISA slices by record because its deletion unit is one data point. Ours is a subject who owns many records — their rows scatter across slices and the rollback point becomes slice 0 every time. Fix: slices cut at subject boundaries.
→ [ADR 0005](docs/adr/0005-subject-aligned-slices.md)

**2. Refitting preprocessing on rebuild reintroduces the leak**
Refit scalers on the shard's remaining data, then resume from a checkpoint trained under the *old* scaling constants. The subject's influence survives in the mismatch. Fix: fit on slice 0 only, immutably.
→ [ADR 0004](docs/adr/0004-per-shard-preprocessing.md)

---

## 🗺️ Roadmap

| Phase | Status | Description |
|:---:|:---:|---|
| 0 | ✅ Done | Repo scaffold, Docker, CI |
| 1 | ✅ Done | Determinism harness |
| 2 | ✅ Done | Offline unlearning engine (shard, slice, train, rebuild) |
| 3 | 🔲 Next | Manifest signing & Merkle verification pipeline |
| 4 | 🔲 | API gateway + async worker (FastAPI, Postgres queue) |
| 5 | 🔲 | Inference — batched ensemble across shards |
| 6 | 🔲 | Ops dashboard (Streamlit) |
| 7 | 🔲 | Hardening — SLA enforcement, lease reaping, monitoring |

> Full plan: [docs/implementation-plan.md](docs/implementation-plan.md) · Corrections: [docs/plan-corrections.md](docs/plan-corrections.md)

---

## 🧪 Testing

```bash
# Unit tests (29 tests across 6 files)
PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/unit -q

# Determinism spot-check on pinned image
docker compose run --rm determinism python scripts/spot_check_determinism.py
```

| Test Suite | Covers |
|---|---|
| `test_determinism.py` | Seed pinning, PYTHONHASHSEED enforcement, digest reproducibility |
| `test_merkle.py` | RFC 6962 tree, domain separation, absence proofs, adjacency checks |
| `test_rebuild.py` | End-to-end purge + retrain, manifest correctness |
| `test_preprocessing_isolation.py` | Scalers fit on slice 0 only, rebuild doesn't refit |
| `test_sharder.py` | Hot/cold assignment, frozen routing |
| `test_slicer.py` | Subject-aligned slice boundaries |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Ensure tests pass: `PYTHONHASHSEED=0 pytest tests/unit -q`
4. Submit a pull request

When modifying the training pipeline, run the determinism spot-check on the pinned Docker image — a change that passes locally but fails in the container has changed float reduction order.

---

## 📄 License

[MIT](LICENSE) — Copyright © 2026 Mrityunjay
