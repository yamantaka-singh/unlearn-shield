# Corrections to the implementation plan

Reviewed 2026-08-21 against [implementation-plan.md](implementation-plan.md).
The plan is sound in structure; these are the defects worth fixing before code
encodes them. Numbers 1–4 are correctness bugs. The rest are gaps.

## 1. `subject_shard_map.slice_idx` breaks the deletion guarantee

**Plan:** one `slice_idx` per subject, PK on `subject_ref`; `rebuild.py` rolls
back to `checkpoint_{shard}_{slice_idx - 1}`.

**Problem:** a subject owns many records, not one. PaySim rows are transactions;
a real subject has transactions scattered across many slices. If a subject has
rows in slices 1, 3 and 5 and the map stores 5, the rebuild resumes from
checkpoint 4 — which already has slices 1 and 3 baked in, including the rows
just promised to be erased. The rebuild completes, the manifest signs, and the
erasure did not happen. Nothing in the plan's Phase 2 test suite catches this,
because `test_rebuild.py` checks slices ≥ the rollback point.

**Fix:** store `min_slice_idx` — the earliest slice containing any of the
subject's rows — plus `record_count`. Rollback point is `min_slice_idx - 1`.
Applied in `db/schema.sql`. The Phase 2 test must be strengthened to assert the
target is absent from **every** slice, not every slice at or after the rollback
point.

## 2. The Phase 1 harness does not pin determinism

**Plan:** `torch.manual_seed`, `use_deterministic_algorithms(True)`, the cuDNN
flags, the TF32 flags, `CUBLAS_WORKSPACE_CONFIG`.

**Problem:** on a CPU-only build the cuDNN and TF32 flags are inert, and
`CUBLAS_WORKSPACE_CONFIG` set inside a function after import is a no-op even
where CUDA exists — it is read at CUDA init. Meanwhile the live CPU hazard is
unaddressed: float addition is not associative and OpenMP splits reductions by
thread count. Measured on torch 2.5.1, summing 2^22 floats gives
`-0x1.633c04p+9` at one thread and `-0x1.633c10p+9` at two or more. GEMM at MLP
shapes turned out thread-invariant, which is why a small forward/backward does
not surface it — but Phase 2c fits per-shard scalers with means and standard
deviations over a whole shard, and that is a large reduction.

`PYTHONHASHSEED` is a second unhandled source and cannot be fixed from inside
the process at all.

**Fix:** pin `torch.set_num_threads(1)`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`;
seed `random` and `numpy` as well as torch; raise if `PYTHONHASHSEED != 0`. Also
state that determinism is scoped **within** a `code_digest` — otherwise Phase 7's
drift alert fires on every dependency bump. Applied in `config/determinism.py`,
documented in [ADR 0003](adr/0003-cpu-only-determinism.md).

## 3. Phase 2c and Phase 5 contradict each other

**Plan:** fit preprocessing strictly per shard (2c, called the most important
correctness rule); stack shard parameters into one batched forward pass and
expect roughly `S×` savings (Phase 5).

**Problem:** per-shard preprocessing means each shard's submodel needs its own
scaled copy of the input. `torch.func.stack_module_state` plus `vmap` batches the
parameters fine, but it needs `S` differently-preprocessed copies of every input
batch. The `S×` claim counts the forward passes and ignores the `S×`
preprocessing that 2c made mandatory.

**Fix:** keep 2c — it is correct and the isolation guarantee depends on it. Drop
the `S×` assertion from Phase 5 and let the benchmark report what it reports.
Use `torch.func.stack_module_state` rather than hand-rolled stacking.

## 4. The manifest proves dataset state, not weight provenance

**Plan:** manifest binds `dataset_root`, `result_weights`, `code_digest`, signed
with Ed25519; a standalone verifier checks the signature and the absence proof.

**Problem:** the absence proof shows the subject is missing from the retained
set. Nothing binds `result_weights` to `dataset_root`. Purge the record, publish
a clean root, ship the previous checkpoint — the signature still verifies,
because a signature over a false claim is a valid signature. Phase 4's 1%
reproducibility check is the only thing closing the gap, and the operator
chooses the 1%.

**Fix:** derive spot-check selection from `HMAC(erasure_id, audit_key)` so the
worker cannot steer the sample and an auditor holding `audit_key` can verify it
was not gamed. Record every check in `reproducibility_checks`, not just failures
— a table with no rows for a period is indistinguishable from a period with no
drift otherwise. Say plainly in the README what the proof does and does not
cover; an auditor will find this whether or not it is documented.

## 5. Worker queue: no lease, no reaper

`SELECT ... FOR UPDATE SKIP LOCKED` written naively holds a transaction open
across a multi-minute rebuild, which bloats Postgres and strands the job in
`processing` forever if the worker dies. Claim and commit first, then train.
Added `lease_expires_at`, `leased_by`, `attempts`, `last_error` to
`erasure_jobs`, plus a partial index on the poll predicate.

## 6. `idempotency_key` is nullable

`TEXT UNIQUE` permits unlimited NULLs, so "Idempotency-Key required" is enforced
only by app code a retry path can bypass. Made `NOT NULL`.

## 7. The 202 response leaks a behavioural bit

`POST /v1/erasure` returns `shard` in the body. Shard is assigned from
`churn_score`, so returning it discloses a coarse signal about the subject's
churn propensity — to a caller the rest of the design assumes may log its
responses. This is the same class of mistake the plan correctly avoids by
keeping subject IDs out of URLs. Return `erasure_id`, `status` and
`sla_deadline`; keep `shard` internal.

## 8. Shard assignment must be frozen at ingest

The plan never says so. If `churn_score` is recomputed upstream and the map
follows it, every existing checkpoint's rollback point becomes meaningless.
Recorded as `assigned_at` with a comment in the schema; enforce it in Phase 2.

## 9. Slicing is tuned for the secondary use case

Recency-ascending slices make deletions cheap when deletion likelihood tracks
recency of joining — true for consent revocation. Fraud excision is the opposite
shape: a ring detected in August may have been ingested in January, landing in
slice 0 and forcing a full-shard retrain. Assumption 4 names fraud as the lead
use case, so the strategy optimises the one that is not leading.

Not a bug, and not obviously worth changing — but the plan should say the
primary use case is the expensive path, and Phase 7's load test should measure
the full-shard-retrain case rather than the average.

## 10. Phase 0's "three services up" is unbuildable

`docker compose up` cannot bring up gateway and worker at Phase 0 because
neither exists; stub services would only supply a crash loop. `docker-compose.yml`
runs Postgres and the determinism check, and grows the other two in Phase 4.

Correction: this note said Phase 4 would add them; Phase 4's commit built the
code but left `docker-compose.yml` unchanged, so `docker compose up` still
couldn't bring up the gateway or worker for two more phases. Fixed in Phase 6,
alongside the dashboard service, since fixing an ops dashboard's own compose
file was the natural moment to close out the debt rather than defer it again.
