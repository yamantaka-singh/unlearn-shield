# 0006 — Content-addressed checkpoint storage, and claim-then-commit queueing

**Status:** accepted, 2026-08-21
**Amends:** Phase 0's `checkpoints` table, Phase 4's worker design

## Context

Building the worker against a real Postgres instance (rather than assuming the
schema and route contracts from the plan alone) surfaced two defects neither
code review nor unit tests would have caught, because both only appear once a
shard is rebuilt more than once.

## Defect 1: checkpoint files are silently overwritten

`engine/train.py` writes each slice checkpoint to a path keyed only by
`(shard, slice_idx)` -- it has to, since resuming mid-rebuild needs a stable
name. But that means the second rebuild of any shard overwrites the first
rebuild's file at that same path. Phase 0's schema recorded a `checkpoint_hash`
per rebuild (correctly, content-addressed) alongside a `file_path` column
pointing at that mutable, conventional path.

The result: a `checkpoints` row from an earlier rebuild ends up with a
`file_path` that no longer contains the bytes its `checkpoint_hash` describes.
Nothing detects this — the row still exists, `predict.py` would still load
*something* from that path — but it silently isn't the model the row claims it
is, the moment a later rebuild of that shard runs.

This was caught by running the actual worker against real jobs twice in a row
against the same shard, the exact scenario a schema unit test wouldn't exercise
without deliberately simulating repeated rebuilds.

**Fix:** `worker/jobs.py::_promote` (and the equivalent bootstrap step in
`scripts/load_routing.py`) copy the finished checkpoint into
`checkpoints/cas/{hash}.pt` and record *that* path in the `checkpoints` table.
`gateway/routes/predict.py` reads `file_path` from the DB rather than
reconstructing `checkpoint_path(shard, slice_idx)` itself — reconstructing it
is exactly the mistake that silently loads the wrong file.

A companion schema bug: `checkpoints_shard_slice_idx`, a unique index on
`(shard, slice_idx, code_digest)`, assumed only one checkpoint would ever exist
per shard/slice/image. That's wrong by design — every rebuild produces a new
row for the same triple, and that accumulation is the point. The index made the
second rebuild of any shard fail outright with a unique-violation. Dropped to a
plain (non-unique) index; `checkpoint_hash` is already the correct PK.

## Defect 2: partial failure between offline mutation and DB bookkeeping

`engine.rebuild_batch_by_ref` purges the shard file, retrains, and rewrites
`routing.json` — all direct file I/O, none of it transactional with Postgres.
If the worker's subsequent DB writes (manifest insert, promotion, status
update) fail for any reason, the offline mutation has already happened and
cannot be rolled back, but the DB has no record of it. A naive retry then calls
`rebuild_batch_by_ref` again for a subject `routing.json` no longer lists,
producing a confusing `KeyError` that hides the real, already-resolved problem
behind what looks like a routing bug.

This is not hypothetical: it happened during this build, from the
`checkpoints_shard_slice_idx` bug above, and required manual reconciliation
(regenerating the manifest for the already-completed rebuild rather than
retrying it).

**Fix, scoped to what Phase 4 needs:** the per-shard bookkeeping block in
`worker/jobs.py::process_claimed` is wrapped in its own exception handler, so a
bookkeeping failure marks the affected jobs `'failed'` immediately with an
error that says plainly the data-side work is already done and a blind retry
is wrong. This does not make the operation atomic — a true fix needs the
offline mutation and the DB write to share a commit boundary (a write-ahead
staging step, or moving shard state into Postgres entirely), which is a
storage redesign out of scope for wiring the gateway and worker together.
Flagged for Phase 7.

## Also fixed along the way

- `claim_batch`'s `UPDATE ... RETURNING` does not actually preserve the
  subquery's `ORDER BY shard, sla_deadline` — Postgres gives no such guarantee.
  The function now sorts the fetched rows by shard itself before returning,
  rather than asserting an ordering property that wasn't real.
- `psycopg2.extras.register_uuid()` returns `erasure_id` as a `uuid.UUID`;
  `claim_batch` now stringifies it at the boundary so every downstream
  consumer (SQL params, dict keys, the HMAC input for spot-check sampling)
  sees the same string form the API returns.

## Consequences

- Every promoted checkpoint has a permanent, content-addressed copy. Disk usage
  grows with every rebuild rather than staying flat; acceptable at the volumes
  in Assumption 2, worth revisiting with a retention policy before it isn't.
- A bookkeeping failure after a successful rebuild still requires manual
  reconciliation today. It fails loudly and immediately instead of silently
  stalling for a full lease period and then failing confusingly on retry, but
  it is not yet self-healing.
