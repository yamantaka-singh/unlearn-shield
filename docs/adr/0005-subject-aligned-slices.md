# 0005 — Slices are cut at subject boundaries, ordered by churn

**Status:** accepted, 2026-08-21
**Amends:** Phase 2b of the implementation plan

## Context

Textbook SISA slices a shard by record and rolls back to the checkpoint before
the deleted record's slice. The deletion unit there is one data point.

Here the deletion unit is a **subject**, who owns many records. Under
record-level slicing a subject's rows scatter across slices, and the rollback
point is the *minimum* slice index among them. For k records spread over n
slices, E[min] ≈ n/(k+1). The synthetic corpus in `data/synth.py` has subjects
owning up to 30 records against 5 slices, which puts the expected rollback point
at slice 0 — a full-shard retrain for essentially every deletion.

Slicing exists to avoid exactly that. Applied at record level, against a
subject-shaped deletion unit, it provides nothing.

## Decision

Cut slices at subject boundaries. Every record a subject owns lands in one slice,
so `min_slice_idx == max_slice_idx` and rollback costs the intended `(n-i)/n`.
`tests/unit/test_rebuild.py::test_every_subject_occupies_exactly_one_slice`
enforces this against real shard files.

Order subjects within a shard by **churn score ascending**, so the subjects most
likely to be deleted sit in the last slice where rollback is cheapest. The plan
used record recency as a proxy for this because PaySim has no churn signal;
`data/churn_score.py` synthesises one, so we key on it directly instead.

Pack subjects into slices targeting equal **record** counts rather than equal
subject counts. Subjects own wildly uneven numbers of records, and equal-subject
slices make per-slice training time just as uneven.

## Consequences

- `min_slice_idx` in `db/schema.sql` stays, and becomes an assertable invariant
  rather than defensive generality.
- A single subject owning a very large share of a shard's records distorts slice
  packing. Not handled: at that point the subject probably warrants its own
  shard. Worth a guard if real data shows it.
- Fraud excision remains the expensive case. A ring detected late may have been
  ingested early and scored low-churn, landing in slice 0. Ordering by churn does
  not help there, and no ordering can — the signal that would place them late
  did not exist at ingest. Phase 7's load test should measure the full-shard
  retrain path, not the average.
