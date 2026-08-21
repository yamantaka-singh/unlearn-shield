# 0004 — Preprocessing is fitted per shard, on slice 0 only

**Status:** accepted, 2026-08-21
**Amends:** Phase 2c and 2e of the implementation plan

## Context

Fitting scalers on the full dataset before sharding bakes every subject's values
into every shard's normalisation constants. A deleted subject's numbers then
survive in four other shards' feature scale after their rows are gone from their
own. The plan identifies this correctly and requires per-shard fitting.

The plan then specifies, for rebuild: purge the subject's rows, refit
preprocessing on the shard's remaining data, roll back to `checkpoint_{i-1}`,
retrain forward.

Those two steps are in conflict. `checkpoint_{i-1}` was trained under scaling
constants fit on the shard *including* the subject being deleted. Resuming from
it carries that subject's influence forward into the rebuilt model. The rebuild
is deterministic, the reproducibility spot-check passes, and the erasure is
incomplete — the failure is invisible to every check the system runs.

Refitting on the whole shard and retraining from slice 0 would be correct, but it
discards the slice-level saving entirely, which is most of why SISA is here.

## Decision

Fit the preprocessor on **slice 0's retained rows only**.

The constants are then a function of slice-0 subjects and nobody else:

- Deleting a subject in slice j > 0 leaves the preprocessor untouched, and
  `checkpoint_{j-1}` is genuinely free of them. Rollback is valid.
- Deleting a subject in slice 0 forces a refit and a full-shard retrain, which
  is the correct cost for that case.

Slices are ordered by churn ascending (ADR 0005), so slice 0 holds the subjects
least likely to be deleted. The expensive path is the rare one.

This is only sound because slices are subject-aligned. Under record-level
slicing a subject would have rows in slice 0 *and* later slices, so every
deletion would hit the refit path.

## What is not a per-shard statistic

The transaction-type enum is declared globally as a constant. It is schema — the
set of values the column may hold — not a quantity derived from anyone's data.
Deriving the vocabulary from data would both leak and give shards different
feature widths, breaking the fixed architecture Phase 5's batched ensemble needs.

Frequency encodings, target encodings and any other fitted statistic remain
strictly per-shard.

## Consequences

- `engine/train.py::fit_preprocessor` takes `slice_idx == 0` rows and nothing else.
- Constants are fit on a fifth of the shard, so they are noisier than a
  full-shard fit. Acceptable: they are location and scale parameters, not a
  model, and slice 0 is a churn-ordered stratum rather than a random one — worth
  revisiting if a shard's slice 0 turns out badly unrepresentative.
- A shard whose slice 0 lacks a transaction type still emits full-width
  features; the guard against a zero-variance column keeps the divide finite.
