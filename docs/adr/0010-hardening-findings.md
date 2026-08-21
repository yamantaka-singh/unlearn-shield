# 0010 — Phase 7 hardening: what the review actually found

**Status:** accepted, 2026-08-21

Phase 7's brief was five items: security review, load test, digest-pin check,
wire up the reproducibility spot-check, and write the runbook. Two of them
produced findings worth recording; the rest confirmed what was expected.

## Finding 1 (security): `Idempotency-Key` collided across principals

`erasure_jobs.idempotency_key` was `UNIQUE` globally. Probing with two
principals sending the same key exposed two problems:

1. **Disclosure.** The second caller received the *first* caller's
   `erasure_id`, usable against `GET /v1/erasure/{id}` and `/certificate` by
   anyone also holding `erasure:attest`.
2. **Silent loss.** The second caller's request was never enqueued — the
   `ON CONFLICT DO NOTHING` swallowed it — while the API still returned `202`
   with a queued status.

The second is the serious one. A legally-required erasure that reports success
and never happens is precisely the failure mode this whole system exists to
make impossible, arriving through the intake path rather than the model.

**Fixed:** uniqueness is now `(requested_by, idempotency_key)`, and the
fallback lookup is scoped the same way — an unscoped `SELECT` would still have
handed back another principal's id even with the constraint corrected.
Migration 0004. Regression tests assert both that the leak is closed and that
same-principal replay still dedupes, since a "fix" that broke real idempotency
would be a worse trade.

This was reachable by any two callers of a shared gateway, which is the
deployment the plan describes ("service-to-service; end users hit a consent
manager, which calls this").

### Still open, deliberately

`erasure:attest` has no per-`subject_ref` rate limit. The plan raises this as
an open question, and it remains one: an auditor endpoint that answers
"has this subject been erased" is an oracle worth rate-limiting before this
serves more than one tenant. Not fixed here because the right limit depends on
real auditor traffic patterns, which do not exist yet.

Token comparison is a dict lookup, not constant-time. The timing signal on a
hash lookup is very weak, and the tokens are high-entropy service credentials
rather than guessable secrets, so this is noted rather than changed.

## Finding 2: the spot-check never needed a snapshot

Phase 4 deferred the reproducibility re-run with a note saying it required
pre-purge shard state that nothing captured. That was wrong, and the error cost
three phases of a known gap.

The manifest claims: *retraining from `resumed_from` on the retained data
yields `result_weights`*. The retained data is exactly what the shard file
holds immediately after the rebuild. So re-running inside the same worker pass
needs no snapshot at all — it is the one moment the inputs are still precisely
what the manifest describes. A later rebuild of that shard would move them,
which is presumably where the "needs a snapshot" intuition came from.

**Built.** `worker/jobs.py::run_spot_check`, on the existing HMAC-gated sample.

One hazard found while wiring it: `train_shard` writes checkpoints to fixed
paths, so a re-run would overwrite the promoted checkpoint. Harmless when the
check passes (identical bytes), destructive exactly when it fails — leaving the
next rebuild to resume from divergent weights matching no recorded hash. The
re-run now writes to a temporary directory, and a test asserts the real
checkpoint files are byte-identical before and after.

`test_spot_check_detects_a_genuine_divergence` is the negative control: a check
that only ever passes proves nothing.

### What it proves, and does not

Proves the rebuild is reproducible — nondeterminism has not crept in via thread
count, a library upgrade, or a DataLoader change. Does **not** prove an
operator ran the rebuild honestly; running the same code twice agrees with
itself either way. That gap is the one ADR 0002 records against `code_digest`,
and it is unchanged.

## Confirmed, no action

**Load ceiling.** `scripts/load_test.py` drains 100 jobs across 5 worker
passes: **20 jobs discharged per rebuild**, which is Phase 4c's batching doing
what it was designed to. Enqueue costs 0.3 ms/job against a rebuild measured in
hundreds of milliseconds *for this toy model*, so the Postgres queue is
nowhere near being the bottleneck. Quoting the raw rate would be meaningless —
it measures our model's size, not the system. What transfers is the ratio:

| Real rebuild duration | Erasures/hour | Erasures/day |
|---|---:|---:|
| 1 min | 1,200 | 28,800 |
| 5 min | 240 | 5,760 |
| 15 min | 80 | 1,920 |

Against Assumption 2's tens-to-low-hundreds per day, every row has substantial
headroom. Note the plan's own ~10k/day adaptive-sharding threshold is only
reachable with sub-minute rebuilds.

**Digest pinning.** `docker compose build --no-cache determinism` rebuilds from
the pinned digest and the determinism spot-check still passes. The pinned
digest still matches what `python:3.11-slim` currently resolves to, so the tag
has not moved since pinning — the pin has not yet had to do its job, but it is
correctly in place to.

**Runbook.** `docs/runbook.md` covers determinism drift, queue backlog past
SLA, and minority-class regression. The drift entry leads with checking whether
`code_digest` differs between the manifest and the check, because a version
skew looks identical to real drift and is far more likely.

## Consequences

- Migration 0004 alters a constraint on a live table. On a populated database,
  the `ADD CONSTRAINT` fails if duplicate `(requested_by, idempotency_key)`
  pairs already exist — which they cannot, since the old global unique was
  stricter. Safe in that direction only; the reverse migration is not.
- The spot-check doubles rebuild cost for the sampled fraction
  (`SPOT_CHECK_RATE`, default 1%). That was always the intended trade.
- A drift now writes `matched = false` and logs at ERROR. Nothing pages yet —
  wiring that to a real alerting channel is a deployment concern, not a code
  one, and the runbook assumes someone is watching the log.
