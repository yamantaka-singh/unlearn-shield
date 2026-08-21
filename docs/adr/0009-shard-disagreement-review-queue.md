# 0009 — Shard-disagreement review queue (optional, off by default)

**Status:** accepted as an optional feature, 2026-08-21
**Scope:** additive only. Nothing in the erasure guarantee, the manifest, the
proofs, or the signing path depends on it.

## Context

The ensemble serves the **mean** probability across shards. The proposal: also
compute the **spread**, and when shards disagree sharply, queue the transaction
for review rather than block it — on the theory that disagreement is an early
signal of a fraud pattern only some shards have been trained on.

The underlying idea is well-founded. Ensemble disagreement as an
epistemic-uncertainty proxy is the basis of query-by-committee active learning
and of deep-ensemble uncertainty estimation. The question was whether it holds
*here*, in an ensemble whose members are non-identically distributed by
construction.

## What was measured before building

Against this repo's frozen eval corpus (1,484 rows, 56 fraud):

| Signal | AUC as a fraud detector |
|---|---|
| Mean across shards (**what we serve**) | 0.5151 |
| **Spread across shards** | **0.5742** |

Spread carries *more* fraud signal than the score currently served. Both are
weak in absolute terms — this is a small synthetic model, not a production
fraud system — but the ordering is the surprising and relevant part, and it is
the direction the proposal predicted.

### The SISA-specific confound, quantified

Shards here are **not** i.i.d. samples: assignment is by churn score
(ADR 0005), and each shard has its own preprocessing constants (ADR 0004). So
some baseline disagreement is structural rather than anomalous. More
importantly, a rebuild retrains one shard and leaves the rest alone, shifting
its decision boundary relative to theirs — which would move the spread
distribution for reasons unrelated to fraud.

Measured, for a single-shard rebuild: mean spread **+2.0%**, flag rate at a
fixed p99 threshold **1.0% → 1.1%**. Real, but far smaller than expected.
Mitigated by recording `model_version` on every row, so a spread trend is only
ever compared within one ensemble version.

## Decision

Build it, **off by default**, physically separated from the core.

- `DISAGREEMENT_THRESHOLD=0.0` (the default) disables it entirely.
- `gateway/disagreement.py` is self-contained; deleting the file and its table
  leaves the rest of the system unchanged.
- `predict_proba`'s signature is untouched. It now delegates to a new
  `shard_probabilities`, and
  `test_predict_proba_is_exactly_the_mean_of_shard_probabilities` asserts the
  refactor changed no served score.
- The spread computation reuses the forward pass that already happened, so it
  costs one `np.std` over `n_shards` floats.
- The insert runs in a FastAPI `BackgroundTask` — after the response is sent,
  no new dependency.

### Latency, measured rather than asserted

| Configuration | p50 | p99 |
|---|---:|---:|
| Off (default) | 3.14 ms | 3.82 ms |
| On, realistic threshold (~1% flagged) | 3.05 ms | 3.85 ms |
| On, threshold so low 100% flag | 3.82 ms | 5.29 ms |

"Zero latency" holds for the caller in all cases — `BackgroundTasks` run after
the response is sent. It does **not** hold for server occupancy: at 100%
flagging the insert adds ~0.7 ms of post-response work per request. At a
sensible threshold that is ~1% of requests and disappears into noise. Worth
stating precisely, because "asynchronous" and "free" are not the same claim.

## The change made to the proposal: no transaction features

The proposal has the review agent inspect "what features triggered" each shard.
That would mean storing the transaction's features. This table deliberately
does not.

`gateway/schemas.py`'s `PredictRequest` carries **no `subject_id`**. Erasure
routes `subject_ref → subject_shard_map → shard rebuild`, so a row with no
subject linkage can never be reached by any erasure request. Storing features
would therefore accumulate personal data that this system — whose entire claim
is that it can prove erasure — has no mechanism to delete. That is a hole in
the product's own core guarantee, introduced by a fraud-detection side feature.

What is stored instead: per-shard scores, the mean, the spread, the threshold
in force, and `model_version`. That is enough for a reviewer to see *which*
shards fired and against which ensemble. The features themselves live in the
caller's own request logs, already governed by that caller's retention policy.

`test_table_stores_no_transaction_features` fails if anyone adds a feature
column. Adding one requires first adding `subject_ref` here and wiring this
table into `engine/rebuild.py`'s purge — at which point that test should be
updated deliberately, not deleted in passing.

## Two claims in the proposal that do not hold here

**"20 shards."** `NUM_SHARDS` is 5. The motivating example (18 calm, 2 alarmed)
becomes roughly 4-vs-1, and a variance estimated from five samples is far
noisier than one from twenty. The feature still works; its statistical power is
weaker than the proposal assumes, and the threshold must be calibrated against
observed traffic rather than reasoned about from the example.

**"Within 1 hour all 20 shards learn the new pattern."** This does not fit
SISA. A record belongs to exactly one shard — that is what makes erasure cheap.
Adding a newly-labelled transaction to every shard would mean a later erasure
of that subject requires rebuilding *every* shard, destroying the property the
whole system exists to provide. The architecturally consistent version: the
label lands in whichever shard already owns that subject, and only that shard
rebuilds. Cheaper than the proposal describes, but it is one shard learning,
not all of them.

## Consequences

- A `spread` trend is only interpretable within one `model_version`.
- `record()` swallows its own exceptions: an optional side-channel must never
  turn a successful prediction into a 500. Failures are therefore silent —
  acceptable while off by default, not acceptable if this becomes
  load-bearing, at which point it needs an error counter.
- The review→relabel→retrain loop is **not built**. This ADR covers producing
  the queue; consuming it is a separate decision that has to answer how a
  label change interacts with slice boundaries and rollback points.
