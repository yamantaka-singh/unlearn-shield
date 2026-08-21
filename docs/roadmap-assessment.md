# Roadmap assessment

An "enterprise-grade" expansion proposal was evaluated on 2026-08-21. Three
items were built, and the rest were declined with reasons. Recorded here so
the same proposals do not get re-litigated from scratch, and so the declines
can be argued with.

## Built

### Sparse Merkle absence proofs

The proposal identified the neighbour leak that [ADR 0002](adr/0002-manifest-over-hash-equality.md)
had recorded as open, and was right that it should be closed. It offered a
sparse Merkle tree *or* a ZK-SNARK; the sparse tree is the correct half of
that choice (see the SNARK entry below). Built in `verify/smt.py`, closing the
gap: a certificate now names zero other subjects, down from exactly two.
→ [ADR 0007](adr/0007-sparse-merkle-and-serving-latency.md)

### Serving latency

The proposal was right that latency mattered and wrong about where it was.
It prescribed ONNX export and a Rust inference gateway to accelerate the shard
ensemble. Measurement showed the ensemble was 0.15 ms of a 13.3 ms request —
1.7%. The real costs were a fresh database connection per call (6.22 ms) and
loading five checkpoints from disk per call (2.63 ms).

Pooling and caching took p50 from 13.3 ms to 3.5 ms and p99 from 45.9 ms to
9.8 ms, beating the proposal's own sub-10 ms target without touching the
inference stack. Batching was implemented too, and it is the smallest of the
three wins.

### GBDT / XGBoost SISA — BUILT

*Assessed as the strongest deferred item; built 2026-08-21, see
[ADR 0011](adr/0011-gbdt-sisa.md). Original assessment kept below because the
predictions in it were checked rather than assumed, and one was wrong.*

**The strongest idea in the proposal, and the one most worth building next.**
The premise is correct: most production tabular fraud models are gradient
boosted trees, not neural networks. A SISA implementation that only supports
PyTorch MLPs addresses a minority of the market it is aimed at.

It is also a better technical fit than the proposal realised. SISA's
slice-rollback maps onto boosting *more* cleanly than onto gradient descent:

- Slice *i* adds *K* trees via `xgb.train(..., xgb_model=previous)`.
- Rolling back to slice *i-1* is **truncating the tree list** to the first
  `(i-1) * K` trees. No checkpoint reload, no retraining to reach the rollback
  point — the rollback is exact and nearly free.
- GBDTs need no feature scaling, so the per-shard preprocessing rule
  ([ADR 0004](adr/0004-per-shard-preprocessing.md)) and its slice-0 subtlety
  mostly evaporate.
- Determinism fits the existing doctrine: `nthread=1` plus a fixed seed, for
  the same non-associative-float reason documented in
  [ADR 0003](adr/0003-cpu-only-determinism.md).

Deferred at assessment time because it is a second engine — its own training
path, determinism verification, checkpoint format, and serving path — not
because it was doubtful.

**Outcome.** Every prediction above was checked rather than assumed, and all
held except one. "Rollback is exact and nearly free" and "determinism fits the
existing doctrine" are now measured facts: truncation is bit-identical to
training fewer rounds, and a rebuild is byte-identical to a clean retrain on
retained data. The preprocessing prediction held too — no fitted per-shard
statistic remains.

The exception was an intermediate claim, made while building, that XGBoost's
fitted `base_score` carried erased subjects into rebuilt models. Measured in
the real code path it does not: the estimate is drawn from slice 0 only, and a
slice-0 erasure restarts training anyway. It is pinned regardless, because that
safety is incidental rather than designed. ADR 0011 records the correction.

The offline engine is done. Serving is not, and needs its own decision —
`inference/batched_ensemble.py` is PyTorch-specific with no tree analogue.

## Declined

### ZK-SNARK absence proofs

The sparse Merkle tree already reveals no subject identifiers. A SNARK would
additionally hide the sibling *hashes*, at the cost of a circom or arkworks
toolchain, a trusted setup to run and then be trusted about, and seconds of
proving time per erasure.

Against the threat model that motivated this work — an auditor who holds the
certificate and wants to learn about other subjects — it buys nothing the
sparse tree does not already provide. Revisit if a threat model appears where
the sibling hashes themselves are the problem.

The claim that this "satisfies the strictest DoD and Top Secret / SCI
requirements" is not one this project can make. Those are accreditation
regimes about process, personnel, and environment, not properties a hash
construction confers.

### Graph and vector embedding unlearning

Unlearning a node from a GNN is an open research problem, not a sprint item —
message passing means a node's influence propagates to its k-hop neighbourhood,
so "remove this node" has no exact analogue to shard rollback. Vector database
scrubbing is a different product with a different data model.

Presenting either as a week of work would be misleading about the state of the
research.

### Dynamic / adaptive re-sharding

Assumption 2 of the plan puts the static-sharding ceiling around 10k
deletions/day. More importantly, re-sharding invalidates every checkpoint's
rollback point and every in-flight manifest's `resumed_from`. The correctness
question — what happens to certificates issued against a shard layout that no
longer exists — has no cheap answer, and it is a compliance question, not a
performance one.

Worth building when volume demands it, after that question is settled. Not
before.

### Kafka, Snowflake, and Databricks CDC connectors

The integration point already exists: `POST /v1/erasure` is idempotent
(`Idempotency-Key` enforced at the schema level), returns 202, and is safe to
call from a CDC consumer, a Kafka worker, or a warehouse trigger. A connector
is roughly thirty lines of glue calling that endpoint.

Writing a Snowflake connector that has never run against Snowflake, or a Kafka
consumer that has never seen a broker, produces untested code that *looks*
like an integration. The honest position is that the API is the integration
point, and connectors get built against real infrastructure by whoever has it.

### ONNX export and a Rust inference gateway

Superseded by measurement — see above. The forward pass is 1.7% of the request.
Revisit if the profile changes materially, for instance under large batch
scoring where the forward pass would actually dominate.

### HSM / KMS key custody

Partly already satisfied and worth being precise about. `verify/sign.py` reads
the signing key from an environment variable, which is exactly the shape a
Vault agent, a Kubernetes secret, or an injected KMS secret provides — so
"keys are not in the repo" already holds, and `.signing_key` is gitignored and
written `0600`.

What is genuinely missing is *remote* signing, where the private key never
leaves the HSM and the service sends a digest to be signed. That is a different
function shape, not a config change, and it cannot be built untested against a
real KMS. Worth doing at the point there is an account to test against.

### "Court-ready" compliance PDF asserting Article 17 adherence

A certificate export — including a QR code linking to verification — is
reasonable and cheap, and may get built.

What will not be generated is a document asserting **legal** conclusions.
Whether a given erasure satisfies Article 17 depends on retention obligations,
legitimate-interest bases, and the wider processing landscape, none of which
this system can see. Producing a formal-looking artefact that asserts
compliance is worse than producing nothing, because it invites reliance the
underlying evidence does not support.

The defensible artefact presents cryptographic facts — root, signature,
timestamps, what is proven, and explicitly what is not — and leaves the legal
conclusion to counsel. That is also the more credible document in front of a
regulator.
