# 0007 — Sparse Merkle absence proofs, and where serving latency actually goes

**Status:** accepted, 2026-08-21
**Closes:** the neighbour leak recorded as open in [ADR 0002](0002-manifest-over-hash-equality.md)
**Supersedes:** Phase 5's assumption that batched inference is the serving win

## Part 1: absence proofs that name nobody

ADR 0002 recorded that a sorted-Merkle non-inclusion proof works by naming the
target's two sort-order neighbours, so every certificate discloses two real
subject refs in cleartext. It named a sparse Merkle tree as the fix and did not
build one.

`verify/smt.py` now does. The `subject_ref` *is* the path: bit *d* selects left
or right at depth *d*, so each key has one fixed position among 2²⁵⁶ and
absence means "the leaf at my own position is empty". The proof carries sibling
subtree hashes along that path. A sibling covering populated ground is a hash
of that region — it commits to those subjects without revealing which they are.

Measured on a 400-subject shard: proofs carry 6–9 siblings rather than 256
(default all-empty siblings are recomputable by the verifier, so only
informative ones travel), verification takes 0.23 ms, and an end-to-end
certificate contains **zero** of the 400 retained subject refs.
`test_proof_names_no_other_subject` asserts both halves: the old scheme leaks
exactly two, the new one leaks none.

### What still leaks

The count of non-default siblings, which approximates the depth at which the
target's path leaves the populated region, and so hints at tree density near
that path. Much weaker than two exact identifiers, and unlike the sorted-tree
leak it does not accumulate into a population census as certificates pile up.

### Why not a ZK-SNARK

A SNARK would hide the sibling hashes too, at the cost of a circom or arkworks
toolchain, a trusted setup to run and then be trusted about, and seconds of
proving time per erasure. Against the threat model that motivated this work —
an auditor who already holds the certificate and wants to learn about *other*
subjects — it buys nothing the sparse tree does not already provide, because
the sparse tree's siblings already name nobody. Revisit only if a threat model
appears where the sibling hashes themselves are the problem.

### Compatibility

Proofs declare a `scheme`. The verifier dispatches on it and still checks
sorted-Merkle proofs that declare none. A certificate outlives the code that
issued it; a verifier that cannot check last year's certificate is broken.
`verify/merkle.py` and its tests stay for exactly that reason.

## Part 2: the serving bottleneck was never the ensemble

The plan's Phase 5, and every subsequent proposal to speed up serving, assumed
the cost was ensembling across shards — one forward pass per shard — and
prescribed batching, ONNX export, or a Rust inference gateway.

Measured, per `/v1/predict` call, before any change:

| | Cost | Share |
|---|---|---|
| Fresh `psycopg2.connect` | 6.22 ms | ~70% |
| Load 5 checkpoints from disk | 2.63 ms | ~30% |
| Warm DB query | 0.23 ms | — |
| **Ensemble forward passes** | **0.15 ms** | **~1.7%** |

Observed end-to-end: p50 13.26 ms, p99 45.90 ms, in-process with no network.

Exporting to ONNX to accelerate the forward pass optimises 1.7% of the request
and leaves 8.85 ms of connection setup and file I/O untouched. The prescription
was aimed at the wrong term, and it looked plausible because a batched-ensemble
module is the obvious-looking performance work.

**Decision:** pool connections (`db.conn.pooled`, used by request handlers;
the worker keeps a dedicated connection since it holds one for the length of a
rebuild) and cache loaded ensembles keyed on the promoted version's checkpoint
hashes. Batching stays — it is genuinely faster and Phase 5 asked for it — but
it is the smallest of the three wins, and the module says so.

Result: **p50 3.45 ms, p95 6.41 ms, p99 9.80 ms** — a 3.8× p50 and 4.7× p99
improvement, from pooling and caching rather than from the inference stack.

### The cache key is a correctness decision, not a performance one

The key is the tuple of checkpoint hashes for the promoted `model_version`.
A rebuild that promotes new weights produces a different key, so the stale
entry stops being reachable with no invalidation call to forget.

A cache invalidated by hand would eventually miss one, and the failure mode is
the precise thing this project exists to prevent: the erasure lands, every job
row says `done`, the certificate verifies — and the serving layer keeps
scoring from a model that still contains the erased subject, silently.

### On the plan's `S×` claim

Phase 5 predicted batching would win by roughly `S×`. It does not, and cannot:
per-shard preprocessing (ADR 0004) means each shard needs its own scaled copy
of the input, so `vmap` maps over parameters *and* inputs together and the
preprocessing cost stays `S×`. The module reports measured numbers instead of
asserting the ratio.

## Consequences

- The pool holds up to 16 connections. Tests reset it between cases
  (`db.conn.reset_pool`) because a pooled connection outlives a `TRUNCATE`.
- The ensemble cache is unbounded, keyed by promotion rather than by traffic,
  so it grows with rebuild count. An LRU is the upgrade if a long-lived process
  promotes often enough to matter.
- Sub-10 ms p99 holds for single-row scoring on this hardware with a local
  database. It is not a claim about production hardware, network hops, or
  batch scoring, and no such claim should be made from these numbers.
