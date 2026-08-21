# 0002 — Signed manifest with a non-inclusion proof, not hash equality

**Status:** accepted, 2026-08-21

## Context

An erasure needs evidence an auditor can check. The obvious construction is to
retrain the shard from scratch on the retained data and show the weights match
the rebuild's. It is also circular: verifying one deletion costs a second full
training run, so the verification is more expensive than the operation it
verifies, and it does not scale past a handful of requests.

## Decision

Emit a signed manifest carrying a Merkle non-inclusion proof over the shard's
retained subject set, and ship a verifier that runs with no access to the
training system.

Construction details that are load-bearing:

- **RFC 6962 hashing.** Leaves are `SHA256(0x00 || data)`, internal nodes
  `SHA256(0x01 || left || right)`. Without domain separation an internal node's
  hash is a valid leaf hash, letting a prover pass a subtree off as a leaf.
- **RFC 6962 splitting**, not odd-node duplication. Duplication lets two
  different leaf sets share a root — the CVE-2012-2459 shape.
- **The root commits to the leaf count**: `SHA256(0x02 || uint64(n) || MTH)`.
  See below.
- **Adjacency is checked.** A non-inclusion proof is two inclusion proofs for
  leaves that bracket the target *and sit at consecutive indices*. Without the
  index check a prover can hand over any two leaves straddling the target,
  leaving an unexamined gap that contains it.
- **Leaves are subject refs, not record ids.** The deletion unit is a subject,
  so subject-level leaves make the claim exactly one proof. Record-level leaves
  would require proving every one of a subject's records absent without knowing
  how many there were.

## On binding the leaf count

RFC 6962's audit-path check tracks tree size through its `fn`/`sn` bookkeeping,
and a brute-force search over all trees with n < 40 found no way to turn a
misreported `tree_size` into a false absence proof — the escalation attempt
(shrink the claimed size so the last real leaf falls outside it, then call that
leaf absent) was rejected in every case.

But that is evidence over a bounded range, not a property. Committing the leaf
count into the published root makes any size lie change the root outright, so
the guarantee holds by construction rather than by an argument that would need
re-deriving after every change to this file. One extra hash.

## What this proves, and what it does not

**Proves:** the subject is absent from the record set whose root the manifest
names, and the manifest was signed by the holder of the private key.

**Does not prove:** that `result_weights` was trained on that record set.
Nothing binds the two. An operator could purge the record, publish a clean root,
and ship the previous checkpoint; the signature would still verify, because a
signature over a false claim is a valid signature.

That gap is closed by `code_digest` plus re-running a sample of completed
rebuilds and comparing weight digests (Phase 4), with sample selection derived
from `HMAC(erasure_id, audit_key)` so the operator cannot steer it. The verifier
prints this limitation on every successful verification rather than leaving an
auditor to infer more than was shown.

## Known leak

A sorted-Merkle non-inclusion proof necessarily reveals the target's two
neighbours in sort order. Those neighbours are other subjects' HMAC refs. They
are pseudonymous, but an auditor collecting many certificates accumulates a
growing slice of the shard's population.

This is inherent to the construction, not an implementation choice. A sparse
Merkle tree over the full key space would prove the slot at `H(subject_ref)` is
empty without revealing neighbours, at the cost of a 256-level (compressible)
proof. Not built. The plan's open question about rate-limiting `erasure:attest`
per subject_ref is the cheaper partial mitigation and should be answered before
this serves a second tenant.

## Rejected

- **Hash equality against a clean-slate retrain.** Costs a full training run per
  deletion. If stronger evidence is wanted, the sampled reproducibility check is
  the right instrument, not a per-request shadow retrain.
- **Trusting an unsigned manifest.** `engine.rebuild` refuses to emit one.
