# 0003 — CPU-only training, and what "deterministic" actually means here

**Status:** accepted, 2026-08-21
**Supersedes:** the seed-only harness sketched in the first implementation plan

## Context

A signed manifest claims a shard was retrained from its retained data and
nothing else. Nobody can check that claim by inspecting weights. The only cheap
check is to re-run the rebuild and compare digests, which requires that a rebuild
be a pure function of (retained data, code, config, seed).

The first draft of the harness set `torch.manual_seed`, the cuDNN flags, and the
TF32 flags. That is not sufficient, and two of those flags are inert on a
CPU-only build.

## What we measured

`torch.use_deterministic_algorithms(True)` does not make CPU results invariant to
thread count, because float addition is not associative and OpenMP splits
reductions by thread count. Measured on torch 2.5.1, macOS arm64, summing 2^22
normal floats:

| `torch.set_num_threads` | `x.sum()` |
|---|---|
| 1 | `-0x1.633c040000000p+9` |
| 2 | `-0x1.633c100000000p+9` |
| 4 | `-0x1.633c100000000p+9` |
| 8 | `-0x1.633c100000000p+9` |

Linear-layer GEMM turned out to be thread-invariant at the shapes we tested —
oneDNN blocks those deterministically — so a small MLP forward/backward alone did
not surface the divergence. Reductions did. That matters because Phase 2 fits
per-shard scalers by taking means and standard deviations over a whole shard,
which is exactly a large reduction, and because loss reduction over a large batch
is another. A shard preprocessed on a 4-core runner and rebuilt on a 16-core one
would produce different normalisation constants and therefore different weights,
with nothing in the pipeline reporting a problem.

Python's hash randomisation is a second source: it changes `set` and `dict`
iteration order, and any place that derives record or feature ordering from a set
inherits that. It can only be fixed before the interpreter starts.

## Decision

1. CPU only. GPU needs its own ADR and its own determinism strategy, not a flag.
2. `enforce_determinism` pins thread count to 1 (`torch.set_num_threads`,
   `OMP_NUM_THREADS`, `MKL_NUM_THREADS`) in addition to seeding torch, numpy and
   `random`.
3. It raises if `PYTHONHASHSEED != 0` rather than pretending to set it. The
   Dockerfile and CI set it; local runs are told to.
4. Determinism is asserted **within one `code_digest`, not across code_digests**.
   A torch upgrade is allowed to change weights. It is not allowed to change them
   for a fixed image digest. Without this scoping, every dependency bump would
   look like a compliance incident.
5. Base image pinned by digest, every dependency pinned with `==`. This is part
   of the guarantee, not hygiene.

## Consequences

- Single-threaded training is slower. Acceptable at the assumed volume; if it
  stops being acceptable, pin an explicit thread count and fold it into
  `config_digest` rather than unpinning.
- The nightly spot-check compares against manifests carrying the same
  `code_digest`. Cross-digest comparisons are not evidence of drift.
- CI runs the suite twice: once on host Python, once on the pinned image. Only
  the second one is the claim.
