"""The four operations that differ between the MLP and GBDT engines.

`MODEL_ENGINE` selects which engine a deployment runs (config/settings.py).
This module is the only place that branches on it: the worker and the gateway
call these functions and never import `engine.rebuild` or `engine.gbdt`
directly, so adding a third engine would mean writing four functions here
rather than hunting for `if engine ==` scattered through the request path.

Deliberately four plain functions and no class hierarchy. There is no
"ModelEngine" interface with two implementations because nothing needs to hold
one as a value, pass it around, or have two live at once -- the choice is fixed
for the lifetime of the process by an environment variable. An abstract base
class here would be a factory for a product nobody constructs twice.

What is NOT here is as deliberate. Training, manifests, absence proofs,
signing, routing, the SMT, and the queue are all engine-independent already:
both engines write the same manifest shape over the same routing table and the
same shard files. The seam is genuinely four functions wide, and the reason it
is that narrow is that `engine/gbdt.py` was made to satisfy the contract
`engine/rebuild.py` already had, rather than a compatibility layer being built
to reconcile two different ones.
"""

import os
import shutil
import tempfile

from config.settings import CHECKPOINT_DIR, MODEL_ENGINE, NUM_SLICES

IS_GBDT = MODEL_ENGINE == "gbdt"


def rebuild_batch_by_ref(refs: list) -> dict:
    """Purge, roll back, retrain, and return signed manifests plus a replay
    payload. Both engines return the same keys; see engine/gbdt.py's copy of
    this function for what differs underneath."""
    if IS_GBDT:
        from engine.gbdt import rebuild_batch_by_ref as impl
    else:
        from engine.rebuild import rebuild_batch_by_ref as impl
    return impl(refs)


def live_model_path(shard: int) -> tuple[str, str]:
    """(path, extension) of the shard's live model at its conventional name.

    The MLP's is its final slice checkpoint; a booster has no per-slice files
    at all, because any earlier slice is a prefix of the one model (ADR 0011),
    so the whole booster is the artifact. Both are the mutable path the next
    rebuild of this shard overwrites -- never hand this to the DB, hand it to
    `promote_artifact` first.
    """
    if IS_GBDT:
        from engine.gbdt import booster_path
        return booster_path(shard), ".json"
    from engine.train import checkpoint_path
    return checkpoint_path(shard, NUM_SLICES - 1), ".pt"


def promote_artifact(shard: int, result_weights: str) -> str:
    """Copy the finished model to a content-addressed path and return it.

    Content-addressing is the point (ADR 0006), and it matters for both
    engines for the same reason: each writes its live model to a fixed,
    conventional path -- `shard{k}_slice{i}.pt` for the MLP, `shard{k}_gbdt.json`
    for trees -- which the NEXT rebuild of that shard overwrites. A checkpoints
    row recorded against the conventional path would end up describing bytes
    that are no longer there.
    """
    source, suffix = live_model_path(shard)

    cas_dir = os.path.join(CHECKPOINT_DIR, "cas")
    os.makedirs(cas_dir, exist_ok=True)
    cas_path = os.path.join(cas_dir, f"{result_weights}{suffix}")
    if not os.path.exists(cas_path):
        shutil.copyfile(source, cas_path)
    return cas_path


def load_ensemble(shard_paths: dict):
    """An object with `.shard_probabilities(records, rows)` and
    `.predict_proba(records, rows)`, whichever engine is active.

    The MLP needs its per-shard preprocessors alongside the weights (ADR 0004);
    those live next to the checkpoints under their conventional names and are
    derived here rather than by each caller, which is how the gateway and the
    worker previously came to hold two copies of the same path expression.
    Trees fit nothing per shard, so there is nothing to pair.
    """
    if IS_GBDT:
        from inference.batched_ensemble import load_gbdt_ensemble
        return load_gbdt_ensemble(shard_paths)

    from inference.batched_ensemble import load_ensemble as impl
    preproc = {s: os.path.join(CHECKPOINT_DIR, f"shard{s}_preproc.json") for s in shard_paths}
    return impl(shard_paths, preproc)


def replay_digest(shard: int, replay: dict) -> str:
    """Re-run a completed rebuild from its replay payload; return the weight
    digest to compare against the manifest's `result_weights`.

    The MLP re-runs into a temporary directory and throws it away. That is not
    tidiness: writing to the real checkpoint paths would be harmless when the
    check passes and destructive exactly when it fails, leaving the next
    rebuild to resume from divergent weights matching no recorded hash. Trees
    have no such hazard -- a booster is one in-memory object and nothing is
    written -- so the GBDT path needs no scratch directory.
    """
    if IS_GBDT:
        from engine.gbdt import replay_digest as impl
        return impl(shard, replay)

    from engine.train import train_shard
    with tempfile.TemporaryDirectory(prefix="unlearnshield-spotcheck-") as scratch:
        digests = train_shard(shard, records=replay["records"],
                              from_slice=replay["from_slice"],
                              resume_state=replay["resume_state"],
                              seed=replay["seed"], checkpoint_dir=scratch)
    return digests[max(digests)]
