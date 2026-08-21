"""Excise a subject from a shard and retrain forward from the rollback point.

    python -m engine.rebuild --subject C0000042

The rollback point is min_slice_idx - 1, and min_slice_idx comes from the
routing table rather than being recomputed from the shard file -- the shard file
no longer contains the subject once purged, so it cannot answer the question.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from hashlib import sha256

import numpy as np
import torch

from config.settings import (BATCH_SIZE, CODE_DIGEST, EPOCHS_PER_SLICE, LEARNING_RATE,
                             NUM_SHARDS, NUM_SLICES, SEED, SHARD_DIR, subject_ref)
from engine.train import (checkpoint_path, load_routing, load_shard, save_shard,
                          train_shard)
from verify import manifest as manifest_mod
from verify.merkle import build_root, prove_absence
from verify.sign import sign_manifest


def config_digest() -> str:
    """Binds the training hyperparameters into the manifest. A rebuild run with
    different slice counts or epochs is a different computation, and a manifest
    that did not say so would let one be substituted for another."""
    config = json.dumps({"num_shards": NUM_SHARDS, "num_slices": NUM_SLICES,
                         "epochs_per_slice": EPOCHS_PER_SLICE,
                         "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                         "seed": SEED}, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(config.encode()).hexdigest()


def purge(records: dict, ref: str) -> tuple[dict, int]:
    """Physically drop the subject's rows. Returns (retained records, count dropped).

    Dropped, not flagged: a 'purged' column would leave the values on disk, and
    Phase 3's Merkle root is built over whatever this file contains. A retained
    set that still holds the record is a root that proves the wrong thing.
    """
    keep = records["subject_ref"] != ref
    dropped = int((~keep).sum())
    return {k: v[keep] for k, v in records.items()}, dropped


def rebuild_batch_by_ref(refs: list, seed: int = SEED, sign: bool = True) -> dict:
    """Same as `rebuild_batch`, but takes subject_refs (already HMAC'd) directly.

    This is the entrypoint the worker uses. `erasure_jobs.subject_ref` is the
    only form of the identifier that ever reaches the worker -- raw subject IDs
    stop existing past ingest by design -- so the worker cannot call
    `rebuild_batch`, which hashes a raw ID as its first step. Rather than have
    the worker re-derive or fake a raw ID, the hashing step is factored out so
    both entrypoints share everything after it.
    """
    if not refs:
        raise ValueError("rebuild_batch_by_ref called with no subjects")
    purged_at = datetime.now(timezone.utc).isoformat()
    routing = load_routing()

    for ref in refs:
        if ref not in routing:
            raise KeyError(f"no routing entry for subject (ref {ref[:12]}...)")

    entries = [routing[r] for r in refs]
    shards = {e["shard"] for e in entries}
    if len(shards) != 1:
        raise ValueError(f"batch spans shards {shards}; must be pre-grouped by shard")
    shard = shards.pop()
    min_slice = min(e["min_slice_idx"] for e in entries)

    records = load_shard(shard)
    dropped_total = 0
    for ref in refs:
        records, dropped = purge(records, ref)
        if dropped == 0:
            raise ValueError(f"routing says shard {shard} holds {ref[:12]}... but no rows matched")
        dropped_total += dropped
    save_shard(shard, records)

    # min_slice 0 means there is no earlier checkpoint to resume from, and the
    # preprocessor was fit on slice 0, so it must be refit too -- a full retrain.
    # train_shard refits unconditionally, so this is just the resume argument.
    resume_state = None
    if min_slice > 0:
        resume_state = torch.load(checkpoint_path(shard, min_slice - 1), weights_only=True)

    digests = train_shard(shard, records=records, from_slice=min_slice,
                          resume_state=resume_state, seed=seed)
    result_weights = digests[NUM_SLICES - 1]
    resumed_from = f"slice{min_slice - 1}" if min_slice > 0 else "fresh_init"
    completed_at = datetime.now(timezone.utc).isoformat()

    # Built from the shard file AFTER the purge, so the root commits to the
    # retained set and every proof is about the set that actually trained the
    # model -- not the set as it was when any of these requests arrived.
    retained = set(records["subject_ref"].tolist())
    dataset_root = build_root(retained)
    cfg_digest = config_digest()

    manifests = {}
    for ref in refs:
        m = manifest_mod.build(
            subject_ref=ref, shard=shard, resumed_from=resumed_from,
            dataset_root=dataset_root, absence_proof=prove_absence(ref, retained),
            code_digest=CODE_DIGEST, config_digest=cfg_digest,
            result_weights=result_weights,
            model_version=f"shard{shard}-{result_weights[:12]}",
            purged_at=purged_at, completed_at=completed_at,
        )
        if sign:
            m["signature"] = sign_manifest(m)
        manifests[ref] = m

    # Drop the routing rows too. Keeping them would retain a record that these
    # subjects existed and which shard held them -- the kind of residue an
    # erasure is supposed to remove. Repeat requests are made idempotent by
    # Phase 4's idempotency_key, not by leaving these rows behind.
    for ref in refs:
        routing.pop(ref)
    with open(os.path.join(SHARD_DIR, "routing.json"), "w") as f:
        json.dump(routing, f, sort_keys=True, indent=1)

    return {
        "shard": shard,
        "resumed_from": resumed_from,
        "rows_purged": dropped_total,
        "slices_retrained": list(range(min_slice, NUM_SLICES)),
        "result_weights": result_weights,
        "manifests": manifests,  # subject_ref -> signed manifest
    }


def rebuild_batch(subject_ids: list, seed: int = SEED, sign: bool = True) -> dict:
    """Raw-subject-ID entrypoint: hash each id, then delegate. Used by the CLI
    and anywhere else that still holds a raw ID (i.e. at ingest, before it's
    reduced to a ref)."""
    return rebuild_batch_by_ref([subject_ref(s) for s in subject_ids], seed=seed, sign=sign)


def rebuild(subject_id: str, seed: int = SEED, sign: bool = True) -> dict:
    ref = subject_ref(subject_id)
    result = rebuild_batch([subject_id], seed=seed, sign=sign)
    return {
        "subject_ref": ref,
        "shard": result["shard"],
        "resumed_from": result["resumed_from"],
        "rows_purged": result["rows_purged"],
        "slices_retrained": result["slices_retrained"],
        "result_weights": result["result_weights"],
        "manifest": result["manifests"][ref],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--manifest-out", help="write the signed manifest here")
    args = parser.parse_args()
    result = rebuild(args.subject)
    if args.manifest_out:
        with open(args.manifest_out, "w") as f:
            json.dump(result["manifest"], f, indent=1, sort_keys=True)
    summary = {k: v for k, v in result.items() if k != "manifest"}
    summary["dataset_root"] = result["manifest"]["dataset_root"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
