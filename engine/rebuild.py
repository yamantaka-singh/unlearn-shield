"""Excise a subject from a shard and retrain forward from the rollback point.

    python -m engine.rebuild --subject C0000042

The rollback point is min_slice_idx - 1, and min_slice_idx comes from the
routing table rather than being recomputed from the shard file -- the shard file
no longer contains the subject once purged, so it cannot answer the question.
"""

import argparse
import json
import os

import numpy as np
import torch

from config.determinism import state_dict_digest
from config.settings import NUM_SLICES, SEED, SHARD_DIR, subject_ref
from engine.train import (checkpoint_path, load_routing, load_shard, save_shard,
                          train_shard)


def purge(records: dict, ref: str) -> tuple[dict, int]:
    """Physically drop the subject's rows. Returns (retained records, count dropped).

    Dropped, not flagged: a 'purged' column would leave the values on disk, and
    Phase 3's Merkle root is built over whatever this file contains. A retained
    set that still holds the record is a root that proves the wrong thing.
    """
    keep = records["subject_ref"] != ref
    dropped = int((~keep).sum())
    return {k: v[keep] for k, v in records.items()}, dropped


def rebuild(subject_id: str, seed: int = SEED) -> dict:
    ref = subject_ref(subject_id)
    routing = load_routing()
    if ref not in routing:
        raise KeyError(f"no routing entry for subject (ref {ref[:12]}...)")

    entry = routing[ref]
    shard, min_slice = entry["shard"], entry["min_slice_idx"]
    records, dropped = purge(load_shard(shard), ref)
    if dropped == 0:
        raise ValueError(f"routing says shard {shard} holds {ref[:12]}... but no rows matched")
    save_shard(shard, records)

    # min_slice 0 means there is no earlier checkpoint to resume from, and the
    # preprocessor was fit on slice 0, so it must be refit too -- a full retrain.
    # train_shard refits unconditionally, so this is just the resume argument.
    resume_state = None
    if min_slice > 0:
        resume_state = torch.load(checkpoint_path(shard, min_slice - 1), weights_only=True)

    digests = train_shard(shard, records=records, from_slice=min_slice,
                          resume_state=resume_state, seed=seed)

    # Drop the routing row too. Keeping it would retain a record that this
    # subject existed and which shard held them -- which is the kind of residue
    # an erasure is supposed to remove. Repeat requests are made idempotent by
    # Phase 4's idempotency_key, not by leaving this row behind.
    routing.pop(ref)
    with open(os.path.join(SHARD_DIR, "routing.json"), "w") as f:
        json.dump(routing, f, sort_keys=True, indent=1)

    return {
        "subject_ref": ref,
        "shard": shard,
        "resumed_from": f"slice{min_slice - 1}" if min_slice > 0 else "fresh_init",
        "rows_purged": dropped,
        "slices_retrained": list(range(min_slice, NUM_SLICES)),
        "result_weights": digests[NUM_SLICES - 1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    result = rebuild(args.subject)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
