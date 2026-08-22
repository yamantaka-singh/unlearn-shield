"""SISA over gradient-boosted trees. Rollback is truncation.

Most production tabular fraud models are GBDTs, not neural networks, so a SISA
implementation that only handles MLPs addresses a minority of its target. This
is the tree half.

It is also a *better* fit than the MLP path, for a reason worth stating plainly:

    Rolling back to slice i means keeping the trees slices 0..i-1 produced and
    discarding the rest. `booster[0:n]` does exactly that, and it is exact --
    verified bit-identical against training only n rounds in the first place
    (test_truncation_equals_training_fewer_rounds).

Consequences that fall out of that:

  * **No per-slice checkpoint files.** The MLP engine writes
    `shard{k}_slice{i}.pt` for every slice because gradient descent cannot be
    un-done. A booster contains its own history: one file per shard, and any
    earlier checkpoint is a slice of it. Five files become one.
  * **Reaching the rollback point is free.** No checkpoint to load, no partial
    retrain to get back there -- just drop trees.
  * **Nothing is fitted per shard except one number.** GBDTs are invariant to
    monotone feature scaling, so the scaler machinery of ADR 0004 has nothing
    to do. The features here are raw numerics plus a fixed-vocabulary one-hot,
    neither of which is fitted to anyone's data.

That last point has one sharp exception, which is the whole reason
`base_score` is pinned below.
"""

import json
import os
from datetime import datetime, timezone
from hashlib import sha256

import numpy as np
import xgboost as xgb

from config.determinism import enforce_determinism
from config.settings import CODE_DIGEST, NUM_SHARDS, NUM_SLICES, SEED, SHARD_DIR, subject_ref
from data.synth import NUMERIC_COLUMNS, TYPES
from engine.train import load_routing, load_shard, save_shard
from verify import manifest as manifest_mod
from verify.sign import sign_manifest
from verify.smt import build_root, prove_absence

TREES_PER_SLICE = int(os.environ.get("GBDT_TREES_PER_SLICE", "20"))

PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.1,
    "tree_method": "hist",
    # Single-threaded for the same reason config/determinism.py pins torch:
    # thread count changes the order of float reductions, and a manifest is
    # only auditable if a rebuild is reproducible (ADR 0003).
    "nthread": 1,
    "seed": SEED,

    # Pinned defensively. Left to its default, base_score is a FITTED global
    # statistic, and a rebuild continuing from a truncated booster inherits it
    # from the loaded model rather than recomputing -- the shape of ADR 0004's
    # preprocessing leak, where a purged subject survives in a constant.
    #
    # Measured honestly: that leak does NOT currently manifest here, and the
    # reason is worth writing down because it is incidental rather than
    # designed. XGBoost estimates base_score on the first train() call, which
    # in this engine sees slice 0 only. A subject in slice >= 1 is therefore
    # not in the data the estimate is drawn from; a subject in slice 0 makes
    # rollback() return None, so training restarts and the estimate is redrawn
    # from retained data. Both cases are clean, by accident of when XGBoost
    # happens to estimate.
    #
    # That is a thin thing to rest an erasure guarantee on -- it breaks if
    # training ever starts mid-slice, or if XGBoost moves when it estimates.
    # A constant is fitted to nobody, costs nothing, and makes the property
    # structural instead of contingent.
    "base_score": 0.5,
}


def features(records: dict, rows: np.ndarray) -> np.ndarray:
    """Raw numerics plus a fixed-vocabulary one-hot. Nothing fitted.

    No scaling: trees split on thresholds, so any monotone rescaling produces
    the same tree. The one-hot vocabulary is schema (the set of values the
    column may hold), not a statistic derived from data -- the same
    distinction ADR 0004 draws for the MLP path.
    """
    numeric = np.column_stack([records[c][rows] for c in NUMERIC_COLUMNS])
    onehot = np.zeros((len(rows), len(TYPES)))
    onehot[np.arange(len(rows)), records["type_idx"][rows]] = 1.0
    return np.column_stack([numeric, onehot]).astype(np.float32)


def _matrix(records: dict, upto_slice: int) -> xgb.DMatrix:
    """Cumulative: at slice i, SISA trains on slices 0..i."""
    rows = np.flatnonzero(records["slice_idx"] <= upto_slice)
    return xgb.DMatrix(features(records, rows), label=records["isFraud"][rows])


def booster_path(shard: int, directory: str | None = None) -> str:
    return os.path.join(directory or SHARD_DIR, f"shard{shard}_gbdt.json")


def train_shard(shard: int, records: dict, from_slice: int = 0,
                booster: xgb.Booster | None = None,
                trees_per_slice: int = TREES_PER_SLICE) -> xgb.Booster:
    """Boost `trees_per_slice` rounds per slice over [from_slice, NUM_SLICES).

    `booster` is the truncated model to continue from; None starts fresh.
    """
    if from_slice > 0 and booster is None:
        raise ValueError(f"resuming at slice {from_slice} needs a booster to continue from")
    enforce_determinism(SEED)
    for slice_idx in range(from_slice, NUM_SLICES):
        booster = xgb.train(PARAMS, _matrix(records, slice_idx),
                            num_boost_round=trees_per_slice, xgb_model=booster)
    return booster


def rollback(booster: xgb.Booster, to_slice: int,
             trees_per_slice: int = TREES_PER_SLICE) -> xgb.Booster | None:
    """Discard every tree slices >= `to_slice` produced.

    Returns None for slice 0 -- there is nothing before it, so the rebuild
    starts fresh. Exact, and the reason this engine needs no checkpoint files.
    """
    if to_slice == 0:
        return None
    return booster[0:to_slice * trees_per_slice]


def rebuild_booster(shard: int, refs: list, records: dict, min_slice: int,
                    booster: xgb.Booster,
                    trees_per_slice: int = TREES_PER_SLICE) -> tuple[xgb.Booster, dict]:
    """Purge `refs`, roll back to `min_slice`, boost forward on what remains.

    Pure in-memory operation -- no file I/O, no manifest. `rebuild_batch_by_ref`
    below is the entrypoint that adds those and is what a caller normally wants;
    this is the piece it's built from, and what the unit tests exercise
    directly since they construct their own in-memory shards.

    Returns (rebuilt booster, retained records). The purge happens here rather
    than in the caller so it cannot be forgotten, and the retained rows come
    back because that is what the absence proof must be built over.
    """
    keep = ~np.isin(records["subject_ref"], refs)
    if keep.all():
        raise ValueError(f"none of {len(refs)} subject(s) found in shard {shard}")
    retained = {k: v[keep] for k, v in records.items()}
    return train_shard(shard, retained, from_slice=min_slice,
                       booster=rollback(booster, min_slice, trees_per_slice),
                       trees_per_slice=trees_per_slice), retained


def predict(booster: xgb.Booster, records: dict, rows: np.ndarray) -> np.ndarray:
    return booster.predict(xgb.DMatrix(features(records, rows)))


def save_booster(booster: xgb.Booster, shard: int) -> None:
    os.makedirs(SHARD_DIR, exist_ok=True)
    booster.save_model(booster_path(shard))


def load_booster(shard: int) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(booster_path(shard))
    return booster


def config_digest() -> str:
    """Same purpose as engine/rebuild.py's function of the same name: binds
    the hyperparameters that shape the boosted trees into the manifest, so a
    rebuild run under different settings cannot be substituted for another.
    """
    config = json.dumps({"num_shards": NUM_SHARDS, "num_slices": NUM_SLICES,
                         "trees_per_slice": TREES_PER_SLICE, "seed": SEED,
                         **{k: v for k, v in PARAMS.items() if k != "seed"}},
                        sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(config.encode()).hexdigest()


def build(routing: dict, trees_per_slice: int = TREES_PER_SLICE) -> None:
    """Train and save an initial booster for every shard `routing` covers.

    Reuses engine.train.build()'s routing and shard files -- the corpus is not
    engine-specific, only which model trains on it. Run after
    `engine.train.build()`, against the same data/shards/ directory:

        routing = engine.train.build(n_subjects=...)
        engine.gbdt.build(routing)
    """
    shards = {e["shard"] for e in routing.values()}
    for shard in sorted(shards):
        records = load_shard(shard)
        save_booster(train_shard(shard, records, trees_per_slice=trees_per_slice), shard)


def rebuild_batch_by_ref(refs: list, trees_per_slice: int = TREES_PER_SLICE,
                         sign: bool = True) -> dict:
    """The GBDT counterpart to engine/rebuild.py::rebuild_batch_by_ref. Same
    routing table, same shard file format, same manifest and proof machinery
    -- only the model differs.

    Deliberately independent rather than merged into the MLP path. The two
    engines are not designed to run against the same routing.json and shard
    directory at once: this function pops routing rows exactly like the MLP
    path does, so running both against one data/shards/ directory would have
    each engine's rebuild erase the other's routing entry too. ADR 0011 treats
    these as alternative, non-coexisting deployments, not a hybrid; making that
    coexist is a real design decision this function does not make.
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
    booster = load_booster(shard)
    rebuilt, retained = rebuild_booster(shard, refs, records, min_slice, booster,
                                       trees_per_slice=trees_per_slice)
    save_shard(shard, retained)
    save_booster(rebuilt, shard)
    # Recomputed rather than returned out of rebuild_booster: `rollback` is
    # pure (`booster[0:n]` builds a new Booster, it does not mutate), so this
    # is the identical object rebuild_booster trained from, and taking it here
    # keeps rebuild_booster's signature -- which the unit tests drive directly
    # against in-memory shards -- unchanged.
    resume_booster = rollback(booster, min_slice, trees_per_slice)

    result_weights = sha256(rebuilt.save_raw(raw_format="json")).hexdigest()
    resumed_from = f"slice{min_slice - 1}" if min_slice > 0 else "fresh_init"
    completed_at = datetime.now(timezone.utc).isoformat()

    # Built from the retained set AFTER the purge, same reasoning as the MLP
    # path: the proof has to be about the data that actually trained the
    # model, not the set as it stood when the request arrived.
    retained_refs = set(retained["subject_ref"].tolist())
    dataset_root = build_root(retained_refs)
    cfg_digest = config_digest()

    manifests = {}
    for ref in refs:
        m = manifest_mod.build(
            subject_ref=ref, shard=shard, resumed_from=resumed_from,
            dataset_root=dataset_root, absence_proof=prove_absence(ref, retained_refs),
            code_digest=CODE_DIGEST, config_digest=cfg_digest,
            result_weights=result_weights,
            model_version=f"gbdt-shard{shard}-{result_weights[:12]}",
            purged_at=purged_at, completed_at=completed_at,
        )
        if sign:
            m["signature"] = sign_manifest(m)
        manifests[ref] = m

    for ref in refs:
        routing.pop(ref)
    with open(os.path.join(SHARD_DIR, "routing.json"), "w") as f:
        json.dump(routing, f, sort_keys=True, indent=1)

    return {
        "shard": shard,
        "resumed_from": resumed_from,
        "rows_purged": len(records["subject_ref"]) - len(retained["subject_ref"]),
        "slices_retrained": list(range(min_slice, NUM_SLICES)),
        "result_weights": result_weights,
        "manifests": manifests,
        # Same shape and same purpose as engine/rebuild.py's `replay`: enough
        # to re-run this exact rebuild for the reproducibility spot-check,
        # captured at the only moment the inputs still match what the manifest
        # describes. `resume_booster` stands where the MLP path puts
        # `resume_state` -- for trees the resume point is a truncated model
        # rather than a loaded checkpoint, which is the whole of ADR 0011.
        "replay": {"records": retained, "from_slice": min_slice,
                   "resume_booster": resume_booster,
                   "trees_per_slice": trees_per_slice, "seed": SEED},
    }


def replay_digest(shard: int, replay: dict) -> str:
    """Re-run a rebuild from its `replay` payload and return the weight digest.

    The GBDT counterpart to re-running engine.train.train_shard in a scratch
    directory. Nothing is written: a booster is one in-memory object, so the
    MLP path's careful "never overwrite the promoted checkpoint" temporary
    directory has no equivalent hazard to guard against here.
    """
    rebuilt = train_shard(shard, replay["records"], from_slice=replay["from_slice"],
                          booster=replay["resume_booster"],
                          trees_per_slice=replay["trees_per_slice"])
    return sha256(rebuilt.save_raw(raw_format="json")).hexdigest()


def rebuild_batch(subject_ids: list, **kwargs) -> dict:
    """Raw-subject-ID entrypoint -- hash each id, then delegate. Mirrors
    engine/rebuild.py's function of the same name."""
    return rebuild_batch_by_ref([subject_ref(s) for s in subject_ids], **kwargs)


def rebuild(subject_id: str, **kwargs) -> dict:
    ref = subject_ref(subject_id)
    result = rebuild_batch([subject_id], **kwargs)
    return {
        "subject_ref": ref,
        "shard": result["shard"],
        "resumed_from": result["resumed_from"],
        "rows_purged": result["rows_purged"],
        "result_weights": result["result_weights"],
        "manifest": result["manifests"][ref],
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--manifest-out")
    args = parser.parse_args()
    result = rebuild(args.subject)
    if args.manifest_out:
        with open(args.manifest_out, "w") as f:
            json.dump(result["manifest"], f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
