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

import os

import numpy as np
import xgboost as xgb

from config.determinism import enforce_determinism
from config.settings import NUM_SLICES, SEED, SHARD_DIR
from data.synth import NUMERIC_COLUMNS, TYPES

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


def rebuild(shard: int, refs: list, records: dict, min_slice: int,
            booster: xgb.Booster,
            trees_per_slice: int = TREES_PER_SLICE) -> tuple[xgb.Booster, dict]:
    """Purge `refs`, roll back to `min_slice`, boost forward on what remains.

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
