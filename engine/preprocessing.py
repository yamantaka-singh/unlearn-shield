"""Per-shard preprocessing. The single most important correctness rule here.

Fit scalers on ONE shard's data, never globally. A scaler fit on the full
dataset before sharding bakes every subject's values into every shard's
normalisation constants, so a deleted subject's numbers survive in the feature
scale of four other shards after their rows are gone from their own. The
isolation guarantee that engine/train.py sells is false if this is broken
anywhere.

What is NOT a violation: declaring the transaction-type enum globally.
`TYPES` is schema -- the set of values the column may hold -- not a statistic
derived from anyone's data. Deriving the vocabulary from data would be a leak
AND would give shards different feature widths, breaking the fixed architecture
that Phase 5's batched ensemble needs. Frequency or target encodings, by
contrast, ARE fitted statistics and must stay per-shard.
"""

import json

import numpy as np

from data.synth import TYPES

# log1p first: PaySim balances and amounts are heavy-tailed enough that raw
# standardisation leaves almost all mass in a spike near zero. A fixed transform
# applied before fitting, so it derives nothing from the data.
_LOG_COLUMNS = ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
_RAW_COLUMNS = ("step",)
_NUMERIC = _RAW_COLUMNS + _LOG_COLUMNS

N_FEATURES = len(_NUMERIC) + len(TYPES)


def _numeric_matrix(records: dict, rows: np.ndarray) -> np.ndarray:
    cols = [records[c][rows] for c in _RAW_COLUMNS]
    cols += [np.log1p(records[c][rows]) for c in _LOG_COLUMNS]
    return np.column_stack(cols)


class ShardPreprocessor:
    """Mean/std over one shard's SLICE 0 rows. numpy, not sklearn -- this is
    two reductions, and a dependency that pickles fitted state is a liability
    when the fitted state has to be re-derived on every rebuild.

    Slice 0 rather than the whole shard, and that is load-bearing. If these
    constants were fit on every slice, then every checkpoint in the shard would
    encode the scaling influence of every subject in it -- so rolling back to
    checkpoint j-1 to delete a subject in slice j would resume from weights
    that still carry that subject. The rebuild would be deterministic, the
    spot-check would pass, and the erasure would be incomplete.

    Fitting on slice 0 makes the preprocessor a function of slice-0 subjects
    only. Deleting anyone in a later slice leaves it untouched. Deleting a
    slice-0 subject correctly forces a refit and a full-shard retrain -- and
    slices are churn-ascending, so slice 0 holds the subjects least likely to
    be deleted. Only sound because slices are subject-aligned (engine/slicer.py).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, records: dict, rows: np.ndarray) -> "ShardPreprocessor":
        if len(rows) == 0:
            raise ValueError("cannot fit a preprocessor on zero rows")
        x = _numeric_matrix(records, rows)
        std = x.std(axis=0)
        # A constant column within a shard is legitimate (a small cold shard may
        # hold one transaction type). Guard the divide rather than dropping the
        # column, which would change the feature width per shard.
        return cls(x.mean(axis=0), np.where(std < 1e-9, 1.0, std))

    def transform(self, records: dict, rows: np.ndarray) -> np.ndarray:
        numeric = (_numeric_matrix(records, rows) - self.mean) / self.std
        onehot = np.zeros((numeric.shape[0], len(TYPES)))
        onehot[np.arange(numeric.shape[0]), records["type_idx"][rows]] = 1.0
        return np.column_stack([numeric, onehot]).astype(np.float32)

    def to_json(self) -> str:
        return json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()},
                          sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ShardPreprocessor":
        d = json.loads(text)
        return cls(np.array(d["mean"]), np.array(d["std"]))
