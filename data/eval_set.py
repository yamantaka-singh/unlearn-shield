"""A frozen, non-subject evaluation corpus for scoring a promoted model.

Deliberately synthetic rows with no subject_ref -- they are not anyone's data,
so they can never be the target of an erasure and never need purge-state of
their own. That sidesteps the harder version of eval-set versioning the
original plan describes, where the eval set contains real subjects and an
erasure has to be reflected in it too. This is the easier, honest version:
noted here rather than silently presented as solving that one.

Frozen means literally frozen -- same seed, same size, forever. A benchmark
that moves is not a benchmark; an AUC drop after a rebuild has to be
attributable to the rebuild, not to also having changed what "eval" means that
day.

Note: data.synth.generate() names subjects "C{i:07d}" starting from 0, the
same scheme engine.train.build() uses, so these labels can coincide with real
shard subjects' labels as plain strings. That is cosmetic, not a leak: eval
rows are never inserted into subject_shard_map, so an erasure request -- which
looks a subject up there -- can never reach them regardless of what their
label says, and the distinct EVAL_SEED means the actual feature values differ
from any shard built with a different seed even where labels coincide.
"""

import numpy as np

from data.synth import generate

EVAL_SEED = 999_983  # arbitrary, fixed; changing it invalidates every past score
EVAL_SIZE = 300


def load() -> dict:
    return generate(n_subjects=EVAL_SIZE, seed=EVAL_SEED, max_step=720)


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve, computed directly as a rank statistic --
    P(score of a random positive > score of a random negative), with ties
    counted as half a win. No sklearn dependency for one function.

    Returns 0.5 (chance) if either class is absent, since AUC is undefined
    there and 0.5 is the least misleading placeholder for "no signal to rank."
    """
    y_true = np.asarray(y_true, dtype=bool)
    pos, neg = y_score[y_true], y_score[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average tied ranks -- otherwise arbitrary tie-breaking order shifts AUC.
    for value in np.unique(y_score):
        tied = y_score == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum_pos = ranks[y_true].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if __name__ == "__main__":
    d = load()
    print(f"{len(d['step'])} rows, {d['isFraud'].sum()} fraud, "
          f"seed={EVAL_SEED} (frozen)")
    assert abs(auc(d["isFraud"], d["isFraud"].astype(float)) - 1.0) < 1e-9, \
        "a perfect score must rank every fraud row above every non-fraud row"
    assert abs(auc(d["isFraud"], np.zeros(len(d["isFraud"]))) - 0.5) < 1e-9, \
        "no signal must score exactly chance"
    print("self-check passed")
