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


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve, the step-interpolated form
    (sum of precision at each threshold, weighted by the recall it gains).

    This, not ROC-AUC, is the metric that carries information at fraud
    prevalence. ROC-AUC's denominator is the negative class, and when 99.9% of
    rows are negative a model can rank almost every negative correctly, score
    0.99, and still be useless -- the false positives it does produce swamp the
    ~0.1% of rows that are actually fraud. Average precision is scored against
    the positives instead, so its baseline is the prevalence itself: a random
    model scores ~0.001 here, where it scores 0.5 on ROC-AUC. That difference
    is the whole reason this function exists alongside `auc` rather than
    instead of it -- both are reported, and only one of them moves when the
    model is genuinely good at the rare class.

    Returns 0.0 if there are no positives: undefined, and 0.0 is the reading
    that cannot be mistaken for success.
    """
    y_true = np.asarray(y_true, dtype=bool)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    # Descending score; ties broken deterministically so the number is stable.
    order = np.argsort(-y_score, kind="mergesort")
    hits = y_true[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    # Only the ranks where a positive is retrieved gain recall, so only those
    # contribute -- each by 1/n_pos of the curve.
    return float(precision[hits].sum() / n_pos)


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, target: float) -> dict:
    """Precision achievable while catching `target` of the fraud, plus the
    alert volume that costs.

    The operational question a fraud team actually asks, and the one neither
    AUC answers: "if I must catch 80% of fraud, how many alerts do I review per
    true one?" `alerts_per_catch` is the inverse of precision, reported because
    a review queue is staffed in alerts, not in percentages.
    """
    y_true = np.asarray(y_true, dtype=bool)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return {"recall_target": target, "precision": 0.0, "alerts_per_catch": None,
                "flagged": 0, "flagged_fraction": 0.0}
    order = np.argsort(-y_score, kind="mergesort")
    hits = y_true[order]
    tp = np.cumsum(hits)
    # First rank at which cumulative recall reaches the target.
    k = int(np.searchsorted(tp, np.ceil(target * n_pos))) + 1
    k = min(k, len(hits))
    precision = float(tp[k - 1] / k)
    return {"recall_target": target, "precision": precision,
            "alerts_per_catch": (1.0 / precision) if precision > 0 else None,
            "flagged": k, "flagged_fraction": k / len(hits)}


if __name__ == "__main__":
    d = load()
    print(f"{len(d['step'])} rows, {d['isFraud'].sum()} fraud, "
          f"seed={EVAL_SEED} (frozen)")
    assert abs(auc(d["isFraud"], d["isFraud"].astype(float)) - 1.0) < 1e-9, \
        "a perfect score must rank every fraud row above every non-fraud row"
    assert abs(auc(d["isFraud"], np.zeros(len(d["isFraud"]))) - 0.5) < 1e-9, \
        "no signal must score exactly chance"

    y = d["isFraud"].astype(bool)
    assert abs(average_precision(y, y.astype(float)) - 1.0) < 1e-9, \
        "a perfect ranking must have average precision 1"
    # The property that motivates reporting it: with no signal, AP falls to the
    # prevalence while AUC sits at a reassuring 0.5.
    rng = np.random.default_rng(0)
    noise_ap = average_precision(y, rng.random(len(y)))
    prevalence = y.mean()
    assert abs(noise_ap - prevalence) < 5 * prevalence, \
        f"random-ranking AP {noise_ap:.4f} should sit near prevalence {prevalence:.4f}"

    perfect = precision_at_recall(y, y.astype(float), 0.8)
    assert perfect["precision"] == 1.0, "a perfect ranking needs no false positives"
    print(f"self-check passed (prevalence {prevalence:.4f}, random-ranking AP {noise_ap:.4f})")
