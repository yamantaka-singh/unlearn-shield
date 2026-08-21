import numpy as np
import pytest

from data.eval_set import EVAL_SEED, auc, load


def test_perfect_ranking_scores_one():
    y = np.array([0, 1, 0, 1, 1], dtype=bool)
    assert abs(auc(y, y.astype(float)) - 1.0) < 1e-9


def test_no_signal_scores_exactly_chance():
    y = np.array([0, 1, 0, 1, 1], dtype=bool)
    assert abs(auc(y, np.zeros(len(y))) - 0.5) < 1e-9


def test_single_class_returns_chance_not_undefined():
    """AUC is mathematically undefined with one class present. 0.5 is the
    least misleading placeholder -- 'no signal to rank' -- rather than NaN,
    which would silently poison a mean() or a chart."""
    assert auc(np.array([True, True, True]), np.array([0.9, 0.1, 0.5])) == 0.5
    assert auc(np.array([False, False]), np.array([0.9, 0.1])) == 0.5


def test_tied_scores_average_their_ranks():
    """A naive rank assignment (first-seen order) would make AUC depend on
    array order for tied scores, which is not a property AUC should have."""
    y = np.array([True, False, True, False])
    tied = np.array([0.5, 0.5, 0.5, 0.5])
    assert auc(y, tied) == 0.5


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force_pairwise_count(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(4, 30)
    y = rng.integers(0, 2, size=n).astype(bool)
    if y.all() or not y.any():
        pytest.skip("degenerate class split")
    scores = rng.normal(size=n)
    pos, neg = scores[y], scores[~y]
    wins = sum(1 for p in pos for q in neg if p > q)
    ties = sum(1 for p in pos for q in neg if p == q)
    brute = (wins + 0.5 * ties) / (len(pos) * len(neg))
    assert abs(auc(y, scores) - brute) < 1e-9


def test_frozen_seed_is_actually_frozen():
    """The whole point: same corpus, forever. If this constant ever changes,
    every historical eval_results row becomes incomparable to new ones."""
    assert EVAL_SEED == 999_983


def test_load_is_deterministic_across_calls():
    a, b = load(), load()
    assert np.array_equal(a["step"], b["step"])
    assert np.array_equal(a["isFraud"], b["isFraud"])


def test_eval_rows_carry_no_subject_ref():
    """The property that lets this data skip purge-state entirely: it is
    never inserted into subject_shard_map, so no erasure request can ever
    reach it regardless of what its synthetic label happens to be."""
    assert "subject_ref" not in load()
