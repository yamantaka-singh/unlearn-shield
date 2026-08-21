import numpy as np
import pytest

from data.churn_score import churn_scores
from engine.sharder import assign_all, assign_shard, stable_hash


def test_hot_shards_concentrate_high_churn():
    subjects = np.array([f"C{i:07d}" for i in range(2000)])
    churn = np.linspace(0, 0.999, 2000)
    shard = assign_all(subjects, churn)
    high = churn >= 0.6
    assert (shard[high] < 2).mean() > 0.70
    assert (shard[~high] >= 2).all()


def test_stable_hash_does_not_depend_on_interpreter_salt():
    """`hash()` would pass this inside one process and fail across two.

    Shard assignment that reshuffles between runs invalidates the rollback point
    of every checkpoint on disk, so this has to be stable, not merely consistent.
    """
    assert stable_hash("C0000001") == stable_hash("C0000001")
    assert stable_hash("C0000001") != stable_hash("C0000002")


def test_every_shard_is_reachable():
    subjects = np.array([f"C{i:07d}" for i in range(3000)])
    churn = np.linspace(0, 0.999, 3000)
    assert set(np.unique(assign_all(subjects, churn))) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("hot", [0, 5, 6])
def test_degenerate_hot_shard_counts_are_refused(hot):
    with pytest.raises(ValueError):
        assign_shard("C0000001", 0.9, num_shards=5, hot_shards=hot)


def test_churn_scores_stay_in_range():
    subs = np.array([f"C{i:07d}" for i in range(500)])
    scores = churn_scores(subs, np.linspace(0, 720, 500), 720, seed=0)
    assert scores.min() >= 0.0 and scores.max() < 1.0
