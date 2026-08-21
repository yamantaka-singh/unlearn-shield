import numpy as np

from engine.slicer import assign_slices


def _subjects(n, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array([f"C{i:07d}" for i in range(n)])
    return ids, rng.random(n), 1 + rng.geometric(0.25, size=n)


def test_high_churn_subjects_land_in_late_slices():
    """Late slices are where rollback is cheapest, so the likeliest deletions
    belong there."""
    ids, _, counts = _subjects(400)
    churn = np.linspace(0, 0.999, 400)
    slices = assign_slices(ids, churn, counts)
    assert slices[churn >= 0.8].mean() > slices[churn <= 0.2].mean()


def test_slices_balance_by_record_count_not_subject_count():
    ids, churn, counts = _subjects(600)
    slices = assign_slices(ids, churn, counts)
    per_slice = np.bincount(slices, weights=counts, minlength=5)
    assert per_slice.max() / per_slice.min() < 1.25


def test_all_slices_used_and_in_range():
    ids, churn, counts = _subjects(600)
    slices = assign_slices(ids, churn, counts)
    assert set(np.unique(slices)) == {0, 1, 2, 3, 4}
