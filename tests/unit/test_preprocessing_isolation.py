import numpy as np
import pytest

from engine.preprocessing import N_FEATURES, ShardPreprocessor
from data.synth import TYPES, generate


def _records(seed):
    r = generate(n_subjects=120, seed=seed)
    r["slice_idx"] = np.zeros(len(r["step"]), dtype=np.int64)
    return r


def test_a_shards_statistics_ignore_another_shards_values():
    """The leak this rule exists to prevent.

    A scaler fit on the full dataset before sharding bakes every subject's
    values into every shard's normalisation constants, so a deleted subject's
    numbers survive in four other shards' feature scale after their rows are
    gone from their own.
    """
    a, b = _records(1), _records(2)
    rows_a = np.arange(len(a["step"]))

    before = ShardPreprocessor.fit(a, rows_a)
    b["amount"] = b["amount"] * 1000.0  # a large change in a different shard
    after = ShardPreprocessor.fit(a, rows_a)

    assert np.array_equal(before.mean, after.mean)
    assert np.array_equal(before.std, after.std)


def test_transform_width_is_identical_across_shards():
    """Phase 5 stacks these submodels into one batched forward pass, which needs
    matching input widths -- so the type vocabulary must be schema, not fitted."""
    widths = set()
    for seed in range(4):
        r = _records(seed)
        rows = np.arange(len(r["step"]))
        widths.add(ShardPreprocessor.fit(r, rows).transform(r, rows).shape[1])
    assert widths == {N_FEATURES}


def test_shard_missing_a_transaction_type_keeps_full_width():
    r = _records(3)
    rows = np.flatnonzero(r["type_idx"] != TYPES.index("DEBIT"))
    out = ShardPreprocessor.fit(r, rows).transform(r, rows)
    assert out.shape[1] == N_FEATURES
    assert out[:, len(_NUMERIC_NAMES) + TYPES.index("DEBIT")].sum() == 0.0


_NUMERIC_NAMES = ("step", "amount", "oldbalanceOrg", "newbalanceOrig",
                  "oldbalanceDest", "newbalanceDest")


def test_constant_column_does_not_produce_nan():
    r = _records(4)
    rows = np.arange(len(r["step"]))
    r["step"] = np.full_like(r["step"], 5.0)
    out = ShardPreprocessor.fit(r, rows).transform(r, rows)
    assert np.isfinite(out).all()


def test_fitting_on_zero_rows_is_refused():
    with pytest.raises(ValueError, match="zero rows"):
        ShardPreprocessor.fit(_records(5), np.array([], dtype=np.int64))


def test_json_round_trip_preserves_constants():
    r = _records(6)
    p = ShardPreprocessor.fit(r, np.arange(len(r["step"])))
    q = ShardPreprocessor.from_json(p.to_json())
    assert np.array_equal(p.mean, q.mean) and np.array_equal(p.std, q.std)
