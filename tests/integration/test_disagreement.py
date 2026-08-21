"""The optional shard-disagreement review queue (ADR 0009).

Off by default. These tests cover both states, and the privacy decision that
shaped the table -- a schema guard that fails if anyone adds a transaction
feature column without also wiring this table into the erasure path.
"""

import numpy as np
import pytest

from gateway import disagreement


def test_disabled_by_default():
    """The whole point of "optional": a fresh checkout runs with this off,
    and DISAGREEMENT_THRESHOLD=0.0 means the code path is never entered."""
    from config.settings import DISAGREEMENT_THRESHOLD
    assert DISAGREEMENT_THRESHOLD == 0.0
    assert not disagreement.is_enabled()


def test_spread_is_population_std():
    """Population (ddof=0), not sample: these shards are the entire ensemble,
    not a sample from a larger population, so there is no n-1 correction."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert disagreement.spread(scores) == pytest.approx(np.std(scores, ddof=0))
    assert disagreement.spread(scores) != pytest.approx(np.std(scores, ddof=1))


def test_unanimous_shards_have_zero_spread():
    assert disagreement.spread(np.array([0.3, 0.3, 0.3, 0.3, 0.3])) == pytest.approx(0.0)


def test_one_dissenting_shard_raises_spread():
    """The scenario the feature exists for: most shards calm, one alarmed."""
    calm = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    dissent = np.array([0.05, 0.05, 0.05, 0.05, 0.90])
    assert disagreement.spread(dissent) > disagreement.spread(calm)


def test_table_stores_no_transaction_features(pg):
    """Regression guard for the privacy decision in ADR 0009.

    PredictRequest carries no subject_id, so a row here can never be reached
    by an erasure (which routes subject_ref -> subject_shard_map -> rebuild).
    A feature column would therefore create personal data this system
    promises to be able to delete and could not. Adding one requires adding
    subject_ref here first and wiring this table into engine/rebuild.py's
    purge -- at which point this test should be updated deliberately, not
    deleted in passing.
    """
    with pg.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'disagreement_reviews'
        """)
        columns = {r[0] for r in cur.fetchall()}

    feature_columns = {"step", "amount", "oldbalanceorg", "newbalanceorig",
                       "oldbalancedest", "newbalancedest", "type", "type_idx",
                       "features", "payload", "transaction"}
    assert not (columns & feature_columns), (
        f"disagreement_reviews gained a transaction-feature column "
        f"({columns & feature_columns}) with no subject_ref and no purge path")
    assert "subject_ref" not in columns, (
        "a subject_ref here would need engine/rebuild.py to purge this table too")


def test_record_writes_expected_row(pg, monkeypatch):
    monkeypatch.setattr(disagreement, "DISAGREEMENT_THRESHOLD", 0.05)
    scores = np.array([0.10, 0.12, 0.11, 0.90, 0.13])

    disagreement.record("v-test", scores, float(scores.mean()),
                        disagreement.spread(scores))

    with pg.cursor() as cur:
        cur.execute("""
            SELECT model_version, shard_scores, mean_score, spread, threshold, status
            FROM disagreement_reviews
        """)
        rows = cur.fetchall()
    assert len(rows) == 1
    model_version, shard_scores, mean_score, spread, threshold, status = rows[0]
    assert model_version == "v-test"
    assert len(shard_scores) == 5
    assert shard_scores[3] == pytest.approx(0.90, abs=1e-5)
    assert mean_score == pytest.approx(scores.mean(), abs=1e-5)
    assert spread == pytest.approx(disagreement.spread(scores), abs=1e-5)
    assert threshold == pytest.approx(0.05)
    assert status == "pending"


def test_record_swallows_db_errors(monkeypatch):
    """An optional side-channel must never turn a successful prediction into
    a 500. record() runs in a BackgroundTask; a raise here would surface as a
    server error on a request that already succeeded."""
    def exploding_pool():
        raise RuntimeError("database on fire")

    monkeypatch.setattr(disagreement, "pooled", exploding_pool)
    disagreement.record("v", np.array([0.1, 0.2]), 0.15, 0.05)  # must not raise


def test_threshold_is_recorded_per_row_not_read_at_review_time(pg, monkeypatch):
    """The threshold is env-configurable, so a later change would otherwise
    make existing rows uninterpretable -- flagged because it was extreme, or
    because the bar was low that week?"""
    scores = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
    monkeypatch.setattr(disagreement, "DISAGREEMENT_THRESHOLD", 0.05)
    disagreement.record("v1", scores, 0.42, 0.39)
    monkeypatch.setattr(disagreement, "DISAGREEMENT_THRESHOLD", 0.30)
    disagreement.record("v2", scores, 0.42, 0.39)

    with pg.cursor() as cur:
        cur.execute("SELECT model_version, threshold FROM disagreement_reviews ORDER BY review_id")
        assert [(v, round(t, 2)) for v, t in cur.fetchall()] == [("v1", 0.05), ("v2", 0.30)]
