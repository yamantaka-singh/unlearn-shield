"""worker/jobs.py::record_eval, against a real corpus and real Postgres.

Not a mock: a fabricated AUC would be exactly the failure mode this project
exists to catch someone else committing. These tests assert the recorded
number is a real score against real weights, not merely that a row appears.
"""

import shutil
from hashlib import sha256

import psycopg2.extras
import pytest

from config.settings import CODE_DIGEST, NUM_SHARDS, NUM_SLICES
from data.eval_set import auc, load
from engine import train as train_mod
from worker.jobs import record_eval


def _file_hash(path):
    with open(path, "rb") as f:
        return sha256(f.read()).hexdigest()


@pytest.fixture
def promoted_corpus(tmp_path, monkeypatch, pg):
    shard_dir, ckpt_dir = str(tmp_path / "shards"), str(tmp_path / "ckpt")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
    train_mod.build(n_subjects=150, seed=13)
    for k in range(NUM_SHARDS):
        train_mod.train_shard(k)

    with pg, pg.cursor() as cur:
        shard_checkpoints = {}
        for shard in range(NUM_SHARDS):
            path = train_mod.checkpoint_path(shard, NUM_SLICES - 1)
            digest = _file_hash(path)
            cas_path = str(tmp_path / f"cas_{shard}.pt")
            shutil.copyfile(path, cas_path)
            cur.execute("""
                INSERT INTO checkpoints (checkpoint_hash, shard, slice_idx, file_path, code_digest)
                VALUES (%s,%s,%s,%s,%s)
            """, (digest, shard, NUM_SLICES - 1, cas_path, CODE_DIGEST))
            shard_checkpoints[str(shard)] = digest
        cur.execute("""
            INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
            VALUES ('v-test', %s, 'v0')
        """, (psycopg2.extras.Json(shard_checkpoints),))

    return shard_checkpoints


def test_record_eval_stores_a_real_auc(promoted_corpus, pg):
    with pg, pg.cursor() as cur:
        score = record_eval(cur, "v-test", promoted_corpus)

    with pg.cursor() as cur:
        cur.execute("SELECT auc, n_eval FROM eval_results WHERE model_version = 'v-test'")
        stored_auc, n_eval = cur.fetchone()

    assert stored_auc == pytest.approx(score, abs=1e-4)
    assert n_eval > 0
    assert 0.0 <= stored_auc <= 1.0


def test_recorded_auc_matches_an_independent_recomputation(promoted_corpus, pg):
    """The strongest form of this check: rebuild the same score from scratch
    using only the ensemble and the frozen eval set, independent of
    record_eval's own internals, and require an exact match."""
    from inference.batched_ensemble import load_ensemble
    import numpy as np

    with pg, pg.cursor() as cur:
        record_eval(cur, "v-test", promoted_corpus)
        cur.execute("SELECT shard, file_path FROM checkpoints WHERE checkpoint_hash = ANY(%s)",
                   (list(promoted_corpus.values()),))
        shard_paths = {str(s): p for s, p in cur.fetchall()}

    from engine.train import CHECKPOINT_DIR
    import os
    preproc_paths = {s: os.path.join(CHECKPOINT_DIR, f"shard{s}_preproc.json") for s in shard_paths}
    records = load()
    rows = np.arange(len(records["step"]))
    independent = auc(records["isFraud"],
                      load_ensemble(shard_paths, preproc_paths).predict_proba(records, rows))

    with pg.cursor() as cur:
        cur.execute("SELECT auc FROM eval_results WHERE model_version = 'v-test'")
        stored = cur.fetchone()[0]
    assert stored == pytest.approx(independent, abs=1e-6)


def test_second_promotion_does_not_overwrite_the_first(promoted_corpus, pg):
    """eval_results is history, not a single current-value row -- ON CONFLICT
    DO NOTHING, same reasoning as the checkpoints table."""
    with pg, pg.cursor() as cur:
        record_eval(cur, "v-test", promoted_corpus)
        cur.execute("""
            INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
            VALUES ('v-test-2', %s, 'v0')
        """, (psycopg2.extras.Json(promoted_corpus),))
        record_eval(cur, "v-test-2", promoted_corpus)

    with pg.cursor() as cur:
        cur.execute("SELECT model_version FROM eval_results ORDER BY model_version")
        assert [r[0] for r in cur.fetchall()] == ["v-test", "v-test-2"]


def test_eval_set_size_matches_n_eval(promoted_corpus, pg):
    with pg, pg.cursor() as cur:
        record_eval(cur, "v-test", promoted_corpus)
    with pg.cursor() as cur:
        cur.execute("SELECT n_eval FROM eval_results WHERE model_version = 'v-test'")
        n_eval = cur.fetchone()[0]
    assert n_eval == len(load()["step"])
