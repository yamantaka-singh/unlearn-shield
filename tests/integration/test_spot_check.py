"""The reproducibility spot-check (Phase 7), deferred since Phase 4.

Phase 4's note said this needed a pre-purge snapshot of the shard. It does
not -- the claim being checked is that retraining from `resumed_from` on the
RETAINED data yields `result_weights`, and the retained data is exactly what
the shard holds immediately after the rebuild.
"""

import psycopg2.extras
import pytest
from nacl.signing import SigningKey

from config.settings import NUM_SHARDS, subject_ref
from engine import rebuild as rebuild_mod
from engine import train as train_mod
from worker.jobs import _should_spot_check, run_spot_check


@pytest.fixture
def corpus(tmp_path, monkeypatch, pg):
    shard_dir, ckpt_dir = str(tmp_path / "shards"), str(tmp_path / "ckpt")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(rebuild_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())
    routing = train_mod.build(n_subjects=120, seed=31)
    for k in range(NUM_SHARDS):
        train_mod.train_shard(k)
    return routing


def _job_row(cur, ref, shard):
    cur.execute("""
        INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard, idempotency_key,
                                  sla_deadline, requested_by, status)
        VALUES (gen_random_uuid(), %s, 'fraud_excision', %s, %s,
                now() + interval '720 hours', 'test', 'done')
        RETURNING erasure_id
    """, (ref, shard, f"spot-{ref[:8]}"))
    return str(cur.fetchone()[0])


def test_spot_check_reproduces_a_real_rebuild(corpus, pg):
    """The positive case, end to end: rebuild, immediately re-run it, and get
    byte-identical weights."""
    target = next(s for s in (f"C{i:07d}" for i in range(120)) if subject_ref(s) in corpus)
    result = rebuild_mod.rebuild_batch_by_ref([subject_ref(target)])

    with pg, pg.cursor() as cur:
        erasure_id = _job_row(cur, subject_ref(target), result["shard"])
        matched = run_spot_check(cur, erasure_id, result["shard"],
                                 result["replay"], result["result_weights"])
    assert matched

    with pg.cursor() as cur:
        cur.execute("""
            SELECT expected_weights, observed_weights, matched
            FROM reproducibility_checks WHERE erasure_id = %s
        """, (erasure_id,))
        expected, observed, stored_match = cur.fetchone()
    assert stored_match is True
    assert expected == observed == result["result_weights"]


def test_spot_check_detects_a_genuine_divergence(corpus, pg):
    """The negative control, and the one that matters.

    A check that only ever passes proves nothing. Here the manifest's claimed
    weights are corrupted, so the re-run legitimately disagrees -- exactly the
    shape of a real determinism drift, where the recorded digest and a fresh
    run stop matching.
    """
    target = next(s for s in (f"C{i:07d}" for i in range(120)) if subject_ref(s) in corpus)
    result = rebuild_mod.rebuild_batch_by_ref([subject_ref(target)])
    tampered = "0" * 64

    with pg, pg.cursor() as cur:
        erasure_id = _job_row(cur, subject_ref(target), result["shard"])
        matched = run_spot_check(cur, erasure_id, result["shard"],
                                 result["replay"], tampered)
    assert not matched

    with pg.cursor() as cur:
        cur.execute("SELECT observed_weights, matched FROM reproducibility_checks "
                   "WHERE erasure_id = %s", (erasure_id,))
        observed, stored_match = cur.fetchone()
    assert stored_match is False
    assert observed == result["result_weights"], (
        "the row must record what the re-run ACTUALLY produced, not the "
        "claim it was compared against")


def test_spot_check_does_not_touch_the_real_checkpoints(corpus, pg, tmp_path):
    """A re-run writes checkpoints. If it wrote to the real paths it would
    overwrite the promoted checkpoint -- harmless on a pass, destructive on a
    failure, which is precisely when you least want the next rebuild resuming
    from weights that match no recorded hash."""
    from pathlib import Path

    target = next(s for s in (f"C{i:07d}" for i in range(120)) if subject_ref(s) in corpus)
    result = rebuild_mod.rebuild_batch_by_ref([subject_ref(target)])

    ckpt_dir = Path(train_mod.CHECKPOINT_DIR)
    before = {p.name: p.read_bytes() for p in ckpt_dir.glob("*.pt")}

    with pg, pg.cursor() as cur:
        run_spot_check(cur, _job_row(cur, subject_ref(target), result["shard"]),
                       result["shard"], result["replay"], result["result_weights"])

    after = {p.name: p.read_bytes() for p in ckpt_dir.glob("*.pt")}
    assert before == after


def test_selection_is_deterministic_and_not_the_workers_choice():
    """Same erasure_id and audit_key always give the same answer, so an
    auditor holding audit_key can recompute the sample and confirm the
    operator did not steer it."""
    ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(500)]
    first = [_should_spot_check(i) for i in ids]
    assert first == [_should_spot_check(i) for i in ids]


def test_selection_rate_is_near_the_configured_rate(monkeypatch):
    import worker.jobs as jobs
    monkeypatch.setattr(jobs, "SPOT_CHECK_RATE", 0.10)
    ids = [f"{i:08d}-0000-0000-0000-00000000abcd" for i in range(5000)]
    rate = sum(jobs._should_spot_check(i) for i in ids) / len(ids)
    assert 0.08 < rate < 0.12, f"HMAC-derived selection rate {rate:.3f} is off target"
