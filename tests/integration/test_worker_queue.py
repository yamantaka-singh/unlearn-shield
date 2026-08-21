"""Claim-then-commit and the lease reaper -- what makes a crashed worker's
jobs recoverable instead of stuck in 'processing' until the schema itself
runs out of headroom.
"""

from datetime import timedelta
from uuid import uuid4

from worker.queue import claim_batch, reap_expired_leases


def _job(cur, ref, key):
    cur.execute("""
        INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard,
                                  idempotency_key, sla_deadline, requested_by)
        VALUES (%s, %s, 'fraud_excision', 0, %s, now() + interval '720 hours', 'test')
    """, (str(uuid4()), ref, key))


def test_claim_marks_processing_and_returns_the_row(pg):
    with pg, pg.cursor() as cur:
        _job(cur, "ref-a", "k-a")
        jobs = claim_batch(cur, "worker-1")
    assert len(jobs) == 1
    assert jobs[0]["subject_ref"] == "ref-a"
    assert isinstance(jobs[0]["erasure_id"], str)  # not a uuid.UUID -- see worker/queue.py

    with pg.cursor() as cur:
        cur.execute("SELECT status, leased_by FROM erasure_jobs WHERE subject_ref = 'ref-a'")
        status, leased_by = cur.fetchone()
    assert status == "processing" and leased_by == "worker-1"


def test_a_queued_job_is_claimed_only_once(pg):
    """Two workers racing for the same row: SKIP LOCKED means the second
    worker's claim simply excludes whatever the first already locked."""
    with pg, pg.cursor() as cur:
        _job(cur, "ref-b", "k-b")

    with pg.cursor() as cur1, pg.cursor() as cur2:
        first = claim_batch(cur1, "worker-1")
        # cur1 has not committed yet -- SKIP LOCKED must exclude the row cur1 holds.
        second = claim_batch(cur2, "worker-2")
    pg.commit()
    assert len(first) == 1
    assert second == []


def test_reaper_requeues_only_expired_leases(pg):
    with pg, pg.cursor() as cur:
        _job(cur, "ref-c", "k-c")
        _job(cur, "ref-d", "k-d")
        cur.execute("""
            UPDATE erasure_jobs SET status = 'processing', leased_by = 'dead-worker',
                lease_expires_at = now() - interval '1 minute' WHERE subject_ref = 'ref-c'
        """)
        cur.execute("""
            UPDATE erasure_jobs SET status = 'processing', leased_by = 'live-worker',
                lease_expires_at = now() + interval '30 minutes' WHERE subject_ref = 'ref-d'
        """)
        n = reap_expired_leases(cur)

    assert n == 1
    with pg.cursor() as cur:
        cur.execute("SELECT subject_ref, status, leased_by FROM erasure_jobs ORDER BY subject_ref")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    assert rows["ref-c"] == ("queued", None)
    assert rows["ref-d"] == ("processing", "live-worker")


def test_claim_orders_by_shard_so_a_batch_groups_naturally(pg):
    with pg, pg.cursor() as cur:
        for i in range(4):
            cur.execute("""
                INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard,
                                          idempotency_key, sla_deadline, requested_by)
                VALUES (%s, %s, 'fraud_excision', %s, %s, now() + interval '720 hours', 'test')
            """, (str(uuid4()), f"ref-{i}", i % 2, f"k-{i}"))
        jobs = claim_batch(cur, "worker-1", limit=10)
    shards = [j["shard"] for j in jobs]
    assert shards == sorted(shards)
