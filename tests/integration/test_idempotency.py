"""INSERT ... ON CONFLICT DO NOTHING closes the race a check-then-insert pair
would leave open, even inside one transaction, without needing SERIALIZABLE."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from gateway.idempotency import insert_or_get


def _deadline():
    return datetime.now(timezone.utc) + timedelta(hours=720)


def test_same_key_returns_the_same_job(pg):
    ref = uuid4().hex
    with pg, pg.cursor() as cur:
        id1, created1 = insert_or_get(cur, subject_ref=ref, reason="fraud_excision", shard=0,
                                      idempotency_key="k1", sla_deadline=_deadline(),
                                      requested_by="test")
        id2, created2 = insert_or_get(cur, subject_ref=ref, reason="fraud_excision", shard=0,
                                      idempotency_key="k1", sla_deadline=_deadline(),
                                      requested_by="test")
    assert id1 == id2
    assert created1 and not created2


def test_same_key_produces_exactly_one_row(pg):
    ref = uuid4().hex
    with pg, pg.cursor() as cur:
        for _ in range(5):
            insert_or_get(cur, subject_ref=ref, reason="fraud_excision", shard=0,
                          idempotency_key="k-repeat", sla_deadline=_deadline(),
                          requested_by="test")
        cur.execute("SELECT count(*) FROM erasure_jobs WHERE idempotency_key = 'k-repeat'")
        assert cur.fetchone()[0] == 1


def test_different_keys_produce_different_jobs(pg):
    ref = uuid4().hex
    with pg, pg.cursor() as cur:
        id1, _ = insert_or_get(cur, subject_ref=ref, reason="fraud_excision", shard=0,
                               idempotency_key="k-a", sla_deadline=_deadline(), requested_by="test")
        id2, _ = insert_or_get(cur, subject_ref=ref, reason="fraud_excision", shard=0,
                               idempotency_key="k-b", sla_deadline=_deadline(), requested_by="test")
    assert id1 != id2


def test_idempotency_key_is_required_at_the_schema_level(pg):
    """NOT NULL, not just UNIQUE -- a nullable-unique key lets every request
    missing the header through, enforced only by app code a retry path could
    bypass. This is the schema fix from Phase 0's plan review."""
    import psycopg2
    with pg.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard,
                                          idempotency_key, sla_deadline, requested_by)
                VALUES (gen_random_uuid(), %s, 'fraud_excision', 0, NULL, now(), 'test')
            """, (uuid4().hex,))
            pg.commit()
            assert False, "NULL idempotency_key should have been rejected"
        except psycopg2.errors.NotNullViolation:
            pg.rollback()
