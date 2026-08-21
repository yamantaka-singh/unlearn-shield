"""Claim-then-commit job polling. No long-held transaction across a rebuild.

A naive `SELECT ... FOR UPDATE SKIP LOCKED` inside the same transaction that
then runs training holds a row lock and an open transaction for however long
the rebuild takes -- minutes. That bloats Postgres and, if the worker dies
mid-rebuild, leaves the job locked forever with no way for another worker to
pick it up. Claiming sets a lease and commits immediately; training happens
outside any transaction. `reap_expired_leases` requeues anything whose lease
ran out, which is what makes a crashed worker's jobs recoverable at all.
"""

from config.settings import LEASE_SECONDS, POLL_BATCH_SIZE


def reap_expired_leases(cur) -> int:
    cur.execute("""
        UPDATE erasure_jobs SET status = 'queued', leased_by = NULL, lease_expires_at = NULL
        WHERE status = 'processing' AND lease_expires_at < now()
    """)
    return cur.rowcount


def claim_batch(cur, worker_id: str, limit: int = POLL_BATCH_SIZE) -> list[dict]:
    """Claims up to `limit` queued jobs, returned sorted by shard.

    The subquery's `ORDER BY shard, sla_deadline` governs which rows SKIP
    LOCKED chooses -- earliest-deadline-first within each shard -- but
    Postgres does not guarantee an UPDATE...RETURNING preserves a subquery's
    row order. Sorting the fetched rows here, rather than trusting the SQL
    order, is what actually makes it safe for a caller to group consecutive
    rows by shard into one batch.
    """
    cur.execute("""
        UPDATE erasure_jobs SET status = 'processing', leased_by = %(worker_id)s,
            lease_expires_at = now() + make_interval(secs => %(lease_seconds)s),
            attempts = attempts + 1
        WHERE erasure_id IN (
            SELECT erasure_id FROM erasure_jobs
            WHERE status = 'queued'
            ORDER BY shard, sla_deadline
            FOR UPDATE SKIP LOCKED
            LIMIT %(limit)s
        )
        RETURNING erasure_id, subject_ref, shard
    """, {"worker_id": worker_id, "lease_seconds": LEASE_SECONDS, "limit": limit})
    cols = [d.name for d in cur.description]
    # str(): psycopg2.extras.register_uuid() (db/conn.py) returns erasure_id as
    # a uuid.UUID, but everything downstream -- SQL params, dict keys, HMAC
    # input in worker/jobs.py -- expects the same string form the API returns.
    rows = [{c: (str(v) if c == "erasure_id" else v) for c, v in zip(cols, row)}
           for row in cur.fetchall()]
    rows.sort(key=lambda r: r["shard"])
    return rows
