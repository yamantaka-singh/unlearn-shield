"""Idempotent job creation.

INSERT ... ON CONFLICT DO NOTHING is one statement, so there is no window
between "check if this key exists" and "insert it" for a concurrent duplicate
request to land in. A check-then-insert pair, even inside one transaction,
still needs SERIALIZABLE isolation to close that race; this closes it with the
default isolation level for free.
"""

from uuid import uuid4

import psycopg2.extras


def insert_or_get(cur, *, subject_ref: str, reason: str, shard: int,
                  idempotency_key: str, sla_deadline, requested_by: str) -> tuple[str, bool]:
    """Returns (erasure_id, created). created=False means this Idempotency-Key
    was already used and the original job's id is returned instead."""
    erasure_id = str(uuid4())
    cur.execute("""
        INSERT INTO erasure_jobs
            (erasure_id, subject_ref, reason, shard, idempotency_key, sla_deadline, requested_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING erasure_id
    """, (erasure_id, subject_ref, reason, shard, idempotency_key, sla_deadline, requested_by))
    row = cur.fetchone()
    if row is not None:
        return str(row[0]), True

    cur.execute("SELECT erasure_id FROM erasure_jobs WHERE idempotency_key = %s", (idempotency_key,))
    existing = cur.fetchone()
    return str(existing[0]), False
