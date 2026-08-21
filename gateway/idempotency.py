"""Idempotent job creation, scoped per calling principal.

INSERT ... ON CONFLICT DO NOTHING is one statement, so there is no window
between "check if this key exists" and "insert it" for a concurrent duplicate
request to land in. A check-then-insert pair, even inside one transaction,
still needs SERIALIZABLE isolation to close that race; this closes it with the
default isolation level for free.

The conflict target is (requested_by, idempotency_key), not idempotency_key
alone. Keyed globally, one caller's key collided with another's: the second
caller received the first's erasure_id, and its own request was silently
dropped while still returning 202 -- a legally-required erasure reporting
success without happening. See migration 0004.
"""

from uuid import uuid4


def insert_or_get(cur, *, subject_ref: str, reason: str, shard: int,
                  idempotency_key: str, sla_deadline, requested_by: str) -> tuple[str, bool]:
    """Returns (erasure_id, created). created=False means THIS principal already
    used this Idempotency-Key, and the original job's id is returned instead."""
    erasure_id = str(uuid4())
    cur.execute("""
        INSERT INTO erasure_jobs
            (erasure_id, subject_ref, reason, shard, idempotency_key, sla_deadline, requested_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (requested_by, idempotency_key) DO NOTHING
        RETURNING erasure_id
    """, (erasure_id, subject_ref, reason, shard, idempotency_key, sla_deadline, requested_by))
    row = cur.fetchone()
    if row is not None:
        return str(row[0]), True

    # Scoped to requested_by for the same reason as the conflict target: an
    # unscoped lookup here would hand back another principal's erasure_id even
    # with the constraint fixed.
    cur.execute("""
        SELECT erasure_id FROM erasure_jobs
        WHERE requested_by = %s AND idempotency_key = %s
    """, (requested_by, idempotency_key))
    return str(cur.fetchone()[0]), False
