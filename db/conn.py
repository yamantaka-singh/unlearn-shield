"""psycopg2 connections: fresh for offline work, pooled for the serving path.

The original version here opened a fresh connection per call and left a note to
add a pool "when a profiler shows connection setup cost actually matters."
It does: `psycopg2.connect` costs ~6.2ms against local Postgres, which is 41x
the 0.15ms the shard ensemble spends on actual forward passes, and the single
largest term in /v1/predict's latency.

`connect()` stays fresh-per-call for the worker and CLI scripts -- they hold a
connection for the length of a rebuild, so pooling buys nothing and a pooled
connection checked out for minutes is worse than a dedicated one.

`pooled()` is for request handlers, where connection setup dominates.
"""

import threading
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool

from config.settings import DASHBOARD_DATABASE_URL, DATABASE_URL

_pool = None
_pool_lock = threading.Lock()


def connect():
    conn = psycopg2.connect(DATABASE_URL)
    psycopg2.extras.register_uuid()
    return conn


def connect_readonly():
    """The dashboard's connection, bound to the DB role that cannot write
    (db/schema.sql). A role that cannot write enforces "the dashboard never
    writes to the DB directly" at the database, not only in application code
    a future edit could bypass.

    autocommit=True, deliberately: a plain psycopg2 connection implicitly
    opens a transaction on the first query and leaves it open until an
    explicit commit. The dashboard holds one connection for its whole
    lifetime, so a read-only session with no commit call would hold a
    snapshot -- and its locks -- open indefinitely. Caught by a real deadlock
    in tests/e2e/test_dashboard.py: a stale open transaction on this
    connection blocked a later test's TRUNCATE, which never released the
    connection was waiting to reuse. Autocommit means each SELECT starts and
    ends its own transaction, which is also just the correct mode for a
    connection that only ever reads.
    """
    conn = psycopg2.connect(DASHBOARD_DATABASE_URL)
    conn.autocommit = True
    psycopg2.extras.register_uuid()
    return conn


def _get_pool():
    global _pool
    # Double-checked under a lock: FastAPI serves requests from a thread pool,
    # so first-request races would otherwise build several pools and leak all
    # but one.
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                psycopg2.extras.register_uuid()
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 16, DATABASE_URL)
    return _pool


@contextmanager
def pooled():
    """Borrow a connection for the duration of one request handler."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        # Return it usable: an un-rolled-back error leaves the connection in a
        # failed transaction state, and the next borrower inherits it.
        if conn.closed:
            pool.putconn(conn, close=True)
        else:
            conn.rollback()
            pool.putconn(conn)


def reset_pool():
    """Drop the pool. Tests truncate tables between cases and swap DATABASE_URL;
    a pool holding connections from a previous configuration outlives that."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
