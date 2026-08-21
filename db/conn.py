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

from config.settings import DATABASE_URL

_pool = None
_pool_lock = threading.Lock()


def connect():
    conn = psycopg2.connect(DATABASE_URL)
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
