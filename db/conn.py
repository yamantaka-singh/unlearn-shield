"""One psycopg2 connection helper. No pool, no ORM -- six tables and the volume
in Assumption 2 (tens to low hundreds of jobs/day) don't justify either.

ponytail: a fresh connection per call. Add a pool (e.g. psycopg2.pool) when a
profiler shows connection setup cost actually matters, not before.
"""

import psycopg2
import psycopg2.extras

from config.settings import DATABASE_URL


def connect():
    conn = psycopg2.connect(DATABASE_URL)
    psycopg2.extras.register_uuid()
    return conn
