"""Test-wide signing key.

`engine.rebuild` refuses to produce an unsigned manifest, so every test that
runs a rebuild needs a key. Generating a throwaway one per session keeps the
suite from depending on whatever happens to be in the developer's environment --
which is how the suite passed locally and would have failed in CI.
"""

import os

import pytest
from nacl.signing import SigningKey


@pytest.fixture(autouse=True, scope="session")
def _signing_key():
    os.environ.setdefault("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())


TABLES = ("reproducibility_checks", "erasure_manifests", "erasure_jobs",
         "checkpoints", "model_versions", "subject_shard_map")


@pytest.fixture
def pg():
    """A real Postgres connection, tables truncated first.

    The worker's queue relies on `FOR UPDATE SKIP LOCKED` semantics SQLite
    doesn't have, so faking the DB layer would test something other than what
    ships. Skips cleanly when unreachable rather than failing, so
    `pytest tests/unit` stays fast and hermetic while these run wherever
    Postgres exists: locally with Docker, or in CI.
    """
    from db.conn import connect
    try:
        conn = connect()
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    with conn, conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    yield conn
    conn.close()
