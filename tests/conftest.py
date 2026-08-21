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


# Every table, so a case never inherits another's rows. eval_results was
# already being cleared by CASCADE through its model_versions foreign key;
# disagreement_reviews has no FK, so it genuinely leaked rows between tests
# until listed here. Listing both explicitly rather than relying on which
# ones happen to have a cascading parent: add a table to db/schema.sql, add
# it here in the same commit.
TABLES = ("reproducibility_checks", "erasure_manifests", "erasure_jobs",
         "eval_results", "disagreement_reviews",
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
    from db.conn import connect, reset_pool
    from inference.batched_ensemble import clear_cache
    try:
        conn = connect()
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    # The gateway's pool caches connections across cases; one holding a
    # snapshot from before this TRUNCATE would serve stale rows.
    reset_pool()
    # Same hazard in the serving layer: the ensemble cache is keyed on
    # checkpoint hashes, and successive tests build fresh corpora in fresh
    # tmp_paths that can produce identical weights -- so a key can legitimately
    # repeat while pointing at files that no longer exist.
    clear_cache()
    with conn, conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    yield conn
    conn.close()
    reset_pool()
    clear_cache()
