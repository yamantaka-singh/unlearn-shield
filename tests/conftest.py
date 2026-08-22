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


def override_shard_dir(monkeypatch, shard_dir: str) -> None:
    """Every module that does `from config.settings import SHARD_DIR` gets its
    own frozen copy at import time -- patching `config.settings.SHARD_DIR`
    alone changes nothing for a module that already imported the name.
    `monkeypatch.setattr(engine.train, "SHARD_DIR", ...)` alone silently misses
    the other four; a fixture built that way trains or reads real data from
    whatever is left over in the last-used directory, with no error, on
    whichever module the author forgot.

    This has caused three separate real confusions in one session of manual
    validation work: an empty-shard crash whose repro only patched
    engine.train, a "348 rows" false alarm two shells later, and a booster
    silently trained against stale demo data while being scored against real
    PaySim -- because `gbdt.build()`'s internal `load_shard()` reads
    engine.train's copy of SHARD_DIR, not engine.gbdt's, and only the latter
    had been patched. All three were caught, none were fast to find. This
    helper exists so nobody has to rediscover the module list by trial and
    error again -- keep it in sync with `grep -rn "SHARD_DIR" --include=*.py`.
    """
    import engine.gbdt as gbdt_mod
    import engine.rebuild as rebuild_mod
    import engine.train as train_mod
    for mod in (train_mod, gbdt_mod, rebuild_mod):
        monkeypatch.setattr(mod, "SHARD_DIR", shard_dir)


def override_checkpoint_dir(monkeypatch, checkpoint_dir: str) -> None:
    """The CHECKPOINT_DIR counterpart to override_shard_dir, and it exists for
    the same reason -- but note the module list is DIFFERENT, which is exactly
    why one combined helper would be wrong.

    `engine.active` was added when the GBDT engine was wired into the gateway
    and worker, and it holds its own CHECKPOINT_DIR for the CAS directory and
    the MLP's preprocessor paths. It took over that binding from
    `gateway.routes.predict`, which no longer has one at all -- so every
    fixture that patched `predict_mod.CHECKPOINT_DIR` broke loudly with an
    AttributeError the moment the wiring landed. Loud is the good case; the
    bad case is the silent one this module keeps producing, where a stale
    binding reads real files and nothing errors.

    Keep in sync with `grep -rn "CHECKPOINT_DIR" --include=*.py engine/ gateway/ worker/`.
    """
    import engine.active as active_mod
    import engine.train as train_mod
    for mod in (train_mod, active_mod):
        monkeypatch.setattr(mod, "CHECKPOINT_DIR", checkpoint_dir)


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
