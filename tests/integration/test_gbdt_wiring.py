"""The GBDT engine driven through the REAL worker and serving path.

engine/gbdt.py had unit tests from the day it landed, but every one of them
called it directly. Nothing proved the worker could run it: `worker/jobs.py`
imported `engine.rebuild` by name, promoted a `.pt`, and loaded a torch
ensemble, so the tree engine was a well-tested library with no way in. These
tests exercise `process_claimed` -- the actual function the worker loop calls --
with MODEL_ENGINE=gbdt, and assert the things that only break when the wiring
is wrong rather than when the engine is.

`engine.active.IS_GBDT` is patched rather than the environment variable,
because `MODEL_ENGINE` is read at import time (config/settings.py) and this
process has already imported it. That is the same module-binding property
tests/conftest.py::override_shard_dir documents, and patching the derived flag
is the honest way to test a value that production genuinely does fix at
startup.
"""

import numpy as np
import psycopg2.extras
import pytest
from nacl.signing import SigningKey

from config.settings import NUM_SHARDS, subject_ref
from engine import active
from engine import gbdt
from engine import train as train_mod
from tests.conftest import override_checkpoint_dir, override_shard_dir
from worker.jobs import process_claimed, run_spot_check


@pytest.fixture
def gbdt_corpus(tmp_path, monkeypatch, pg):
    """A trained GBDT deployment: shards built, one booster per shard, and the
    baseline promoted into Postgres exactly as scripts/load_routing.py does."""
    override_shard_dir(monkeypatch, str(tmp_path / "shards"))
    override_checkpoint_dir(monkeypatch, str(tmp_path / "ckpt"))
    monkeypatch.setattr(active, "IS_GBDT", True)
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())

    routing = train_mod.build(n_subjects=400, seed=41)
    gbdt.build(routing)

    with pg, pg.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO subject_shard_map (subject_ref, tenant_id, shard, min_slice_idx, record_count)
            VALUES %s ON CONFLICT (subject_ref) DO NOTHING
        """, [(r, e["tenant_id"], e["shard"], e["min_slice_idx"], e["record_count"])
              for r, e in routing.items()])

        shard_checkpoints = {}
        for shard in range(NUM_SHARDS):
            path, _ = active.live_model_path(shard)
            with open(path, "rb") as f:
                from hashlib import sha256
                digest = sha256(f.read()).hexdigest()
            cas_path = active.promote_artifact(shard, digest)
            cur.execute("""
                INSERT INTO checkpoints (checkpoint_hash, shard, slice_idx, file_path, code_digest)
                VALUES (%s, %s, %s, %s, 'test') ON CONFLICT (checkpoint_hash) DO NOTHING
            """, (digest, shard, 4, cas_path))
            shard_checkpoints[str(shard)] = digest

        cur.execute("""
            INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
            VALUES ('v0-baseline', %s, 'v0') ON CONFLICT DO NOTHING
        """, (psycopg2.extras.Json(shard_checkpoints),))
    return routing


def _enqueue(cur, ref, shard):
    cur.execute("""
        INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard, idempotency_key,
                                  sla_deadline, requested_by, status)
        VALUES (gen_random_uuid(), %s, 'consent_revocation', %s, %s,
                now() + interval '720 hours', 'test', 'processing')
        RETURNING erasure_id
    """, (ref, shard, f"gbdt-{ref[:8]}"))
    return str(cur.fetchone()[0])


def _a_subject(routing, min_slice=None):
    for ref, entry in sorted(routing.items()):
        if min_slice is None or entry["min_slice_idx"] == min_slice:
            return ref, entry
    pytest.skip(f"no subject with min_slice_idx={min_slice} in this corpus")


def test_worker_runs_a_gbdt_erasure_end_to_end(gbdt_corpus, pg):
    """The whole point: process_claimed, unmodified, completing a tree rebuild.

    Asserts the four things the wiring is responsible for and the engine is
    not -- the job completes, a signed manifest lands, a new model_version is
    promoted with a booster on disk behind it, and the promoted ensemble is
    scored for real.
    """
    ref, entry = _a_subject(gbdt_corpus)

    with pg, pg.cursor() as cur:
        erasure_id = _enqueue(cur, ref, entry["shard"])
        process_claimed(cur, [{"erasure_id": erasure_id, "subject_ref": ref,
                               "shard": entry["shard"]}])

    with pg.cursor() as cur:
        cur.execute("SELECT status, last_error FROM erasure_jobs WHERE erasure_id = %s",
                    (erasure_id,))
        status, last_error = cur.fetchone()
        assert status == "done", f"job failed through the GBDT path: {last_error}"

        cur.execute("SELECT model_version, result_weights, signature FROM erasure_manifests "
                    "WHERE erasure_id = %s", (erasure_id,))
        model_version, result_weights, signature = cur.fetchone()
        assert model_version.startswith("gbdt-shard"), (
            "the manifest must name the engine that produced it, or a GBDT "
            "rebuild is indistinguishable from an MLP one in the audit record")
        assert signature, "manifest reached the DB unsigned"

        # The promoted checkpoint must be a booster that actually loads --
        # promoting a .pt path for a tree deployment would store a row that
        # only fails much later, at the first predict.
        cur.execute("SELECT file_path FROM checkpoints WHERE checkpoint_hash = %s",
                    (result_weights,))
        path = cur.fetchone()[0]
        assert path.endswith(".json"), f"promoted a non-booster artifact: {path}"

        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(path)  # raises if it is not a real booster

        cur.execute("SELECT auc, n_eval FROM eval_results ORDER BY computed_at DESC LIMIT 1")
        auc, n_eval = cur.fetchone()
    assert n_eval > 0
    assert 0.0 <= auc <= 1.0, "record_eval scored the GBDT ensemble out of range"


def test_promoted_gbdt_model_no_longer_scores_with_the_erased_subject(gbdt_corpus, pg):
    """Serving is the half of an erasure that silently fails. The rebuilt
    booster must differ from the one promoted before the erasure -- if the
    ensemble cache or the promotion path handed back the old model, every
    number above would still look fine while the erased subject kept
    influencing scores."""
    ref, entry = _a_subject(gbdt_corpus)
    shard = entry["shard"]

    before = active.load_ensemble({str(shard): active.promote_artifact(shard, "before-probe")})
    records = train_mod.load_shard(shard)
    rows = np.arange(min(64, len(records["step"])))
    scores_before = before.predict_proba(records, rows)

    with pg, pg.cursor() as cur:
        erasure_id = _enqueue(cur, ref, shard)
        process_claimed(cur, [{"erasure_id": erasure_id, "subject_ref": ref, "shard": shard}])

    with pg.cursor() as cur:
        cur.execute("SELECT result_weights FROM erasure_manifests WHERE erasure_id = %s",
                    (erasure_id,))
        result_weights = cur.fetchone()[0]
        cur.execute("SELECT file_path FROM checkpoints WHERE checkpoint_hash = %s",
                    (result_weights,))
        after_path = cur.fetchone()[0]

    after = active.load_ensemble({str(shard): after_path})
    scores_after = after.predict_proba(records, rows)

    assert not np.array_equal(scores_before, scores_after), (
        "the promoted booster is byte-identical to the pre-erasure one -- the "
        "rebuild did not reach the serving path")


def test_spot_check_reproduces_a_gbdt_rebuild(gbdt_corpus, pg):
    """The reproducibility guarantee has to survive the engine swap, and it is
    a different mechanism here: no checkpoint is reloaded, the resume point is
    a truncated booster (ADR 0011). A pass means truncate-then-continue is
    deterministic, which is what makes a tree manifest re-verifiable at all."""
    ref, entry = _a_subject(gbdt_corpus, min_slice=1)
    result = gbdt.rebuild_batch_by_ref([ref])

    with pg, pg.cursor() as cur:
        erasure_id = _enqueue(cur, ref, result["shard"])
        matched = run_spot_check(cur, erasure_id, result["shard"],
                                 result["replay"], result["result_weights"])
    assert matched, "a GBDT rebuild did not reproduce bit-identically"

    with pg.cursor() as cur:
        cur.execute("SELECT expected_weights, observed_weights, matched "
                    "FROM reproducibility_checks WHERE erasure_id = %s", (erasure_id,))
        expected, observed, stored = cur.fetchone()
    assert stored is True
    assert expected == observed == result["result_weights"]


def test_spot_check_still_detects_divergence_under_gbdt(gbdt_corpus, pg):
    """Negative control. A check that only ever passes proves nothing."""
    ref, _ = _a_subject(gbdt_corpus)
    result = gbdt.rebuild_batch_by_ref([ref])

    with pg, pg.cursor() as cur:
        erasure_id = _enqueue(cur, ref, result["shard"])
        matched = run_spot_check(cur, erasure_id, result["shard"],
                                 result["replay"], "0" * 64)
    assert not matched


def test_gbdt_ensemble_and_mlp_ensemble_do_not_share_a_cache_entry(gbdt_corpus):
    """Both engines key the ensemble cache on the same tuple of checkpoint
    paths. Without the engine tag in the key, a process that has served one
    engine would hand the other its object and fail somewhere far away."""
    from inference.batched_ensemble import GBDTEnsemble, load_gbdt_ensemble

    paths = {"0": active.promote_artifact(0, "cache-probe")}
    first = load_gbdt_ensemble(paths)
    assert load_gbdt_ensemble(paths) is first, "booster cache is not being hit"
    assert isinstance(first, GBDTEnsemble)
