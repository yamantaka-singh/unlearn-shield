"""Runs dashboard/app.py for real via Streamlit's AppTest -- not a manual
browser check that only happens once and is never repeated. This is what
would have caught three bugs found by actually looking at the rendered page
while building this: a UUID/str key mismatch in the certificate selector,
light-pink row highlights with no explicit text color (nearly invisible under
Streamlit's dark theme), and a deadlock where the dashboard's cached
connection held an open transaction across test cases, blocking a later
test's TRUNCATE indefinitely -- fixed at the root in db.conn.connect_readonly
(autocommit=True), not papered over here.
"""

import shutil
from hashlib import sha256

import psycopg2.extras
import pytest
from nacl.signing import SigningKey
from streamlit.testing.v1 import AppTest

from config.settings import CODE_DIGEST, NUM_SHARDS, NUM_SLICES, subject_ref
from engine import rebuild as rebuild_mod
from engine import train as train_mod
from gateway.routes import predict as predict_mod

APP_PATH = "dashboard/app.py"


@pytest.fixture(autouse=True)
def _clear_dashboard_connection_cache():
    """@st.cache_resource is a process-global cache that outlives any single
    AppTest run -- dashboard/app.py's cached connection would otherwise bleed
    from one test into the next. st.cache_resource.clear() is the public API
    for this; importing dashboard.app directly to reach the function would
    re-execute its whole top-level script (real DB queries included) just to
    clear one cache entry."""
    import streamlit as st
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def dashboard_corpus(tmp_path, monkeypatch, pg):
    """A tiny promoted corpus plus one completed erasure with a real
    certificate, so every section of the dashboard has real data to render."""
    shard_dir, ckpt_dir = str(tmp_path / "shards"), str(tmp_path / "ckpt")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(rebuild_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(predict_mod, "CHECKPOINT_DIR", ckpt_dir)

    # A keypair generated here, not whatever this developer's checkout happens
    # to have on disk (`.signing_key` is gitignored -- a fresh clone or CI
    # runner has none). dashboard/app.py always verifies against
    # verify.sign.PUBLIC_KEY_PATH, so that path is pointed at this test's own
    # public half -- the same isolation tests/unit/test_verifier_isolation.py
    # uses, rather than depending on ambient key material.
    key = SigningKey.generate()
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(key).hex())
    # Not PUBLIC_KEY_PATH: verify.sign.load_public_key's default argument was
    # already bound to the real path at module-import time (Python evaluates
    # default values once, at def, not per call), so patching the constant
    # after the fact would silently do nothing. dashboard/app.py re-imports
    # the *name* `load_public_key` from verify.sign fresh on every script
    # rerun, though, so replacing the function itself is what actually reaches it.
    import verify.sign as sign_mod
    monkeypatch.setattr(sign_mod, "load_public_key", lambda path=None: key.verify_key)

    routing = train_mod.build(n_subjects=120, seed=17)
    for k in range(NUM_SHARDS):
        train_mod.train_shard(k)

    with pg, pg.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO subject_shard_map (subject_ref, tenant_id, shard, min_slice_idx, record_count)
            VALUES %s
        """, [(ref, e["tenant_id"], e["shard"], e["min_slice_idx"], e["record_count"])
              for ref, e in routing.items()])

        shard_checkpoints = {}
        for shard in range(NUM_SHARDS):
            path = train_mod.checkpoint_path(shard, NUM_SLICES - 1)
            with open(path, "rb") as f:
                digest = sha256(f.read()).hexdigest()
            cas_path = str(tmp_path / f"cas_{shard}.pt")
            shutil.copyfile(path, cas_path)
            cur.execute("""
                INSERT INTO checkpoints (checkpoint_hash, shard, slice_idx, file_path, code_digest)
                VALUES (%s,%s,%s,%s,%s)
            """, (digest, shard, NUM_SLICES - 1, cas_path, CODE_DIGEST))
            shard_checkpoints[str(shard)] = digest
        cur.execute("""
            INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
            VALUES ('v0-baseline', %s, 'v0')
        """, (psycopg2.extras.Json(shard_checkpoints),))

    from worker.jobs import record_eval
    with pg, pg.cursor() as cur:
        record_eval(cur, "v0-baseline", shard_checkpoints)

    target = next(s for s in (f"C{i:07d}" for i in range(120)) if subject_ref(s) in routing)
    result = rebuild_mod.rebuild(target)
    manifest = result["manifest"]
    with pg, pg.cursor() as cur:
        cur.execute("""
            INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard, idempotency_key,
                                      sla_deadline, requested_by, status, completed_at)
            VALUES (gen_random_uuid(), %s, 'fraud_excision', %s, 'dash-test',
                    now() + interval '720 hours', 'test', 'done', now())
            RETURNING erasure_id
        """, (subject_ref(target), manifest["shard"]))
        erasure_id = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erasure_manifests
                (erasure_id, shard, resumed_from, dataset_root, absence_proof, code_digest,
                 config_digest, result_weights, model_version, signature, manifest_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (erasure_id, manifest["shard"], manifest["resumed_from"], manifest["dataset_root"],
              psycopg2.extras.Json(manifest["absence_proof"]), manifest["code_digest"],
              manifest["config_digest"], manifest["result_weights"], manifest["model_version"],
              manifest["signature"], psycopg2.extras.Json(manifest)))

    return str(erasure_id)


def test_dashboard_renders_with_no_exception(dashboard_corpus, pg, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]


def test_certificate_selector_lists_the_completed_erasure(dashboard_corpus, pg, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert len(at.selectbox) >= 1
    assert dashboard_corpus in at.selectbox[0].options[0]


def test_certificate_live_verification_shows_verified(dashboard_corpus, pg, monkeypatch):
    """Renders the actual VERIFIED/REJECTED badge from a real
    verify_certificate() call against a real certificate, not a stub."""
    from config import settings
    monkeypatch.setattr(settings, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    body = "\n".join(b.value for b in at.success) + "\n".join(e.value for e in at.error)
    assert "VERIFIED" in body


def test_accuracy_metric_shows_a_real_number(dashboard_corpus, pg, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert len(at.metric) >= 1
    assert "AUC" in at.metric[0].value


def test_force_rebuild_rejects_empty_subject_id(dashboard_corpus, pg, monkeypatch):
    """No network call should happen for an empty subject id -- this is a
    client-side guard, not a round trip to a gateway that may not be running."""
    from config import settings
    monkeypatch.setattr(settings, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    at.text_input[0].set_value("")
    # Form submit buttons surface through the plain .button accessor, not a
    # separate one -- confirmed against AppTest directly rather than guessed.
    # Matched by label, not index: at.button[0] is "Refresh", the page's
    # first button, not the form's "Queue erasure" -- an index-based lookup
    # silently clicked the wrong control the first time this test was written.
    submit = next(b for b in at.button if b.label == "Queue erasure")
    submit.click().run(timeout=30)
    assert any("required" in w.value for w in at.warning)
