"""The ops console (dashboard/app.py), against real Postgres.

Replaced the Streamlit AppTest suite when the dashboard moved to FastAPI plus
a hand-written page. The properties worth guarding did not change: it reads
real data, it re-verifies certificates live rather than trusting a stored
flag, and it cannot write to Postgres.
"""

import shutil
from hashlib import sha256

import psycopg2.extras
import pytest
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from config.settings import CODE_DIGEST, NUM_SHARDS, NUM_SLICES, subject_ref
from engine import rebuild as rebuild_mod
from engine import train as train_mod
from gateway.routes import predict as predict_mod


@pytest.fixture
def client(pg, monkeypatch):
    """The dashboard normally connects as unlearnshield_readonly. Tests point
    it at the same URL the `pg` fixture truncates, so a case sees its own
    fixture data rather than whatever the read-only role can see."""
    from config import settings
    monkeypatch.setattr(settings, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    import db.conn
    monkeypatch.setattr(db.conn, "DASHBOARD_DATABASE_URL", settings.DATABASE_URL)
    from dashboard.app import app
    return TestClient(app)


@pytest.fixture
def corpus(tmp_path, monkeypatch, pg):
    """A promoted model plus one completed erasure with a real certificate."""
    shard_dir, ckpt_dir = str(tmp_path / "shards"), str(tmp_path / "ckpt")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(rebuild_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(predict_mod, "CHECKPOINT_DIR", ckpt_dir)

    key = SigningKey.generate()
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(key).hex())
    import verify.sign as sign_mod
    monkeypatch.setattr(sign_mod, "load_public_key", lambda path=None: key.verify_key)

    routing = train_mod.build(n_subjects=120, seed=53)
    for k in range(NUM_SHARDS):
        train_mod.train_shard(k)

    with pg, pg.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO subject_shard_map (subject_ref, tenant_id, shard, min_slice_idx, record_count)
            VALUES %s
        """, [(r, e["tenant_id"], e["shard"], e["min_slice_idx"], e["record_count"])
              for r, e in routing.items()])

        shard_checkpoints = {}
        for shard in range(NUM_SHARDS):
            path = train_mod.checkpoint_path(shard, NUM_SLICES - 1)
            with open(path, "rb") as f:
                digest = sha256(f.read()).hexdigest()
            cas = str(tmp_path / f"cas_{shard}.pt")
            shutil.copyfile(path, cas)
            cur.execute("""
                INSERT INTO checkpoints (checkpoint_hash, shard, slice_idx, file_path, code_digest)
                VALUES (%s,%s,%s,%s,%s)
            """, (digest, shard, NUM_SLICES - 1, cas, CODE_DIGEST))
            shard_checkpoints[str(shard)] = digest
        cur.execute("""
            INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
            VALUES ('v0-baseline', %s, 'v0')
        """, (psycopg2.extras.Json(shard_checkpoints),))

    from worker.jobs import record_eval
    with pg, pg.cursor() as cur:
        record_eval(cur, "v0-baseline", shard_checkpoints)

    target = next(s for s in (f"C{i:07d}" for i in range(120)) if subject_ref(s) in routing)
    manifest = rebuild_mod.rebuild(target)["manifest"]
    with pg, pg.cursor() as cur:
        cur.execute("""
            INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard, idempotency_key,
                                      sla_deadline, requested_by, status, completed_at)
            VALUES (gen_random_uuid(), %s, 'fraud_excision', %s, 'dash-done',
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
        # One pending job so the SLA view has something to render.
        cur.execute("""
            INSERT INTO erasure_jobs (erasure_id, subject_ref, reason, shard, idempotency_key,
                                      sla_deadline, requested_by)
            VALUES (gen_random_uuid(), 'pending-ref', 'consent_revocation', 1, 'dash-pending',
                    now() + interval '5 hours', 'test')
        """)
    return str(erasure_id)


def test_index_serves_the_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "UnlearnShield" in r.text
    assert "/static/app.js" in r.text


def test_overview_reports_real_counts(corpus, client):
    d = client.get("/api/overview").json()
    assert d["counts"]["done"] == 1
    assert d["counts"]["queued"] == 1
    assert d["subjects_routed"] > 0
    assert d["current_model"]["model_version"] == "v0-baseline"
    assert 0.0 <= d["current_model"]["auc"] <= 1.0
    assert d["drift_failures"] == 0


def test_pending_computes_hours_remaining(corpus, client):
    rows = client.get("/api/pending").json()
    assert len(rows) == 1
    assert rows[0]["reason"] == "consent_revocation"
    # Seeded at +5 hours; allow slack for test runtime.
    assert 4.0 < rows[0]["hours_remaining"] < 5.1


def test_certificate_is_reverified_live_not_cached(corpus, client):
    """The point of shipping a certificate is that anyone can check it
    independently. A stored pass/fail flag would be an assertion, not a proof
    -- so this endpoint runs verify_certificate on every request."""
    d = client.get(f"/api/certificates/{corpus}").json()
    assert d["verified"] is True
    assert any("signature valid" in f for f in d["findings"])
    assert d["manifest"]["absence_proof"]["scheme"] == "smt-256"


def test_tampered_certificate_is_reported_as_rejected(corpus, client, pg):
    """Corrupt the stored manifest and the console must say REJECTED. If it
    reported VERIFIED from a cache, the tool used to audit erasures would be
    the one thing not auditing them."""
    with pg, pg.cursor() as cur:
        cur.execute("""
            UPDATE erasure_manifests
            SET manifest_json = jsonb_set(manifest_json, '{shard}', '99')
            WHERE erasure_id = %s
        """, (corpus,))

    d = client.get(f"/api/certificates/{corpus}").json()
    assert d["verified"] is False
    assert any("SIGNATURE INVALID" in f for f in d["findings"])


def test_unknown_certificate_is_404(corpus, client):
    import uuid
    assert client.get(f"/api/certificates/{uuid.uuid4()}").status_code == 404


def test_force_rebuild_rejects_an_empty_subject_id(client):
    r = client.post("/api/rebuild", json={"subject_id": "  ", "reason": "fraud_excision"})
    assert r.status_code == 400


def test_force_rebuild_surfaces_an_unreachable_gateway(client, monkeypatch):
    """No silent failure: if the gateway is down, the operator is told, rather
    than the console appearing to have queued something it did not."""
    from config import settings
    import dashboard.app as dash
    monkeypatch.setattr(dash, "DASHBOARD_GATEWAY_URL", "http://127.0.0.1:9")
    r = client.post("/api/rebuild", json={"subject_id": "C0000001", "reason": "fraud_excision"})
    assert r.status_code == 502
    assert "could not reach the gateway" in r.json()["detail"]


def test_dashboard_never_writes_to_postgres():
    """ADR 0008's boundary, still enforced by the database rather than by
    convention: the console's own role is granted SELECT only."""
    import psycopg2
    from db.conn import connect_readonly

    conn = connect_readonly()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM erasure_jobs")  # reads fine
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("""
                    INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
                    VALUES ('nope', '{}', 'v0')
                """)
    finally:
        conn.close()


def test_no_endpoint_returns_a_raw_subject_id(corpus, client):
    """subject_ref is the only form an identifier takes past ingest. A console
    that surfaced raw ids would put them in browser history and screenshots."""
    for path in ("/api/overview", "/api/pending", "/api/failed", "/api/certificates"):
        assert "subject_id" not in client.get(path).text
