"""Enqueue -> worker processes -> manifest exists -> verifier confirms absence
-> predict reflects the change -> status reports done.

Everything here runs against a real Postgres (see tests/integration/conftest.py)
and a real tiny corpus built the same way `engine.train --build` builds one --
the only thing standing in for production is scale.
"""

import os
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
from verify.verifier_cli import verify_certificate


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return sha256(f.read()).hexdigest()


@pytest.fixture
def signing_pair(monkeypatch):
    key = SigningKey.generate()
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(key).hex())
    return key.verify_key


@pytest.fixture
def corpus(tmp_path, monkeypatch, pg):
    """A small real corpus, built and trained, then loaded into Postgres --
    same two steps `engine.train --build` + `scripts.load_routing` perform,
    just pointed at a temp directory instead of the repo's real one."""
    shard_dir, ckpt_dir = str(tmp_path / "shards"), str(tmp_path / "ckpt")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(rebuild_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(predict_mod, "CHECKPOINT_DIR", ckpt_dir)

    n_subjects = 150
    routing = train_mod.build(n_subjects=n_subjects, seed=11)
    for k in range(NUM_SHARDS):
        train_mod.train_shard(k)

    with pg, pg.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO subject_shard_map (subject_ref, tenant_id, shard, min_slice_idx, record_count)
            VALUES %s
        """, [(ref, e["tenant_id"], e["shard"], e["min_slice_idx"], e["record_count"])
              for ref, e in routing.items()])

        cas_dir = os.path.join(ckpt_dir, "cas")
        os.makedirs(cas_dir, exist_ok=True)
        shard_checkpoints = {}
        for shard in range(NUM_SHARDS):
            path = train_mod.checkpoint_path(shard, NUM_SLICES - 1)
            digest = _file_hash(path)
            cas_path = os.path.join(cas_dir, f"{digest}.pt")
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

    return routing, n_subjects


def _find_raw_id(routing: dict, n_subjects: int) -> tuple[str, str]:
    """subject_ref is HMAC, so the only way from a ref back to a raw id is to
    check the known candidates -- there is no inverse function, by design."""
    target_ref = next(iter(routing))
    for i in range(n_subjects):
        candidate = f"C{i:07d}"
        if subject_ref(candidate) == target_ref:
            return candidate, target_ref
    raise AssertionError("no candidate subject_id matched a routed ref")


def test_full_erasure_lifecycle(corpus, pg, signing_pair):
    routing, n_subjects = corpus
    target_id, target_ref = _find_raw_id(routing, n_subjects)
    headers = {"Authorization": "Bearer dev-token"}
    predict_body = {"step": 10, "type": "TRANSFER", "amount": 500, "oldbalanceOrg": 500,
                    "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 500}

    from gateway.main import app
    client = TestClient(app)

    before = client.post("/v1/predict", json=predict_body, headers=headers)
    assert before.status_code == 200
    version_before = before.json()["model_version"]

    resp = client.post("/v1/erasure", json={"subject_id": target_id, "reason": "fraud_excision"},
                       headers={**headers, "Idempotency-Key": "e2e-key"})
    assert resp.status_code == 202
    body = resp.json()
    assert set(body) == {"erasure_id", "status", "sla_deadline"}
    assert body["status"] == "queued"
    erasure_id = body["erasure_id"]

    assert client.get(f"/v1/erasure/{erasure_id}", headers=headers).json()["status"] == "queued"

    from worker.jobs import process_claimed
    from worker.queue import claim_batch

    with pg, pg.cursor() as cur:
        jobs = claim_batch(cur, "test-worker")
    assert len(jobs) == 1
    with pg, pg.cursor() as cur:
        process_claimed(cur, jobs)

    assert client.get(f"/v1/erasure/{erasure_id}", headers=headers).json()["status"] == "done"

    cert = client.get(f"/v1/erasure/{erasure_id}/certificate", headers=headers).json()
    ok, findings = verify_certificate(dict(cert), signing_pair)
    assert ok, findings
    assert cert["subject_ref"] == target_ref

    attest = client.post("/v1/erasure/attest", json={"subject_id": target_id}, headers=headers)
    assert attest.status_code == 200
    assert attest.json()["subject_ref"] == target_ref

    after = client.post("/v1/predict", json=predict_body, headers=headers)
    assert after.status_code == 200
    assert after.json()["model_version"] != version_before
