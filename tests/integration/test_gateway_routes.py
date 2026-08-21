"""Route contracts: 202 shape, auth enforcement, 404s. Predict and the full
lifecycle need real model checkpoints and live in tests/e2e/ instead --
these only need subject_shard_map and model_versions rows.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

HEADERS = {"Authorization": "Bearer dev-token"}


@pytest.fixture
def client(pg):
    from gateway.main import app
    return TestClient(app)


def _route_subject(pg, ref: str, shard: int = 0) -> None:
    with pg, pg.cursor() as cur:
        cur.execute("""
            INSERT INTO subject_shard_map (subject_ref, tenant_id, shard, min_slice_idx, record_count)
            VALUES (%s, 'dev', %s, 0, 1)
        """, (ref, shard))


def test_erasure_returns_202_with_correct_shape(client, pg):
    from config.settings import subject_ref
    _route_subject(pg, subject_ref("C0000001"))
    resp = client.post("/v1/erasure", json={"subject_id": "C0000001", "reason": "fraud_excision"},
                       headers={**HEADERS, "Idempotency-Key": "route-1"})
    assert resp.status_code == 202
    body = resp.json()
    assert set(body) == {"erasure_id", "status", "sla_deadline"}, (
        "shard must never appear here -- it is derived from churn_score and "
        "would disclose a behavioural signal to whatever logs this response")
    assert body["status"] == "queued"


def test_erasure_requires_idempotency_key_header(client, pg):
    from config.settings import subject_ref
    _route_subject(pg, subject_ref("C0000002"))
    resp = client.post("/v1/erasure", json={"subject_id": "C0000002", "reason": "fraud_excision"},
                       headers=HEADERS)
    assert resp.status_code == 422


def test_erasure_unknown_subject_is_404(client, pg):
    resp = client.post("/v1/erasure", json={"subject_id": "C9999999", "reason": "fraud_excision"},
                       headers={**HEADERS, "Idempotency-Key": "route-unknown"})
    assert resp.status_code == 404


def test_erasure_status_unknown_id_is_404(client, pg):
    import uuid
    resp = client.get(f"/v1/erasure/{uuid.uuid4()}", headers=HEADERS)
    assert resp.status_code == 404


@pytest.mark.parametrize("headers,expected", [
    ({}, 401),
    ({"Authorization": "Bearer not-a-real-token"}, 401),
])
def test_missing_or_invalid_token_is_rejected(client, pg, headers, expected):
    resp = client.get("/v1/models/current", headers=headers)
    assert resp.status_code == expected


def test_subject_id_never_appears_in_a_url(client, pg):
    """The plan's own requirement: subject identifiers live in bodies, never
    paths or query strings, so they never land in ingress/APM/CDN logs."""
    from fastapi.routing import APIRoute
    from gateway.main import app
    for route in app.routes:
        if isinstance(route, APIRoute):
            assert "subject" not in route.path.lower()


def test_models_current_404_when_nothing_promoted(client, pg):
    resp = client.get("/v1/models/current", headers=HEADERS)
    assert resp.status_code == 404


def test_certificate_404_before_job_completes(client, pg):
    import uuid
    resp = client.get(f"/v1/erasure/{uuid.uuid4()}/certificate", headers=HEADERS)
    assert resp.status_code == 404


def test_attest_404_for_a_subject_with_no_completed_erasure(client, pg):
    resp = client.post("/v1/erasure/attest", json={"subject_id": "C0000003"}, headers=HEADERS)
    assert resp.status_code == 404
