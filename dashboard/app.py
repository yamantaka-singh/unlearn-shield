"""Ops console. Internal tool, not the product.

Replaces the original Streamlit app. Streamlit gave live data quickly but very
little control over layout, density, and colour -- and an ops console is mostly
dense tables and status, which is exactly what it is worst at. This serves a
hand-written page instead: full control, no build step, no bundler, and it
drops thirteen streamlit-family packages from the image.

The security boundary from ADR 0008 is unchanged and is the reason this is a
separate process rather than a route on the gateway: it connects through
`unlearnshield_readonly`, a role granted SELECT only, so "the dashboard never
writes to Postgres" is enforced by the database rather than by convention. The
one write path -- force a rebuild -- is proxied to the gateway's own
POST /v1/erasure exactly as any other caller would call it.

    uvicorn dashboard.app:app --port 8501
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import DASHBOARD_GATEWAY_TOKEN, DASHBOARD_GATEWAY_URL
from db.conn import connect_readonly

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="UnlearnShield Ops", docs_url=None, redoc_url=None)
api = APIRouter(prefix="/api")


def rows(sql: str, params: tuple = ()) -> list[dict]:
    """One short-lived read. The connection is autocommit (db/conn.py), so a
    long-lived console never sits on an open snapshot."""
    conn = connect_readonly()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


@api.get("/overview")
def overview() -> dict:
    counts = {r["status"]: r["n"] for r in
              rows("SELECT status, count(*) AS n FROM erasure_jobs GROUP BY status")}
    by_shard = rows("""
        SELECT shard, status, count(*) AS n FROM erasure_jobs
        GROUP BY shard, status ORDER BY shard
    """)
    current = rows("""
        SELECT m.model_version, m.promoted_at, e.auc, e.n_eval
        FROM model_versions m LEFT JOIN eval_results e USING (model_version)
        ORDER BY m.promoted_at DESC LIMIT 1
    """)
    history = rows("""
        SELECT model_version, auc, n_eval, computed_at
        FROM eval_results ORDER BY computed_at
    """)
    drift = rows("SELECT count(*) AS n FROM reproducibility_checks WHERE NOT matched")

    return {
        "counts": {k: counts.get(k, 0)
                   for k in ("queued", "processing", "done", "failed")},
        "by_shard": [{**r} for r in by_shard],
        "current_model": ({**current[0], "promoted_at": _iso(current[0]["promoted_at"])}
                          if current else None),
        "eval_history": [{**r, "computed_at": _iso(r["computed_at"])} for r in history],
        "drift_failures": drift[0]["n"] if drift else 0,
        "subjects_routed": rows("SELECT count(*) AS n FROM subject_shard_map")[0]["n"],
    }


@api.get("/pending")
def pending() -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for r in rows("""
        SELECT erasure_id, subject_ref, shard, reason, status, attempts,
               last_error, sla_deadline, created_at
        FROM erasure_jobs WHERE status IN ('queued', 'processing')
        ORDER BY sla_deadline ASC LIMIT 100
    """):
        out.append({**r,
                    "erasure_id": str(r["erasure_id"]),
                    "hours_remaining": round((r["sla_deadline"] - now).total_seconds() / 3600, 1),
                    "sla_deadline": _iso(r["sla_deadline"]),
                    "created_at": _iso(r["created_at"])})
    return out


@api.get("/failed")
def failed() -> list[dict]:
    return [{**r, "erasure_id": str(r["erasure_id"]), "completed_at": _iso(r["completed_at"])}
            for r in rows("""
                SELECT erasure_id, subject_ref, shard, reason, attempts, last_error, completed_at
                FROM erasure_jobs WHERE status = 'failed'
                ORDER BY completed_at DESC NULLS LAST LIMIT 50
            """)]


@api.get("/certificates")
def certificates() -> list[dict]:
    return [{**r, "erasure_id": str(r["erasure_id"]), "created_at": _iso(r["created_at"])}
            for r in rows("""
                SELECT erasure_id, shard, model_version, created_at
                FROM erasure_manifests ORDER BY created_at DESC LIMIT 200
            """)]


@api.get("/certificates/{erasure_id}")
def certificate(erasure_id: str) -> dict:
    """Returns the manifest AND a live re-verification of it.

    Re-verified on every request rather than cached: the whole point of
    shipping a certificate is that anyone can check it independently, and a
    stored "it passed once" flag is an assertion, not a proof.
    """
    found = rows("SELECT manifest_json FROM erasure_manifests WHERE erasure_id = %s",
                 (erasure_id,))
    if not found:
        raise HTTPException(404, "no certificate for that erasure_id")
    manifest = found[0]["manifest_json"]

    from verify.sign import load_public_key
    from verify.verifier_cli import verify_certificate
    ok, findings = verify_certificate(dict(manifest), load_public_key())
    return {"manifest": manifest, "verified": ok, "findings": findings}


class RebuildRequest(BaseModel):
    subject_id: str
    reason: str


@api.post("/rebuild")
def force_rebuild(body: RebuildRequest) -> dict:
    """Proxied to the gateway's own POST /v1/erasure -- the same path any other
    caller uses. This process holds no write credentials for Postgres and could
    not insert the row itself even if this code tried to."""
    if not body.subject_id.strip():
        raise HTTPException(400, "subject_id is required")
    payload = json.dumps({"subject_id": body.subject_id, "reason": body.reason}).encode()
    request = urllib.request.Request(
        f"{DASHBOARD_GATEWAY_URL}/v1/erasure", data=payload, method="POST",
        headers={"Authorization": f"Bearer {DASHBOARD_GATEWAY_TOKEN}",
                 "Idempotency-Key": f"ops-{body.subject_id}-{datetime.now(timezone.utc).timestamp()}",
                 "content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return {"ok": True, "response": json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, f"gateway rejected the request: {e.read().decode()[:400]}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"could not reach the gateway at {DASHBOARD_GATEWAY_URL}: {e.reason}")


app.include_router(api)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
