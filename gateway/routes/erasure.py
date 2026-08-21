"""Erasure intake and status. Never trains -- enqueues and reads only.

Subject IDs never appear in a path or query string (see gateway/schemas.py).
`POST /erasure` always returns 202: a synchronous design here would block an
HTTP thread on a multi-minute rebuild, which times out under real load.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException

from config.settings import SLA_HOURS, subject_ref
from db.conn import connect
from gateway.auth import require_scope
from gateway.idempotency import insert_or_get
from gateway.schemas import AttestRequest, ErasureAccepted, ErasureRequest, ErasureStatus

router = APIRouter(prefix="/v1/erasure", tags=["erasure"])


@router.post("", response_model=ErasureAccepted, status_code=202)
def create_erasure(body: ErasureRequest,
                   idempotency_key: str = Header(..., alias="Idempotency-Key"),
                   principal: str = Depends(require_scope("erasure:write"))):
    ref = subject_ref(body.subject_id)
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT shard FROM subject_shard_map WHERE subject_ref = %s", (ref,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(404, "unknown subject")
            shard = row[0]

            sla_deadline = datetime.now(timezone.utc) + timedelta(hours=SLA_HOURS)
            erasure_id, _created = insert_or_get(
                cur, subject_ref=ref, reason=body.reason, shard=shard,
                idempotency_key=idempotency_key, sla_deadline=sla_deadline,
                requested_by=principal)
            conn.commit()

            cur.execute("SELECT sla_deadline FROM erasure_jobs WHERE erasure_id = %s", (erasure_id,))
            actual_deadline = cur.fetchone()[0]
    finally:
        conn.close()

    return ErasureAccepted(erasure_id=erasure_id, status="queued",
                           sla_deadline=actual_deadline.isoformat())


@router.get("/{erasure_id}", response_model=ErasureStatus)
def get_erasure(erasure_id: str, principal: str = Depends(require_scope("erasure:attest"))):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT erasure_id, status, reason, created_at, completed_at, last_error
                FROM erasure_jobs WHERE erasure_id = %s
            """, (erasure_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "no such erasure_id")
    return ErasureStatus(erasure_id=str(row[0]), status=row[1], reason=row[2],
                         created_at=row[3].isoformat(),
                         completed_at=row[4].isoformat() if row[4] else None,
                         last_error=row[5])


@router.get("/{erasure_id}/certificate")
def get_certificate(erasure_id: str, principal: str = Depends(require_scope("erasure:attest"))):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT manifest_json FROM erasure_manifests WHERE erasure_id = %s",
                       (erasure_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "no certificate for this erasure_id yet")
    return row[0]


@router.post("/attest")
def attest(body: AttestRequest, principal: str = Depends(require_scope("erasure:attest"))):
    """Auditor-facing: subject in the body, never a path segment, so an auditor
    checking many subjects never writes one into a URL that gets logged."""
    ref = subject_ref(body.subject_id)
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.manifest_json FROM erasure_manifests m
                JOIN erasure_jobs j ON j.erasure_id = m.erasure_id
                WHERE j.subject_ref = %s ORDER BY m.created_at DESC LIMIT 1
            """, (ref,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "no completed erasure found for this subject")
    return row[0]
