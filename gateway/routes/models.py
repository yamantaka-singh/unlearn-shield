"""Read-only model version info."""

from fastapi import APIRouter, Depends, HTTPException

from db.conn import pooled
from gateway.auth import require_scope
from gateway.schemas import ModelInfo

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("/current", response_model=ModelInfo)
def current(principal: str = Depends(require_scope("predict:invoke"))):
    with pooled() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_version, shard_checkpoints, eval_set_version, promoted_at
                FROM model_versions ORDER BY promoted_at DESC LIMIT 1
            """)
            row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "no model promoted yet")
    return ModelInfo(model_version=row[0], shard_checkpoints=row[1],
                     eval_set_version=row[2], promoted_at=row[3].isoformat())


@router.get("/{version}/manifest")
def manifest_for_version(version: str, principal: str = Depends(require_scope("erasure:attest"))):
    with pooled() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT shard_checkpoints FROM model_versions WHERE model_version = %s",
                       (version,))
            row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "unknown model_version")
    return {"model_version": version, "shard_checkpoints": row[0]}
