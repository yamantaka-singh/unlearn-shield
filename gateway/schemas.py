"""Request/response shapes. Subject identifiers live only in request bodies --
never in a path or query string, which is what keeps them out of ingress logs,
APM traces, and CDN logs (see ADR discussion in docs/plan-corrections.md #7)."""

from typing import Literal

from pydantic import BaseModel, Field

from data.synth import TYPES


class ErasureRequest(BaseModel):
    subject_id: str
    reason: Literal["consent_revocation", "fraud_excision"]


class ErasureAccepted(BaseModel):
    """Deliberately omits `shard`: it is derived from churn_score, so returning
    it would disclose a coarse behavioural signal about the subject to whatever
    logs this response (correction #7)."""
    erasure_id: str
    status: Literal["queued"]
    sla_deadline: str


class ErasureStatus(BaseModel):
    erasure_id: str
    status: Literal["queued", "processing", "done", "failed"]
    reason: str
    created_at: str
    completed_at: str | None
    last_error: str | None


class AttestRequest(BaseModel):
    subject_id: str


class PredictRequest(BaseModel):
    step: float
    type: Literal[TYPES]
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class PredictResponse(BaseModel):
    fraud_probability: float
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    shard_checkpoints: dict
    eval_set_version: str
    promoted_at: str
