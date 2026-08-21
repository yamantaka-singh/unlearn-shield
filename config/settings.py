"""Env-driven config. One place, no defaults that differ between dev and prod."""

import hmac
import json
import os
from hashlib import sha256

SEED = int(os.environ.get("UNLEARNSHIELD_SEED", "1337"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://unlearnshield:unlearnshield@localhost:55432/unlearnshield")

# Identifies the exact image weights were produced by. Determinism is only
# asserted within one code_digest -- see config/determinism.py.
CODE_DIGEST = os.environ.get("CODE_DIGEST", "dev-unpinned")

# Not a secret in this repo, and deliberately so: a checked-in default that
# looks like a key is worse than one that announces it isn't. Production sets
# this from a secrets manager. Rotating it re-keys every subject_ref, which
# invalidates the routing table -- treat it as permanent per tenant.
TENANT_KEY = os.environ.get("TENANT_KEY", "dev-only-not-a-secret").encode()
TENANT_ID = os.environ.get("TENANT_ID", "dev")

NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "5"))
HOT_SHARDS = int(os.environ.get("HOT_SHARDS", "2"))
HOT_THRESHOLD = float(os.environ.get("HOT_THRESHOLD", "0.6"))
NUM_SLICES = int(os.environ.get("NUM_SLICES", "5"))

EPOCHS_PER_SLICE = int(os.environ.get("EPOCHS_PER_SLICE", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-3"))

SHARD_DIR = os.environ.get("SHARD_DIR", "data/shards")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")


def subject_ref(subject_id: str) -> str:
    """The only form a subject identifier takes past ingest.

    HMAC rather than a bare hash: a bare sha256 of an account number is
    trivially reversible by enumeration, and account numbers are a small space.
    """
    return hmac.new(TENANT_KEY, subject_id.encode(), sha256).hexdigest()


# Phase 4
SLA_HOURS = int(os.environ.get("SLA_HOURS", "720"))          # 30 days, GDPR-shaped default
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", "1800"))  # generous: a rebuild runs minutes
POLL_BATCH_SIZE = int(os.environ.get("POLL_BATCH_SIZE", "20"))

# Not a secret in this repo, same reasoning as TENANT_KEY. Selects which
# completed jobs the worker re-runs for the reproducibility spot-check
# (Phase 4c) -- HMAC(erasure_id, audit_key) rather than the worker choosing,
# so an operator cannot steer the sample away from jobs it would rather not
# have re-run.
AUDIT_KEY = os.environ.get("AUDIT_KEY", "dev-only-not-a-secret").encode()
SPOT_CHECK_RATE = float(os.environ.get("SPOT_CHECK_RATE", "0.01"))

# token -> {"principal": str, "scopes": {"predict:invoke", ...}}. A static
# service-to-service map, not a user-account system: erasure:write callers are
# a consent manager or an internal job runner, not end users hitting this API
# directly, and predict:invoke callers are the model-serving consumers.
def auth_tokens() -> dict:
    raw = os.environ.get("AUTH_TOKENS")
    if not raw:
        return {"dev-token": {"principal": "dev", "scopes": {
            "predict:invoke", "erasure:write", "erasure:attest"}}}
    parsed = json.loads(raw)
    return {tok: {"principal": v["principal"], "scopes": set(v["scopes"])}
            for tok, v in parsed.items()}
