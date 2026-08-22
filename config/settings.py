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

# Which model this deployment trains, serves, and rebuilds: "mlp" or "gbdt".
#
# A deployment-wide switch, not a per-request one. ADR 0011 established that
# the two engines are alternatives rather than a hybrid, and the reason is
# concrete rather than stylistic: both pop rows from the same routing.json and
# purge the same shard files, so running them together against one SHARD_DIR
# would have each engine's rebuild delete the other's routing entry. Anything
# that let both answer at once would need two routing tables and two shard
# directories -- a real design decision, deliberately not made here.
MODEL_ENGINE = os.environ.get("MODEL_ENGINE", "mlp")
if MODEL_ENGINE not in ("mlp", "gbdt"):
    # Loud at import, not at the first rebuild: a typo'd MODEL_ENGINE that
    # fell through to a default would train one model and issue manifests
    # describing another.
    raise ValueError(f"MODEL_ENGINE must be 'mlp' or 'gbdt', got {MODEL_ENGINE!r}")


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

# Phase 6
# The dashboard reads its own connection, bound to the read-only role
# (db/schema.sql) rather than the gateway's writable one -- a role that
# cannot write enforces "the dashboard never writes to the DB directly" at
# the database, not only in application code a future edit could bypass.
DASHBOARD_DATABASE_URL = os.environ.get(
    "DASHBOARD_DATABASE_URL",
    "postgresql://unlearnshield_readonly:unlearnshield_readonly@localhost:55432/unlearnshield")

# The dashboard's one write path -- "force rebuild now" -- goes through this
# HTTP endpoint exactly as any other caller would, rather than inserting into
# erasure_jobs itself. GATEWAY_TOKEN needs erasure:write; the dev default
# matches auth_tokens()'s own default below.
DASHBOARD_GATEWAY_URL = os.environ.get("DASHBOARD_GATEWAY_URL", "http://localhost:8000")
DASHBOARD_GATEWAY_TOKEN = os.environ.get("DASHBOARD_GATEWAY_TOKEN", "dev-token")


# OPTIONAL: shard-disagreement review queue. 0.0 disables it entirely, which
# is the default -- this is an experiment bolted alongside the serving path,
# not part of the erasure guarantee, and nothing in the core flow depends on
# it. Set to a population-std threshold (spread runs ~0.08 mean, ~0.14 p99 on
# this repo's eval corpus) to start flagging.
#
# Calibrate against your own traffic rather than copying a number: spread
# depends on shard count, on how non-identical the shards' data is, and shifts
# slightly after each rebuild. Start at your observed p99 and tune from there.
DISAGREEMENT_THRESHOLD = float(os.environ.get("DISAGREEMENT_THRESHOLD", "0.0"))


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
