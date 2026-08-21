-- UnlearnShield schema. Single source of truth; migrations replay to this.
-- Subject identifiers never appear here in raw form -- only subject_ref,
-- which is HMAC-SHA256(subject_id, tenant_key).

CREATE TABLE subject_shard_map (
    subject_ref    TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    shard          INT  NOT NULL,

    -- The EARLIEST slice containing any of this subject's rows, not "the" slice.
    -- A subject owns many records (transactions, events, sessions) scattered
    -- across slices. Rolling back to (a later slice - 1) leaves their earlier
    -- rows baked into the checkpoint we resume from, so the rebuild completes,
    -- the manifest signs, and the erasure silently did not happen.
    -- The rollback point is min_slice_idx - 1. Nothing else is correct.
    min_slice_idx  INT  NOT NULL,
    record_count   INT  NOT NULL,

    -- Shard assignment is frozen at ingest. churn_score may be recomputed
    -- upstream; this row must not follow it, or every existing checkpoint's
    -- rollback point becomes meaningless.
    assigned_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX subject_shard_map_shard_idx ON subject_shard_map (shard, min_slice_idx);

CREATE TABLE erasure_jobs (
    erasure_id      UUID PRIMARY KEY,
    subject_ref     TEXT NOT NULL,
    reason          TEXT NOT NULL CHECK (reason IN ('consent_revocation','fraud_excision')),
    shard           INT  NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued','processing','done','failed')),

    -- NOT NULL, not just UNIQUE: nullable-unique lets every request with a
    -- missing header through, so "Idempotency-Key required" would be enforced
    -- only by app code that a retry path can bypass.
    idempotency_key TEXT NOT NULL UNIQUE,

    -- A rebuild runs for minutes. Holding the SKIP LOCKED transaction open that
    -- whole time bloats Postgres and strands the job in 'processing' forever if
    -- the worker dies. Workers claim-and-commit, then train; a reaper requeues
    -- anything whose lease expired.
    lease_expires_at TIMESTAMPTZ,
    leased_by        TEXT,
    attempts         INT NOT NULL DEFAULT 0,
    last_error       TEXT,

    sla_deadline    TIMESTAMPTZ NOT NULL,
    requested_by    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);
CREATE INDEX erasure_jobs_poll_idx ON erasure_jobs (status, shard, sla_deadline)
    WHERE status IN ('queued','processing');

CREATE TABLE erasure_manifests (
    erasure_id     UUID PRIMARY KEY REFERENCES erasure_jobs(erasure_id),
    shard          INT  NOT NULL,
    resumed_from   TEXT NOT NULL,   -- checkpoint hash the rebuild resumed from
    dataset_root   TEXT NOT NULL,   -- merkle root over the retained set, post-purge
    absence_proof  JSONB NOT NULL,
    code_digest    TEXT NOT NULL,
    config_digest  TEXT NOT NULL,
    result_weights TEXT NOT NULL,   -- sha256 of the new checkpoint
    model_version  TEXT NOT NULL,
    signature      TEXT NOT NULL,
    manifest_json  JSONB NOT NULL,  -- the canonical bytes that were signed
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Which jobs get re-run to check reproducibility. Selection is derived from
-- HMAC(erasure_id, audit_key), not chosen by the worker, so an operator cannot
-- steer the sample away from jobs it would rather not have re-run, and an
-- auditor holding audit_key can confirm the sample was not gamed.
CREATE TABLE reproducibility_checks (
    erasure_id      UUID PRIMARY KEY REFERENCES erasure_jobs(erasure_id),
    expected_weights TEXT NOT NULL,
    observed_weights TEXT NOT NULL,
    matched          BOOLEAN NOT NULL,
    code_digest      TEXT NOT NULL,
    checked_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_versions (
    model_version     TEXT PRIMARY KEY,
    shard_checkpoints JSONB NOT NULL,  -- {"0": "sha256:...", "1": "sha256:..."}
    eval_set_version  TEXT NOT NULL,
    promoted_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE checkpoints (
    checkpoint_hash TEXT PRIMARY KEY,
    shard           INT  NOT NULL,
    slice_idx       INT  NOT NULL,
    file_path       TEXT NOT NULL,
    code_digest     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- No uniqueness constraint beyond the checkpoint_hash PK: every rebuild of a
-- shard produces a NEW checkpoint for the same (shard, slice_idx, code_digest)
-- -- that accumulation is correct, expected history, not a duplicate. A
-- unique index on that triple (an earlier draft of this schema had one)
-- breaks the second rebuild of any shard outright, since content-addressed
-- hashes are the only uniqueness this table needs.
CREATE INDEX checkpoints_shard_slice_idx ON checkpoints (shard, slice_idx);

-- One row per promotion: the ensemble's AUC against a frozen, non-subject eval
-- corpus (data/eval_set.py -- synthetic rows with no subject_ref, so they can
-- never be the target of an erasure and never need their own purge-state).
-- That sidesteps the harder version of this problem, where the eval set
-- itself contains real subjects and an erasure has to be reflected in it too;
-- noted here rather than silently presented as solving that harder case.
CREATE TABLE eval_results (
    model_version TEXT PRIMARY KEY REFERENCES model_versions(model_version),
    auc           REAL NOT NULL,
    n_eval        INT  NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Phase 6's dashboard reads Postgres directly. It never writes -- a
-- "force rebuild now" button goes through the gateway's own /v1/erasure route
-- (gateway/routes/erasure.py), the same as any other caller, rather than
-- inserting a row itself. A role that cannot write enforces that boundary at
-- the database, not just in application code a future edit could bypass.
--
-- CREATE ROLE cannot run inside a transaction alongside CREATE TABLE in some
-- Postgres configurations and is idempotency-sensitive across re-applied
-- migrations, so it is guarded rather than a bare statement.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'unlearnshield_readonly') THEN
        CREATE ROLE unlearnshield_readonly LOGIN PASSWORD 'unlearnshield_readonly';
    END IF;
END
$$;
GRANT CONNECT ON DATABASE unlearnshield TO unlearnshield_readonly;
GRANT USAGE ON SCHEMA public TO unlearnshield_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO unlearnshield_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO unlearnshield_readonly;

-- OPTIONAL FEATURE, off by default (DISAGREEMENT_THRESHOLD=0.0). Not part of
-- the erasure guarantee and not on any path the manifest/SMT/signing logic
-- touches. See docs/adr/0009-shard-disagreement-review-queue.md.
--
-- When shards disagree sharply about a transaction, that spread is an
-- epistemic-uncertainty signal -- possibly a fraud pattern only some shards
-- have seen. Measured on this repo's eval corpus, spread scores AUC 0.574 as
-- a fraud detector against the served mean's 0.515, so the signal is real.
--
-- Deliberately stores NO transaction features. gateway/schemas.py's
-- PredictRequest carries no subject_id, so a row here could never be linked
-- to a subject -- which means the erasure path (subject_ref ->
-- subject_shard_map -> shard rebuild) could never reach it. Storing features
-- would create a growing store of personal data this system cannot erase,
-- in a system whose entire claim is that it can. Per-shard scores and the
-- model_version are what a reviewer needs to see WHICH shards fired and
-- against which model; the features are recoverable from the caller's own
-- logs, where they are already subject to that caller's retention policy.
--
-- Adding a features column requires first adding subject_ref here and wiring
-- this table into engine/rebuild.py's purge. Do not add one without that.
CREATE TABLE disagreement_reviews (
    review_id     BIGSERIAL PRIMARY KEY,

    -- Which ensemble produced this spread. A rebuild retrains one shard and
    -- leaves the others alone, so its decision boundary shifts relative to
    -- theirs -- measured at +2.0% mean spread, flag rate 1.0% -> 1.1% for a
    -- single-shard rebuild. Small, but it means a spread trend is only
    -- interpretable per model_version, not across them.
    model_version TEXT NOT NULL,
    shard_scores  REAL[] NOT NULL,   -- per-shard probability, index = shard
    mean_score    REAL NOT NULL,     -- what the caller was actually served
    spread        REAL NOT NULL,     -- population std across shard_scores

    -- Recorded per row, not read from config at review time: the threshold is
    -- env-configurable, so a later change would otherwise make every existing
    -- row uninterpretable -- "was this flagged because it was extreme, or
    -- because the bar was low that week?"
    threshold     REAL NOT NULL,

    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','reviewed','dismissed')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX disagreement_reviews_queue_idx
    ON disagreement_reviews (status, created_at) WHERE status = 'pending';
GRANT SELECT ON disagreement_reviews TO unlearnshield_readonly;
