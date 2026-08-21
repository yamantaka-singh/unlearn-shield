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
