
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
