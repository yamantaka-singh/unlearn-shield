-- SECURITY FIX: idempotency_key was globally UNIQUE, so it collided across
-- callers. Two consequences, found by probing two principals with the same
-- key in Phase 7's security review:
--
--   1. Disclosure -- the second caller received the FIRST caller's
--      erasure_id, which it could then use against GET /v1/erasure/{id}
--      and /certificate.
--   2. Silent loss -- the second caller's request was never enqueued, but
--      still returned 202. A legally-required erasure that reports success
--      and never happens is the exact failure this system exists to prevent,
--      and it is the more serious half.
--
-- Scoping the key to the principal makes replay-detection mean "this caller
-- already sent this", which is what an Idempotency-Key is for, rather than
-- "somebody, somewhere already used this string".
ALTER TABLE erasure_jobs DROP CONSTRAINT erasure_jobs_idempotency_key_key;
ALTER TABLE erasure_jobs
    ADD CONSTRAINT erasure_jobs_principal_idempotency_key
    UNIQUE (requested_by, idempotency_key);
