
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
