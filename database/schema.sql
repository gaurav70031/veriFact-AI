-- =============================================================================
-- Fake News Detection — PostgreSQL Schema
-- =============================================================================
-- This file is the authoritative schema definition.
-- It is used by:
--   1. Docker Compose (mounted to /docker-entrypoint-initdb.d/) on first init
--   2. Alembic (as reference — Alembic manages incremental migrations)
--   3. Documentation
--
-- DO NOT run this manually on an existing database — use Alembic migrations.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid(), crypt()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- trigram indexes for text search


-- ---------------------------------------------------------------------------
-- ENUM types
-- ---------------------------------------------------------------------------

CREATE TYPE user_role AS ENUM ('user', 'analyst', 'admin');

CREATE TYPE input_type AS ENUM ('text', 'url', 'query');

CREATE TYPE analysis_status AS ENUM ('pending', 'processing', 'completed', 'failed');

CREATE TYPE final_verdict AS ENUM ('FAKE', 'REAL', 'UNVERIFIED', 'MIXED');

CREATE TYPE claim_verdict AS ENUM ('FAKE', 'REAL', 'UNVERIFIED', 'MIXED');

CREATE TYPE prediction_label AS ENUM ('FAKE', 'REAL');

CREATE TYPE model_type AS ENUM ('traditional', 'transformer', 'ensemble');

CREATE TYPE model_algorithm AS ENUM (
    'logistic_regression',
    'linear_svm',
    'naive_bayes',
    'distilbert',
    'ensemble'
);

CREATE TYPE source_type AS ENUM (
    'news_api',
    'rss_feed',
    'web_search',
    'official',
    'fact_check',
    'academic',
    'user_url'
);

CREATE TYPE evidence_relationship AS ENUM ('supporting', 'contradicting', 'inconclusive');

CREATE TYPE run_status AS ENUM ('success', 'failed', 'timeout', 'skipped');


-- ---------------------------------------------------------------------------
-- TABLE: users
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id                  SERIAL          PRIMARY KEY,
    email               VARCHAR(255)    NOT NULL,
    username            VARCHAR(100)    NOT NULL,
    hashed_password     VARCHAR(255)    NOT NULL,
    full_name           VARCHAR(200),
    role                user_role       NOT NULL DEFAULT 'user',
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    is_verified         BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_users_email    UNIQUE (email),
    CONSTRAINT uq_users_username UNIQUE (username)
);

CREATE INDEX ix_users_email_active ON users (email, is_active);
CREATE INDEX ix_users_username     ON users (username);


-- ---------------------------------------------------------------------------
-- TABLE: model_versions
-- (defined before analyses/predictions so FKs can reference it)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    id                  SERIAL          PRIMARY KEY,
    model_name          VARCHAR(100)    NOT NULL,
    version             VARCHAR(50)     NOT NULL,
    model_type          model_type      NOT NULL,
    algorithm           model_algorithm NOT NULL,
    artifact_path       VARCHAR(500)    NOT NULL,
    vectorizer_path     VARCHAR(500),
    dataset_name        VARCHAR(200),
    training_samples    INTEGER,
    test_samples        INTEGER,

    -- Evaluation metrics recorded at training time
    accuracy            FLOAT,
    precision           FLOAT,
    recall              FLOAT,
    f1_score            FLOAT,
    roc_auc             FLOAT,

    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    description         TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_model_name_version UNIQUE (model_name, version),
    CONSTRAINT ck_mv_accuracy  CHECK (accuracy  IS NULL OR (accuracy  BETWEEN 0.0 AND 1.0)),
    CONSTRAINT ck_mv_precision CHECK (precision IS NULL OR (precision BETWEEN 0.0 AND 1.0)),
    CONSTRAINT ck_mv_recall    CHECK (recall    IS NULL OR (recall    BETWEEN 0.0 AND 1.0)),
    CONSTRAINT ck_mv_f1        CHECK (f1_score  IS NULL OR (f1_score  BETWEEN 0.0 AND 1.0)),
    CONSTRAINT ck_mv_roc_auc   CHECK (roc_auc   IS NULL OR (roc_auc   BETWEEN 0.0 AND 1.0))
);

CREATE INDEX ix_model_versions_active    ON model_versions (is_active);
CREATE INDEX ix_model_versions_algorithm ON model_versions (algorithm);


-- ---------------------------------------------------------------------------
-- TABLE: analyses
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analyses (
    id                  SERIAL              PRIMARY KEY,
    user_id             INTEGER             REFERENCES users (id) ON DELETE SET NULL,
    input_type          input_type          NOT NULL,
    original_input      TEXT                NOT NULL,
    source_url          VARCHAR(2000),
    article_title       VARCHAR(500),
    article_text        TEXT,
    status              analysis_status     NOT NULL DEFAULT 'pending',
    final_verdict       final_verdict,
    final_confidence    FLOAT,
    summary             TEXT,
    error_message       TEXT,
    processing_time_ms  INTEGER,            -- total wall-clock ms
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_analysis_confidence CHECK (
        final_confidence IS NULL OR (final_confidence BETWEEN 0.0 AND 1.0)
    ),
    CONSTRAINT ck_analysis_processing_time CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

CREATE INDEX ix_analyses_user_created   ON analyses (user_id, created_at DESC);
CREATE INDEX ix_analyses_status_created ON analyses (status, created_at DESC);
CREATE INDEX ix_analyses_verdict        ON analyses (final_verdict);
-- Trigram index enables fast ILIKE search on original input
CREATE INDEX ix_analyses_input_trgm     ON analyses USING GIN (original_input gin_trgm_ops);


-- ---------------------------------------------------------------------------
-- TABLE: claims
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims (
    id              SERIAL          PRIMARY KEY,
    analysis_id     INTEGER         NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
    claim_text      TEXT            NOT NULL,
    position        INTEGER         NOT NULL DEFAULT 1,
    verdict         claim_verdict,
    confidence      FLOAT,
    explanation     TEXT,
    top_tokens_json TEXT,           -- JSON array: [{"token":"...", "weight": 0.42}]
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_claim_confidence CHECK (
        confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)
    ),
    CONSTRAINT ck_claim_position CHECK (position >= 1)
);

CREATE INDEX ix_claims_analysis_position ON claims (analysis_id, position);
-- Trigram index for claim text search
CREATE INDEX ix_claims_text_trgm         ON claims USING GIN (claim_text gin_trgm_ops);


-- ---------------------------------------------------------------------------
-- TABLE: predictions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL              PRIMARY KEY,
    analysis_id         INTEGER             NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
    model_version_id    INTEGER             NOT NULL REFERENCES model_versions (id) ON DELETE RESTRICT,
    label               prediction_label    NOT NULL,
    confidence          FLOAT               NOT NULL,
    fake_probability    FLOAT               NOT NULL,
    real_probability    FLOAT               NOT NULL,
    explanation_json    TEXT,               -- LIME / attention weights JSON
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_pred_confidence    CHECK (confidence       BETWEEN 0.0 AND 1.0),
    CONSTRAINT ck_pred_fake_prob     CHECK (fake_probability BETWEEN 0.0 AND 1.0),
    CONSTRAINT ck_pred_real_prob     CHECK (real_probability BETWEEN 0.0 AND 1.0)
);

CREATE INDEX ix_predictions_analysis      ON predictions (analysis_id);
CREATE INDEX ix_predictions_model_version ON predictions (model_version_id);
CREATE INDEX ix_predictions_label         ON predictions (label);
CREATE INDEX ix_predictions_created       ON predictions (created_at DESC);


-- ---------------------------------------------------------------------------
-- TABLE: evidence_sources
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_sources (
    id                      SERIAL                  PRIMARY KEY,
    claim_id                INTEGER                 NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
    source_name             VARCHAR(200)            NOT NULL,
    title                   VARCHAR(500)            NOT NULL,
    url                     VARCHAR(2000)           NOT NULL,
    snippet                 VARCHAR(500),           -- ≤500 chars, copyright-safe excerpt
    source_type             source_type             NOT NULL,
    published_at            TIMESTAMPTZ,
    retrieved_at            TIMESTAMPTZ             NOT NULL,
    relevance_score         FLOAT                   NOT NULL,
    rank                    INTEGER                 NOT NULL DEFAULT 1,
    relationship_to_claim   evidence_relationship   NOT NULL DEFAULT 'inconclusive',
    created_at              TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ             NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_evidence_relevance CHECK (relevance_score BETWEEN 0.0 AND 1.0),
    CONSTRAINT ck_evidence_rank      CHECK (rank >= 1)
);

CREATE INDEX ix_evidence_claim_rank    ON evidence_sources (claim_id, rank);
CREATE INDEX ix_evidence_source_type   ON evidence_sources (source_type);
CREATE INDEX ix_evidence_relationship  ON evidence_sources (relationship_to_claim);
CREATE INDEX ix_evidence_relevance     ON evidence_sources (relevance_score DESC);


-- ---------------------------------------------------------------------------
-- TABLE: model_runs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    id                  SERIAL          PRIMARY KEY,
    analysis_id         INTEGER         NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
    model_version_id    INTEGER         NOT NULL REFERENCES model_versions (id) ON DELETE RESTRICT,
    status              run_status      NOT NULL DEFAULT 'success',
    inference_time_ms   INTEGER,        -- wall-clock ms for this model only
    input_length        INTEGER,        -- character count of preprocessed input
    succeeded           BOOLEAN         NOT NULL DEFAULT TRUE,
    error_message       TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_run_inference_time CHECK (
        inference_time_ms IS NULL OR inference_time_ms >= 0
    ),
    CONSTRAINT ck_run_input_length CHECK (
        input_length IS NULL OR input_length >= 0
    )
);

CREATE INDEX ix_model_runs_analysis      ON model_runs (analysis_id);
CREATE INDEX ix_model_runs_status        ON model_runs (status);
CREATE INDEX ix_model_runs_model_version ON model_runs (model_version_id);
CREATE INDEX ix_model_runs_created       ON model_runs (created_at DESC);


-- ---------------------------------------------------------------------------
-- Auto-update updated_at via trigger (optional but useful)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'users', 'analyses', 'claims',
        'model_versions', 'predictions',
        'evidence_sources', 'model_runs'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();',
            t, t
        );
    END LOOP;
END;
$$;
