# Database Layer — Fake News Detection

PostgreSQL 15 · SQLAlchemy 2 (async) · Alembic migrations

---

## Table of Contents

1. [Overview](#overview)
2. [Entity Relationship Summary](#entity-relationship-summary)
3. [Table Reference](#table-reference)
   - [users](#users)
   - [model_versions](#model_versions)
   - [analyses](#analyses)
   - [claims](#claims)
   - [predictions](#predictions)
   - [evidence_sources](#evidence_sources)
   - [model_runs](#model_runs)
4. [ENUM Types](#enum-types)
5. [Indexes](#indexes)
6. [Constraints](#constraints)
7. [File Structure](#file-structure)
8. [Setup and Migration](#setup-and-migration)
9. [Dev Seed Data](#dev-seed-data)
10. [Design Decisions](#design-decisions)

---

## Overview

The database is the system of record for every analysis the application
performs.  It stores:

- **Who** submitted the analysis (users, or anonymous)
- **What** was submitted (raw text, extracted article, URL, query)
- **How** each model assessed it (one prediction row per model per analysis)
- **Why** the verdict was reached (evidence sources, explainability JSON)
- **When and how fast** it was processed (timestamps, processing_time_ms)
- **Which model artifact** was used (model_versions registry)

No full article bodies from third-party sources are stored — only metadata
and short snippets (≤ 500 characters) to respect copyright.

---

## Entity Relationship Summary

```
users
  │
  └── analyses (user_id → users.id, SET NULL on delete)
        │
        ├── claims (analysis_id → analyses.id, CASCADE)
        │     │
        │     └── evidence_sources (claim_id → claims.id, CASCADE)
        │
        ├── predictions (analysis_id → analyses.id, CASCADE)
        │     └── model_versions (model_version_id → model_versions.id, RESTRICT)
        │
        └── model_runs (analysis_id → analyses.id, CASCADE)
              └── model_versions (model_version_id → model_versions.id, RESTRICT)
```

Key design points:
- Deleting a user sets their analyses' `user_id` to NULL (history is preserved).
- Deleting an analysis cascades to all its claims, predictions, evidence, and runs.
- Model versions use RESTRICT on delete — you cannot delete a version that has
  associated predictions or runs (protects audit trail).

---

## Table Reference

### users

Stores registered user accounts.

| Column           | Type         | Notes                                      |
|------------------|--------------|--------------------------------------------|
| id               | SERIAL PK    |                                            |
| email            | VARCHAR(255) | Unique, indexed                            |
| username         | VARCHAR(100) | Unique, indexed                            |
| hashed_password  | VARCHAR(255) | bcrypt hash — never plaintext              |
| full_name        | VARCHAR(200) | Optional display name                      |
| role             | user_role    | `user` \| `analyst` \| `admin`             |
| is_active        | BOOLEAN      | False = soft-deleted / suspended           |
| is_verified      | BOOLEAN      | Email verification status                  |
| created_at       | TIMESTAMPTZ  | Auto-set on insert                         |
| updated_at       | TIMESTAMPTZ  | Auto-updated via trigger                   |

---

### model_versions

Registry of every trained ML artifact.  One row per model name + version
combination.  All predictions and model runs reference a row here so
results are always traceable to the exact artifact that produced them.

| Column           | Type            | Notes                                         |
|------------------|-----------------|-----------------------------------------------|
| id               | SERIAL PK       |                                               |
| model_name       | VARCHAR(100)    | e.g. `tfidf_logistic_regression`              |
| version          | VARCHAR(50)     | Semantic version, e.g. `1.0.0`                |
| model_type       | model_type      | `traditional` \| `transformer` \| `ensemble`  |
| algorithm        | model_algorithm | See ENUM section                              |
| artifact_path    | VARCHAR(500)    | Relative path to `.pkl` or model directory    |
| vectorizer_path  | VARCHAR(500)    | Path to TF-IDF vectorizer (traditional only)  |
| dataset_name     | VARCHAR(200)    | Training dataset name                         |
| training_samples | INTEGER         | Number of training examples                   |
| test_samples     | INTEGER         | Number of test examples                       |
| accuracy         | FLOAT           | 0.0–1.0, CHECK constraint                     |
| precision        | FLOAT           | 0.0–1.0, CHECK constraint                     |
| recall           | FLOAT           | 0.0–1.0, CHECK constraint                     |
| f1_score         | FLOAT           | 0.0–1.0, CHECK constraint                     |
| roc_auc          | FLOAT           | 0.0–1.0, CHECK constraint                     |
| is_active        | BOOLEAN         | Only active versions are used for inference   |
| description      | TEXT            | Human-readable notes                          |
| created_at       | TIMESTAMPTZ     |                                               |
| updated_at       | TIMESTAMPTZ     |                                               |

Unique constraint: `(model_name, version)`.

---

### analyses

Top-level entity.  One row per user submission regardless of input type.

| Column             | Type            | Notes                                              |
|--------------------|-----------------|---------------------------------------------------|
| id                 | SERIAL PK       |                                                   |
| user_id            | INTEGER FK      | → users.id, SET NULL. NULL = anonymous            |
| input_type         | input_type      | `text` \| `url` \| `query`                        |
| original_input     | TEXT            | Raw user input (text, URL string, or query)       |
| source_url         | VARCHAR(2000)   | Populated when input_type = url                   |
| article_title      | VARCHAR(500)    | Extracted title (URL submissions)                 |
| article_text       | TEXT            | Extracted clean article body (URL submissions)    |
| status             | analysis_status | `pending` → `processing` → `completed`/`failed`  |
| final_verdict      | final_verdict   | `FAKE` \| `REAL` \| `UNVERIFIED` \| `MIXED`       |
| final_confidence   | FLOAT           | 0.0–1.0, ensemble final confidence                |
| summary            | TEXT            | Human-readable explanation of verdict             |
| error_message      | TEXT            | Populated on status = failed                      |
| processing_time_ms | INTEGER         | Total wall-clock time from request to response    |
| created_at         | TIMESTAMPTZ     |                                                   |
| updated_at         | TIMESTAMPTZ     |                                                   |

GIN trigram index on `original_input` enables fast text search across history.

---

### claims

An analysis can contain one or more discrete claims extracted from the
submitted content.  Each claim is assessed independently.

| Column          | Type          | Notes                                              |
|-----------------|---------------|----------------------------------------------------|
| id              | SERIAL PK     |                                                    |
| analysis_id     | INTEGER FK    | → analyses.id, CASCADE                             |
| claim_text      | TEXT          | The isolated claim statement                       |
| position        | INTEGER       | Ordinal within analysis (1-based), CHECK ≥ 1       |
| verdict         | claim_verdict | `FAKE` \| `REAL` \| `UNVERIFIED` \| `MIXED`        |
| confidence      | FLOAT         | 0.0–1.0                                            |
| explanation     | TEXT          | Natural language explanation of the verdict        |
| top_tokens_json | TEXT          | JSON: `[{"token": "...", "weight": 0.42}]`         |
| created_at      | TIMESTAMPTZ   |                                                    |
| updated_at      | TIMESTAMPTZ   |                                                    |

`top_tokens_json` stores the top contributing words/tokens produced by LIME
(traditional models) or attention weight extraction (DistilBERT).  Stored as
TEXT rather than a JSON column to avoid a hard dependency on the pgJSON
extension in all environments.

---

### predictions

One row per model per analysis.  For a typical analysis with 4 models
(LR, SVM, NB, DistilBERT) + 1 ensemble, there will be 5 prediction rows
linked to the same analysis.

| Column           | Type             | Notes                                          |
|------------------|------------------|------------------------------------------------|
| id               | SERIAL PK        |                                                |
| analysis_id      | INTEGER FK       | → analyses.id, CASCADE                         |
| model_version_id | INTEGER FK       | → model_versions.id, RESTRICT                  |
| label            | prediction_label | `FAKE` \| `REAL`                               |
| confidence       | FLOAT            | max(fake_prob, real_prob), CHECK 0.0–1.0       |
| fake_probability | FLOAT            | Raw softmax probability for FAKE, 0.0–1.0      |
| real_probability | FLOAT            | Raw softmax probability for REAL, 0.0–1.0      |
| explanation_json | TEXT             | LIME / attention weights JSON for this model   |
| created_at       | TIMESTAMPTZ      |                                                |
| updated_at       | TIMESTAMPTZ      |                                                |

This normalised structure enables per-model accuracy tracking over time via
simple aggregate queries against `predictions JOIN model_versions`.

---

### evidence_sources

External articles/documents retrieved to support or contradict a claim.
Full article bodies are **not** stored — only metadata and a short snippet
(≤ 500 chars) to respect copyright.

| Column               | Type                 | Notes                                           |
|----------------------|----------------------|-------------------------------------------------|
| id                   | SERIAL PK            |                                                 |
| claim_id             | INTEGER FK           | → claims.id, CASCADE                            |
| source_name          | VARCHAR(200)         | Publisher name, e.g. `Reuters`                  |
| title                | VARCHAR(500)         | Article headline                                |
| url                  | VARCHAR(2000)        | Canonical URL of the source article             |
| snippet              | VARCHAR(500)         | Short excerpt, copyright-safe                   |
| source_type          | source_type          | See ENUM section                                |
| published_at         | TIMESTAMPTZ          | Article publication date (nullable)             |
| retrieved_at         | TIMESTAMPTZ          | When this evidence was fetched                  |
| relevance_score      | FLOAT                | Cosine/BM25 similarity vs. claim, 0.0–1.0       |
| rank                 | INTEGER              | Position in result set (1 = most relevant)      |
| relationship_to_claim| evidence_relationship| `supporting` \| `contradicting` \| `inconclusive`|
| created_at           | TIMESTAMPTZ          |                                                 |
| updated_at           | TIMESTAMPTZ          |                                                 |

---

### model_runs

Execution log — one row per model invocation per analysis.
Records latency independently from predictions, enabling monitoring of
model inference speed over time without coupling it to the result.

| Column           | Type        | Notes                                              |
|------------------|-------------|----------------------------------------------------|
| id               | SERIAL PK   |                                                    |
| analysis_id      | INTEGER FK  | → analyses.id, CASCADE                             |
| model_version_id | INTEGER FK  | → model_versions.id, RESTRICT                      |
| status           | run_status  | `success` \| `failed` \| `timeout` \| `skipped`   |
| inference_time_ms| INTEGER     | Wall-clock time for this model only (ms)           |
| input_length     | INTEGER     | Character count of preprocessed input              |
| succeeded        | BOOLEAN     | Convenience flag mirroring status = success        |
| error_message    | TEXT        | Populated on failure                               |
| created_at       | TIMESTAMPTZ |                                                    |
| updated_at       | TIMESTAMPTZ |                                                    |

---

## ENUM Types

| ENUM name              | Values                                                                    |
|------------------------|---------------------------------------------------------------------------|
| `user_role`            | `user`, `analyst`, `admin`                                                |
| `input_type`           | `text`, `url`, `query`                                                    |
| `analysis_status`      | `pending`, `processing`, `completed`, `failed`                            |
| `final_verdict`        | `FAKE`, `REAL`, `UNVERIFIED`, `MIXED`                                     |
| `claim_verdict`        | `FAKE`, `REAL`, `UNVERIFIED`, `MIXED`                                     |
| `prediction_label`     | `FAKE`, `REAL`                                                            |
| `model_type`           | `traditional`, `transformer`, `ensemble`                                  |
| `model_algorithm`      | `logistic_regression`, `linear_svm`, `naive_bayes`, `distilbert`, `ensemble` |
| `source_type`          | `news_api`, `rss_feed`, `web_search`, `official`, `fact_check`, `academic`, `user_url` |
| `evidence_relationship`| `supporting`, `contradicting`, `inconclusive`                             |
| `run_status`           | `success`, `failed`, `timeout`, `skipped`                                 |

---

## Indexes

| Index name                  | Table            | Columns                        | Type    | Purpose                              |
|-----------------------------|------------------|--------------------------------|---------|--------------------------------------|
| ix_users_email_active       | users            | (email, is_active)             | B-Tree  | Login lookups                        |
| ix_users_username           | users            | username                       | B-Tree  | Username uniqueness check            |
| ix_model_versions_active    | model_versions   | is_active                      | B-Tree  | Filter active models at startup      |
| ix_model_versions_algorithm | model_versions   | algorithm                      | B-Tree  | Analytics by model type              |
| ix_analyses_user_created    | analyses         | (user_id, created_at DESC)     | B-Tree  | User history page (paginated)        |
| ix_analyses_status_created  | analyses         | (status, created_at DESC)      | B-Tree  | Processing queue monitoring          |
| ix_analyses_verdict         | analyses         | final_verdict                  | B-Tree  | Analytics: count by verdict          |
| ix_analyses_input_trgm      | analyses         | original_input                 | **GIN** | Full-text ILIKE search on input      |
| ix_claims_analysis_position | claims           | (analysis_id, position)        | B-Tree  | Ordered claim retrieval              |
| ix_claims_text_trgm         | claims           | claim_text                     | **GIN** | Full-text search on claim text       |
| ix_predictions_analysis     | predictions      | analysis_id                    | B-Tree  | Fetch all predictions for analysis   |
| ix_predictions_model_version| predictions      | model_version_id               | B-Tree  | Analytics: accuracy per model        |
| ix_predictions_label        | predictions      | label                          | B-Tree  | Count FAKE vs REAL distribution      |
| ix_predictions_created      | predictions      | created_at DESC                | B-Tree  | Time-series analytics                |
| ix_evidence_claim_rank      | evidence_sources | (claim_id, rank)               | B-Tree  | Ordered evidence retrieval           |
| ix_evidence_source_type     | evidence_sources | source_type                    | B-Tree  | Filter by source category            |
| ix_evidence_relationship    | evidence_sources | relationship_to_claim          | B-Tree  | Count supporting vs contradicting    |
| ix_evidence_relevance       | evidence_sources | relevance_score DESC           | B-Tree  | Sort evidence by relevance           |
| ix_model_runs_analysis      | model_runs       | analysis_id                    | B-Tree  | Fetch all runs for analysis          |
| ix_model_runs_status        | model_runs       | status                         | B-Tree  | Monitor failed runs                  |
| ix_model_runs_created       | model_runs       | created_at DESC                | B-Tree  | Latency trend queries                |

GIN indexes require the `pg_trgm` extension (enabled in schema.sql and migration).

---

## Constraints

All numeric probability/confidence values are guarded by CHECK constraints
so the application layer cannot accidentally store out-of-range values.

| Constraint name              | Table          | Rule                                          |
|------------------------------|----------------|-----------------------------------------------|
| ck_analysis_confidence       | analyses       | final_confidence BETWEEN 0.0 AND 1.0 or NULL  |
| ck_analysis_processing_time  | analyses       | processing_time_ms >= 0 or NULL               |
| ck_claim_confidence_range    | claims         | confidence BETWEEN 0.0 AND 1.0 or NULL        |
| ck_claim_position_positive   | claims         | position >= 1                                 |
| ck_pred_confidence           | predictions    | confidence BETWEEN 0.0 AND 1.0                |
| ck_pred_fake_prob            | predictions    | fake_probability BETWEEN 0.0 AND 1.0          |
| ck_pred_real_prob            | predictions    | real_probability BETWEEN 0.0 AND 1.0          |
| ck_evidence_relevance_range  | evidence_sources| relevance_score BETWEEN 0.0 AND 1.0          |
| ck_evidence_rank_positive    | evidence_sources| rank >= 1                                    |
| ck_run_inference_time        | model_runs     | inference_time_ms >= 0 or NULL                |
| ck_run_input_length          | model_runs     | input_length >= 0 or NULL                     |
| ck_mv_accuracy … ck_mv_roc_auc | model_versions | Each metric BETWEEN 0.0 AND 1.0 or NULL    |
| uq_model_name_version        | model_versions | (model_name, version) is unique               |
| uq_users_email               | users          | email is unique                               |
| uq_users_username            | users          | username is unique                            |

---

## File Structure

```
database/
├── schema.sql              # Full DDL — used by Docker on first init
├── README.md               # This file
└── seeds/
    └── dev_seed.py         # DEV ONLY — inserts test data

backend/
├── alembic.ini             # Alembic configuration
├── alembic/
│   ├── env.py              # Migration environment (reads DB URL from settings)
│   ├── script.py.mako      # Template for generated migration files
│   ├── README              # Alembic command reference
│   └── versions/
│       └── 0001_initial_schema.py   # Initial migration (all tables)
└── app/
    ├── core/
    │   └── config.py       # Pydantic Settings (database_url property)
    ├── db/
    │   ├── base.py         # DeclarativeBase + TimestampMixin
    │   └── session.py      # Async engine + get_db() FastAPI dependency
    └── models/
        ├── __init__.py     # Exports all models + enums
        ├── user.py
        ├── analysis.py
        ├── claim.py
        ├── model_version.py
        ├── prediction.py
        ├── evidence_source.py
        └── model_run.py
```

---

## Setup and Migration

### Prerequisites

```bash
# From backend/
pip install sqlalchemy alembic asyncpg psycopg2-binary pydantic-settings python-dotenv
```

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — set POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
```

### 2. Start PostgreSQL

```bash
# Via Docker Compose (recommended)
docker-compose up db -d

# Or use a local PostgreSQL 15 instance
```

### 3. Apply migrations

```bash
cd backend/
alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, Initial schema — all tables, ENUMs, indexes, constraints, trigger
```

### 4. Verify

```bash
alembic current
# Should show: 0001 (head)

alembic history --verbose
# Should show the single initial migration
```

### Common migration commands

```bash
# Generate a new migration after changing a model
alembic revision --autogenerate -m "add user avatar url"

# Apply pending migrations
alembic upgrade head

# Roll back one step
alembic downgrade -1

# Roll back everything (drops all tables)
alembic downgrade base

# Preview SQL without executing (offline mode)
alembic upgrade head --sql
```

---

## Dev Seed Data

The seed script creates realistic fictional data for development and testing.
**Do not run against a production database.**

```bash
# From the project root (with .env present)
python database/seeds/dev_seed.py
```

The script will refuse to run if the configured database name contains
`prod` or `live`.

### What it seeds

| Entity         | Count | Notes                                       |
|----------------|-------|---------------------------------------------|
| users          | 3     | One per role: admin, analyst, regular user  |
| model_versions | 5     | LR, SVM, NB, DistilBERT, Ensemble           |
| analyses       | 6     | Covers: FAKE, REAL, UNVERIFIED, FAILED, PENDING, text/url/query inputs |
| claims         | 4     | One per completed analysis                  |
| evidence_sources | 7   | Mix of official, news_api, fact_check       |
| predictions    | 16    | Multiple models per analysis                |
| model_runs     | 10    | Execution logs with inference times         |

### Dev credentials

| Role    | Email                        | Password       |
|---------|------------------------------|----------------|
| admin   | dev.admin@example.test       | DevAdmin@1234  |
| analyst | analyst.test@example.test    | Analyst@5678   |
| user    | regular.user@example.test    | User@9999      |

These credentials exist only in development databases and are clearly
fictional (`.test` TLD).

---

## Design Decisions

**Why separate `claims` from `analyses`?**
A single article may contain multiple distinct factual claims that need
independent assessment. Normalising them allows per-claim evidence and
per-claim explainability without duplicating analysis-level data.

**Why separate `predictions` from `analyses`?**
Storing one prediction row per model enables per-model accuracy analytics
over time using simple GROUP BY queries, without any JSON parsing.

**Why separate `model_runs` from `predictions`?**
Execution telemetry (latency, failure reason) is a different concern from
the result of inference. Keeping them separate means monitoring queries
do not touch the predictions table, and a failed run has a record even when
no prediction was produced.

**Why RESTRICT on model_version FK deletion?**
Deleting a model version that has associated predictions or runs would
silently destroy the audit trail linking results to the exact artifact that
produced them. RESTRICT forces an explicit decision before any deletion.

**Why store `top_tokens_json` as TEXT not JSONB?**
Avoids a hard dependency on PostgreSQL's JSON operators in the application
query layer. The JSON is always read and parsed in Python, never queried
inside the database. TEXT is portable and sufficient.

**Why GIN trigram indexes on text columns?**
`pg_trgm` GIN indexes allow efficient `ILIKE '%search term%'` queries on
`original_input` and `claim_text` without a full-table scan. This powers
the history search feature.
