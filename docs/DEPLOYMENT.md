# VeritasAI — Deployment Guide

Complete guide for running the application locally with Docker Compose
and deploying it to a cloud platform.

---

## Table of Contents

1. [Local Setup](#1-local-setup)
2. [Environment Variables](#2-environment-variables)
3. [Database Migration](#3-database-migration)
4. [ML Model Setup](#4-ml-model-setup)
5. [Starting Services](#5-starting-services)
6. [Production Deployment](#6-production-deployment)
7. [Health Checks](#7-health-checks)
8. [Logs](#8-logs)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Local Setup

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker | ≥ 24 | Docker Desktop on Windows/Mac |
| Docker Compose | ≥ 2.20 | Included with Docker Desktop |
| Python | 3.11+ | For training ML models on host |
| Node.js | 18+ | For local frontend dev only |

### Step-by-step

```bash
# 1. Clone the repository
git clone <repo-url>
cd fake-news-detection

# 2. Create your .env file from the template
cp .env.example .env

# 3. Fill in the required secrets (see Section 2)
#    At minimum: SECRET_KEY, POSTGRES_PASSWORD
#    Optionally:  NEWSAPI_KEY (for live evidence retrieval)

# 4. Train ML models (first time only — needs Python on host)
pip install -r backend/requirements.txt
python scripts/prepare_dataset.py \
  --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv
python scripts/train_baseline.py
# Optional (needs GPU for reasonable time):
# python scripts/train_transformer.py

# 5. Build and start all services
docker compose up --build

# The application will be available at:
#   Frontend : http://localhost
#   Backend  : http://localhost:8000
#   API Docs : http://localhost:8000/docs
```

The backend container automatically runs `alembic upgrade head` on startup
before the application starts.

---

## 2. Environment Variables

### Required — must be set before first run

| Variable | Example | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `27e50f88...` | JWT signing secret. Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `POSTGRES_PASSWORD` | `str0ng_pass` | PostgreSQL database password |
| `POSTGRES_DB` | `fakenews_db` | Database name |
| `POSTGRES_USER` | `fakenews_user` | Database user |

### Optional — unlock features when set

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWSAPI_KEY` | _(empty)_ | Enable NewsAPI evidence retrieval |
| `GNEWS_API_KEY` | _(empty)_ | Enable GNews evidence retrieval |
| `SERPAPI_KEY` | _(empty)_ | Enable web search evidence retrieval |
| `BRAVE_SEARCH_KEY` | _(empty)_ | Alternative to SerpAPI |
| `SENTENCE_TRANSFORMERS_ENABLED` | `false` | Better evidence comparison (requires local model cache) |

### Production — must change from defaults

| Variable | Development | Production |
|----------|-------------|------------|
| `APP_ENV` | `development` | `production` |
| `DEBUG` | `true` | `false` |
| `COOKIE_SECURE` | `false` | `true` (requires HTTPS) |
| `COOKIE_SAMESITE` | `lax` | `lax` or `none` |
| `ALLOWED_ORIGINS` | `http://localhost` | `https://yourdomain.com` |

### Variables never placed in Dockerfiles

All secrets — `SECRET_KEY`, `POSTGRES_PASSWORD`, API keys — are read at
runtime from environment variables or Docker secrets. They are never
embedded in the Docker image.

---

## 3. Database Migration

### How it works

The backend container's entrypoint (`docker/entrypoint.sh`) automatically
runs `alembic upgrade head` every time the container starts. This is safe
because Alembic migrations are idempotent.

```
Container start
  → docker/entrypoint.sh
  → Wait for PostgreSQL (max 60s, 2s intervals)
  → alembic upgrade head   (applies any pending migrations)
  → uvicorn app.main:app   (starts the API)
```

### Running migrations manually

```bash
# From outside Docker (host machine, from backend/ directory):
cd backend
alembic upgrade head

# Inside a running container:
docker compose exec backend alembic upgrade head

# Show current migration state:
docker compose exec backend alembic current

# Roll back one migration:
docker compose exec backend alembic downgrade -1

# Generate a new migration after changing ORM models:
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

### Migration history

| Revision | Description |
|----------|-------------|
| `0001` | Initial schema — all 7 tables, ENUMs, indexes, triggers |
| `0002` | Evidence engine — NOT_RELEVANT enum, evidence_verdict columns, comparison_score |

---

## 4. ML Model Setup

ML models are **not included in the Docker image** — they must be trained
separately and mounted into the container.

### Why external models?

- Model files are 50 MB – 500 MB each (too large for a container image)
- Models may be retrained; the container should not need a rebuild
- GPU training is done outside Docker

### Directory structure expected

```
ml/
└── saved_models/
    ├── tfidf_vectorizer.pkl         ← shared TF-IDF vectorizer
    ├── lr_pipeline.pkl              ← Logistic Regression
    ├── svm_pipeline.pkl             ← Linear SVM (calibrated)
    ├── nb_pipeline.pkl              ← Naive Bayes
    ├── distilbert/
    │   └── v1.0.0/
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       └── tokenizer/
    └── eval/
        ├── lr_eval.json
        ├── svm_eval.json
        └── distilbert_eval.json
```

### Training baseline models (CPU — fast)

```bash
# Download the ISOT dataset first:
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Place Fake.csv and True.csv in ml/datasets/raw/

# From the project root:
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt

python scripts/prepare_dataset.py \
  --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv

python scripts/train_baseline.py
python scripts/evaluate_baseline.py
```

### Training DistilBERT (GPU recommended — 1–3 hours on CPU)

```bash
# Uncomment torch and transformers in backend/requirements.txt first
python scripts/train_transformer.py --epochs 3
python scripts/evaluate_transformer.py
```

### Docker volume mount

The compose file mounts `./ml` as read-only into the backend container:

```yaml
volumes:
  - ./ml:/app/ml:ro
```

The backend reads models from `ML_SAVED_MODELS_DIR=ml/saved_models`.
If no models are found, the API returns HTTP 503 for analysis requests
with a clear error message.

---

## 5. Starting Services

### Start everything (build first time)

```bash
docker compose up --build
```

### Start in background

```bash
docker compose up -d
```

### Start specific services

```bash
docker compose up postgres backend   # database + API only
```

### Development mode (hot reload)

```bash
# Uses docker-compose.dev.yml overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Stop services

```bash
docker compose down          # stop, keep volumes
docker compose down -v       # stop + delete database volume (data loss!)
```

### Rebuild after code changes

```bash
docker compose build backend   # rebuild only backend image
docker compose up -d backend   # restart with new image
```

### Run tests inside container

```bash
# Backend tests (no database needed — SQLite in-memory)
docker compose exec backend python -m pytest tests/ -q

# Integration check
docker compose exec backend python integration_check.py
```

---

## 6. Production Deployment

### Architecture overview

```
Internet
    │
    ▼
[Load Balancer / Reverse Proxy]  ← terminates TLS
    │                              (Caddy / Traefik / Nginx / Cloud LB)
    ▼
[Frontend — nginx container]     ← serves React SPA
    │ proxy /api/*
    ▼
[Backend — FastAPI container]    ← stateless, horizontally scalable
    │
    ▼
[PostgreSQL]                     ← managed DB service (recommended)
```

### HTTPS in production

**Option A — Caddy reverse proxy (easiest)**

```yaml
# Add to docker-compose.yml services:
caddy:
  image: caddy:2-alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./docker/caddy/Caddyfile:/etc/caddy/Caddyfile
    - caddy_data:/data
    - caddy_config:/config
  depends_on:
    - frontend
```

```
# docker/caddy/Caddyfile
fakenews.example.com {
    reverse_proxy frontend:80
}
```

Caddy automatically obtains and renews Let's Encrypt certificates.

**Option B — Traefik (Kubernetes-compatible)**

Add `traefik.enable=true` labels to your services and mount `acme.json`.

**Option C — Cloud load balancer**

On AWS/GCP/Azure, terminate TLS at the load balancer and forward HTTP
to the container. Set `COOKIE_SECURE=true` and `COOKIE_SAMESITE=none`
in your production environment variables.

### Production environment checklist

Before deploying to production:

- [ ] `SECRET_KEY` set to a unique 32+ char random value
- [ ] `POSTGRES_PASSWORD` set to a strong unique password
- [ ] `DEBUG=false`
- [ ] `APP_ENV=production`
- [ ] `COOKIE_SECURE=true` (requires HTTPS)
- [ ] `ALLOWED_ORIGINS` set to your actual domain
- [ ] NEWSAPI_KEY and other evidence providers configured
- [ ] ML models trained and mounted
- [ ] TLS termination configured
- [ ] Database using a managed service (RDS, Cloud SQL, etc.)
- [ ] Container registry set up for image storage

### Cloud platform deployment (example: DigitalOcean App Platform)

```bash
# Build and push images to a registry
docker build -t registry.example.com/fakenews-backend:latest ./backend
docker build -t registry.example.com/fakenews-frontend:latest ./frontend \
  --build-arg VITE_API_BASE_URL=https://api.fakenews.example.com/api/v1
docker push registry.example.com/fakenews-backend:latest
docker push registry.example.com/fakenews-frontend:latest
```

Configure the platform to:
1. Run `backend` and `frontend` containers from your registry
2. Set all environment variables as platform secrets (not in docker-compose)
3. Use a managed PostgreSQL instance
4. Point `POSTGRES_HOST` to the managed DB endpoint

---

## 7. Health Checks

### Endpoints

| Service | URL | Expected response |
|---------|-----|-------------------|
| Backend | `GET /api/v1/health` | `{"status": "ok", ...}` |
| Frontend (nginx) | `GET /nginx-health` | `200 healthy` |
| PostgreSQL | `pg_isready -U $POSTGRES_USER` | exit 0 |

### Check service health

```bash
# All services
docker compose ps

# Backend health endpoint
curl http://localhost:8000/api/v1/health | python -m json.tool

# PostgreSQL
docker compose exec postgres pg_isready -U $POSTGRES_USER -d $POSTGRES_DB

# Individual container health
docker inspect fakenews_backend | python -m json.tool | grep -A5 '"Health"'
```

### What the backend health endpoint checks

```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development",
  "components": {
    "database": { "status": "ok", "latency_ms": 1.2 },
    "ml_models": { "status": "ok", "message": "Available: logistic_regression, linear_svm, naive_bayes" }
  }
}
```

Status values: `ok` | `degraded` (some components down) | `unavailable` (critical failure)

---

## 8. Logs

### View logs

```bash
# All services (follow)
docker compose logs -f

# Single service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f postgres

# Last N lines
docker compose logs --tail=100 backend

# Since timestamp
docker compose logs --since="2024-01-01T00:00:00" backend
```

### Backend log format

```
2024-05-01 10:23:45  INFO     app.main  POST /api/v1/analyze/text
2024-05-01 10:23:47  INFO     app.services.analysis_service  ML verdict: FAKE (82.0%) from 3 models
2024-05-01 10:23:48  INFO     app.main  POST /api/v1/analyze/text → 200
```

### Log levels

Set `LOG_LEVEL` in `.env`:
- `DEBUG` — all SQL queries, all evidence provider calls (verbose)
- `INFO` — request/response, analysis outcomes (default)
- `WARNING` — only warnings and errors
- `ERROR` — only errors

### Persistent log storage (production)

In production, configure log shipping to a centralized service:

```yaml
# Example: ship logs to Loki (Grafana)
logging:
  driver: loki
  options:
    loki-url: "http://loki:3100/loki/api/v1/push"
    labels: "service"
```

---

## 9. Troubleshooting

### Backend fails to start — "PostgreSQL not ready"

```
[entrypoint] ERROR: PostgreSQL did not become ready after 30 attempts.
```

**Fix:**
```bash
# Check postgres container status
docker compose ps postgres
docker compose logs postgres

# Check credentials in .env match the postgres service env
# POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB must match exactly
```

### Backend fails — "No trained models are available"

```json
{"error": "MODEL_UNAVAILABLE", "message": "No trained models are available."}
```

**Fix:**
```bash
# Train models on the host, then restart backend
python scripts/train_baseline.py
docker compose restart backend

# Verify model files exist
ls ml/saved_models/*.pkl
```

### Alembic migration fails — "cannot run ALTER TYPE in transaction"

```
ERROR: cannot run ALTER TYPE ... ADD VALUE in a transaction block
```

**Fix:**
The migration `0002` uses `COMMIT` before `ALTER TYPE`. This requires
the connection to not be in autocommit=False mode at the start. If this
error occurs, check your `alembic/env.py` `run_migrations_online()` to
ensure it uses a fresh connection.

### Frontend shows "API Error" — cannot reach backend

**Check 1:** Backend health
```bash
curl http://localhost:8000/api/v1/health
```

**Check 2:** nginx proxy config
```bash
docker compose exec frontend cat /etc/nginx/conf.d/default.conf
# Should show: proxy_pass http://backend;
```

**Check 3:** Network connectivity
```bash
docker compose exec frontend wget -q -O- http://backend:8000/api/v1/health
```

### Cookie auth fails — "Authentication required" after login

This is usually a CORS/HTTPS mismatch.

- Development: `COOKIE_SECURE=false`, `ALLOWED_ORIGINS=http://localhost`
- Production: `COOKIE_SECURE=true`, `ALLOWED_ORIGINS=https://yourdomain.com`

Verify cookies are being sent with `withCredentials: true` (already set in
`frontend/src/lib/apiClient.ts`).

### Out of disk space — Docker volumes

```bash
# Show volume sizes
docker system df

# Remove unused images / containers / volumes (careful!)
docker system prune -f        # only stopped containers + dangling images
docker volume prune -f        # ⚠ removes unnamed volumes (data loss!)

# Remove only the database volume (requires data reload):
docker compose down
docker volume rm fake-news-detection_postgres_data
docker compose up -d
```

### Reset everything to a clean state

```bash
docker compose down -v          # stop + delete all volumes
docker system prune -af         # delete all stopped containers + images
rm -rf ml/saved_models/*        # remove trained models
cp .env.example .env            # reset env
# Then follow Section 1 again
```

---

## Quick Reference

```bash
# First-time setup
cp .env.example .env && nano .env
python scripts/train_baseline.py
docker compose up --build

# Day-to-day
docker compose up -d             # start
docker compose down              # stop
docker compose logs -f backend   # watch logs
docker compose exec backend alembic current   # check migrations

# Production build
docker compose build --no-cache
docker compose up -d

# Health check
curl http://localhost:8000/api/v1/health
```
