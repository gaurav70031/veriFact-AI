# VeriFact AI — Backend Architecture

## Overview

Production-quality FastAPI backend for fake news detection with:
- 49 documented API endpoints
- PostgreSQL async database
- JWT cookie authentication
- Rate limiting (production)
- SSRF-protected URL fetching
- Model loading and caching
- Background training jobs
- PDF report generation

## Quick Start

```bash
cd backend
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Install PyTorch for your platform:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy .env.example to .env and configure:
cp ../.env.example ../.env

# Run:
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

## Architecture

```
backend/app/
├── main.py                    # FastAPI app factory, middleware, lifespan
├── api/v1/                     # All REST endpoints
│   ├── router.py              #   Route registration (49 endpoints)
│   ├── analyze.py             #   POST /analyze/text|url|claim
│   ├── verify.py              #   POST /verify, GET /verify/providers
│   ├── history.py             #   GET /history, /analysis/{id}, /export, /compare, /report/pdf, /report/research
│   ├── models.py              #   GET /models, /model-performance, /stats
│   ├── lab.py                 #   GET/POST /lab/* (training, evaluation, comparison)
│   ├── analytics.py           #   GET /analytics/* (pattern analysis)
│   ├── dataset.py             #   GET/POST /datasets/* (catalog, upload, preprocess)
│   ├── evidence.py            #   GET /evidence/{id}, POST /evidence/search
│   ├── explanation.py         #   GET /explanation/{id}/* (XAI reports)
│   ├── news.py                #   GET /news/search (live news proxy)
│   ├── health.py              #   GET /health
│   └── auth.py                #   POST /auth/register|login|logout, GET /auth/me
├── core/                      # Cross-cutting concerns
│   ├── config.py              #   Pydantic Settings (env vars)
│   ├── security.py            #   SSRF validation, text sanitisation
│   ├── auth_utils.py          #   JWT creation, bcrypt hashing
│   ├── dependencies.py       #   FastAPI auth dependencies
│   ├── errors.py              #   AppError hierarchy + handlers
│   ├── logging.py             #   Structured logging
│   └── rate_limiter.py        #   Sliding-window rate limiter middleware
├── services/                   # Business logic
│   ├── analysis_service.py   #   Full pipeline: ML + evidence + XAI + verdict
│   ├── analytics_service.py   #   Pattern analysis orchestrator
│   ├── history_service.py    #   CRUD + filtering + export
│   ├── lab_service.py         #   Training jobs, evaluation, report generation
│   ├── dataset_service.py     #   Dataset catalog, upload, preprocessing
│   ├── evidence_service.py    #   Evidence retrieval orchestration
│   ├── report_service.py      #   PDF generation (reportlab)
│   ├── stats_service.py       #   Aggregate statistics from DB
│   └── user_service.py        #   User registration, authentication
├── analysis/                   # NLP analysis modules
│   ├── claim_extractor.py     #   Sentence → claim extraction
│   ├── evidence_comparator.py #   Claim vs evidence comparison
│   ├── verdict_engine.py      #   Evidence assessment rules
│   ├── explainer.py           #   LIME + attention attribution
│   ├── shap_explainer.py      #   SHAP LinearExplainer + IG
│   └── pattern_engine.py      #   Word freq, n-grams, LDA, sentiment, clustering
├── evidence/                   # Evidence retrieval providers
│   ├── evidence_service.py    #   Orchestrator: query → providers → dedup → rank
│   ├── base_provider.py        #   Abstract provider + error hierarchy
│   ├── query_extractor.py     #   Claim → search query
│   ├── schema.py               #   EvidenceItem, EvidenceResult
│   └── providers/             #   Concrete adapters
│       ├── tavily_provider.py #   Tavily (AI search, tier-aware)
│       ├── newsapi_provider.py
│       ├── gnews_provider.py
│       ├── rss_provider.py
│       └── search_provider.py  #   SerpAPI / Brave
├── extraction/                 # Article extraction
│   └── article_extractor.py   #   trafilatura + newspaper3k, SSRF-safe
├── ml/                         # ML model registry
│   └── model_registry.py      #   Singleton cache, lazy load, predict()
├── models/                     # SQLAlchemy ORM models
│   ├── analysis.py, claim.py, evidence_source.py
│   ├── model_version.py, model_run.py, prediction.py
│   └── user.py
├── schemas/                    # Pydantic request/response schemas
│   ├── analyze.py, auth.py, dataset.py, health.py
│   ├── history.py, lab.py, models.py, stats.py
├── db/                         # Database layer
│   ├── base.py                #   Declarative base + timestamp mixin
│   └── session.py             #   Async engine + session factory
└── alembic/                    # Database migrations
    └── versions/0001-0004
```

## API Endpoint Reference

### Analysis (POST)
| Endpoint | Auth | Description |
|---|---|---|
| `POST /api/v1/analyze/text` | Optional | Analyze article text |
| `POST /api/v1/analyze/url` | Optional | Extract + analyze URL |
| `POST /api/v1/analyze/claim` | Optional | Analyze short claim |
| `POST /api/v1/verify` | None | Evidence-only verification (no ML) |

### History & Reports (GET)
| Endpoint | Auth | Description |
|---|---|---|
| `GET /api/v1/history` | Required | Paginated, filterable list |
| `GET /api/v1/analysis/{id}` | Optional | Full analysis result |
| `GET /api/v1/history/export` | Required | CSV export |
| `GET /api/v1/history/compare` | Optional | Compare two analyses |
| `GET /api/v1/report/pdf/{id}` | Optional | PDF verification report |
| `GET /api/v1/report/research` | None | PDF research report |

### Model Laboratory
| Endpoint | Auth | Description |
|---|---|---|
| `GET /api/v1/lab/models` | None | All models with metrics |
| `GET /api/v1/lab/models/{id}` | None | Single model detail |
| `GET /api/v1/lab/models/{id}/curves` | None | Loss curves |
| `GET /api/v1/lab/models/{id}/confusion` | None | Confusion matrix |
| `GET /api/v1/lab/models/{id}/roc` | None | ROC curve data |
| `GET /api/v1/lab/comparison` | None | Side-by-side comparison |
| `GET /api/v1/lab/report` | None | Research report (JSON) |
| `POST /api/v1/lab/train` | None | Trigger training job |
| `GET /api/v1/lab/train/{job_id}` | None | Job status + logs |

### Analytics
| Endpoint | Auth | Description |
|---|---|---|
| `GET /api/v1/analytics/patterns` | None | Full pattern analysis |
| `GET /api/v1/analytics/wordfreq` | None | Word frequency |
| `GET /api/v1/analytics/temporal` | None | Date trends |
| `GET /api/v1/analytics/sentiment` | None | VADER sentiment |
| `GET /api/v1/analytics/topics` | None | LDA topics |
| `GET /api/v1/analytics/insights` | None | Research insights |

### Datasets
| Endpoint | Auth | Description |
|---|---|---|
| `GET /api/v1/datasets` | None | Dataset catalog |
| `GET /api/v1/datasets/{name}/stats` | None | Split statistics |
| `GET /api/v1/datasets/{name}/analytics` | None | Word freq, quality |
| `GET /api/v1/datasets/{name}/samples` | None | Sample rows |
| `GET /api/v1/datasets/{name}/audit` | None | Preprocessing log |
| `POST /api/v1/datasets/upload` | None | Upload CSV |
| `POST /api/v1/datasets/preprocess` | None | Run pipeline |
| `GET /api/v1/datasets/{name}/export/{split}` | None | Download processed |

### Other
| Endpoint | Auth | Description |
|---|---|---|
| `GET /api/v1/health` | None | System health |
| `GET /api/v1/stats` | None | Aggregate statistics |
| `GET /api/v1/models` | None | Registered models |
| `GET /api/v1/model-performance` | None | Model metrics |
| `GET /api/v1/news/search` | None | Live news proxy |
| `GET /api/v1/evidence/{id}` | None | Stored evidence |
| `POST /api/v1/evidence/search` | None | Fresh evidence search |
| `GET /api/v1/explanation/{id}` | None | XAI attributions |
| `GET /api/v1/explanation/{id}/report` | None | XAI report JSON |
| `GET /api/v1/verify/providers` | None | Provider status |
| `POST /api/v1/auth/register` | None | Create account |
| `POST /api/v1/auth/login` | None | Login |
| `POST /api/v1/auth/logout` | None | Logout |
| `GET /api/v1/auth/me` | Required | Current user |

## Security Features

### Authentication
- JWT tokens in httpOnly cookies (XSS-resistant)
- bcrypt password hashing (12 rounds)
- Token expiry (30 min default, configurable)
- SameSite=lax cookie policy (CSRF mitigation)
- Secure=True in production (HTTPS only)

### Input Validation
- Pydantic schema validation on all request bodies
- Text length limits (50,000 chars max)
- URL scheme validation (http/https only)
- Text sanitisation (strip whitespace, enforce max length)

### SSRF Prevention
- Private IP range blocking (RFC 1918, loopback, link-local)
- Blocked hostnames (localhost, metadata endpoints)
- Port allowlist (80, 443 only)
- Redirect chain validation (each hop re-validated)
- Response size cap (5 MB max)

### Rate Limiting
- Sliding window per IP (60 req/min default)
- Burst allowance (10 in first second)
- Configurable via env vars: `RATE_LIMIT_ENABLED`, `RATE_LIMIT_PER_MINUTE`
- Health/docs endpoints exempt
- Returns HTTP 429 with `Retry-After` header

### Request Size Limits
- General requests: 1 MB max
- Dataset uploads: 200 MB max
- Enforced before request body is read

### Error Handling
- Custom `AppError` hierarchy → structured JSON responses
- Unhandled exceptions → generic 500 (no internal details leaked)
- No stack traces or secrets in error responses

## Performance Features

### Model Caching
- Singleton `ModelRegistry` loads models once at startup
- Pipelines cached in module-level dicts (no per-request reload)
- `load_all_pipelines()` called during FastAPI lifespan startup
- Transformer model loaded lazily (only when first needed)

### Async Inference
- ML inference runs in `asyncio.run_in_executor()` thread pool
- Event loop stays responsive during long inference
- Evidence retrieval runs concurrently via `asyncio.gather()`
- Multiple providers queried in parallel

### Background Training
- `POST /lab/train` starts an async background task
- Training runs in thread pool (doesn't block event loop)
- Progress tracked via in-memory job store
- `GET /lab/train/{job_id}` polls for status + log lines
- DB updated automatically after training + evaluation completes

### Database
- Async SQLAlchemy with asyncpg
- Connection pool (2 base + 3 overflow = 5 max)
- Pool pre-ping (automatic reconnect on stale connections)
- Pool recycle every 300 seconds

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | development | development/staging/production |
| `SECRET_KEY` | change-me | JWT signing key (must be random) |
| `BACKEND_PORT` | 8000 | Server port |
| `POSTGRES_*` | localhost | Database connection |
| `NEWSAPI_KEY` | | NewsAPI key |
| `GNEWS_API_KEY` | | GNews key |
| `TAVILY_API_KEY` | | Tavily key |
| `SERPAPI_KEY` | | SerpAPI key |
| `BRAVE_SEARCH_KEY` | | Brave Search key |
| `RATE_LIMIT_ENABLED` | auto | true/false (auto = prod only) |
| `RATE_LIMIT_PER_MINUTE` | 60 | Requests per minute per IP |
| `RATE_LIMIT_BURST` | 10 | Burst in first second |
| `ML_SAVED_MODELS_DIR` | ml/saved_models | Model artifact path |
| `TRANSFORMER_VERSION` | 1.0.0 | Which model version to load |

## Database Migrations

```bash
cd backend
python -m alembic upgrade head      # Apply all migrations
python -m alembic revision --autogenerate -m "description"  # Create new
python -m alembic downgrade -1       # Roll back one
```
