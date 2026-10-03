"""
FastAPI application entry point.

Startup sequence (lifespan)
---------------------------
1. Configure structured logging.
2. Create all DB tables if they don't exist (idempotent).
3. Initialise the ML model registry (load artefacts into memory).
   - Baseline models (LR/SVM/NB) are required for the app to be fully healthy.
   - Transformer (DistilBERT) is optional; app starts without it.
4. Mount API router.

Shutdown sequence
-----------------
1. Dispose async DB engine (closes connection pool cleanly).

Middleware
----------
- CORS with configurable origin allowlist.
- Request body size limit (1 MB by default).
- Structured error handlers for AppError hierarchy.

API Documentation
-----------------
  Swagger UI : http://localhost:8000/docs
  ReDoc      : http://localhost:8000/redoc
  OpenAPI    : http://localhost:8000/openapi.json
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.errors import (
    AppError,
    app_error_handler,
    unhandled_error_handler,
)
from app.api.v1.router import api_router
from app.db.session import create_all_tables, dispose_engine
from app.ml.model_registry import get_registry

settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.app_env)

    # Create DB tables (no-op if already exist)
    try:
        await create_all_tables()
        logger.info("Database tables verified.")
    except Exception as exc:
        logger.error("Database setup failed: %s", exc)
        # Don't crash — app can still serve if DB becomes available later

    # Load ML models
    registry = get_registry()
    try:
        registry.initialise(require_transformer=False)
    except Exception as exc:
        logger.error("Model registry initialisation error: %s", exc)

    yield   # ← application is running here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — disposing DB engine.")
    await dispose_engine()


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Real-time fake news detection using TF-IDF baseline models "
            "and fine-tuned DistilBERT transformer.\n\n"
            "## Input types supported\n"
            "- **Text** — raw article body or claim\n"
            "- **URL** — public news article URL (extracted server-side)\n"
            "- **Claim** — short factual statement with optional context\n\n"
            "## Models\n"
            "- TF-IDF + Logistic Regression\n"
            "- TF-IDF + Linear SVM (calibrated)\n"
            "- TF-IDF + Naive Bayes\n"
            "- DistilBERT (fine-tuned, with long-article chunking)\n\n"
            "All predictions come from real trained model artefacts. "
            "No hardcoded or simulated results."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request body size limit
    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_body:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "REQUEST_TOO_LARGE",
                    "message": (
                        f"Request body exceeds the maximum allowed size "
                        f"of {settings.max_request_body // 1024} KB."
                    ),
                },
            )
        return await call_next(request)

    # Request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("%s %s", request.method, request.url.path)
        response = await call_next(request)
        logger.info("%s %s → %d", request.method, request.url.path, response.status_code)
        return response

    # ── Exception handlers ────────────────────────────────────────────────────

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    # ── Routes ────────────────────────────────────────────────────────────────

    app.include_router(api_router)

    @app.get("/", tags=["Root"], include_in_schema=False)
    async def root():
        return {
            "name":    settings.app_name,
            "version": settings.app_version,
            "docs":    "/docs",
            "health":  "/api/v1/health",
        }

    return app


# ── Application instance ──────────────────────────────────────────────────────

app = create_app()


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
