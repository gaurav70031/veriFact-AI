"""
Health check endpoint.

GET /api/v1/health

Checks:
  - Database connectivity (async ping)
  - ML model registry status (are artefacts loaded?)
  - Application version and environment

Returns HTTP 200 when all components are healthy.
Returns HTTP 503 when any critical component is degraded.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.ml.model_registry import get_registry
from app.schemas.health import ComponentStatus, HealthResponse

logger   = logging.getLogger(__name__)
router   = APIRouter(tags=["Health"])
settings = get_settings()


async def _check_database(db: AsyncSession) -> ComponentStatus:
    t0 = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
        latency = (time.perf_counter() - t0) * 1000
        return ComponentStatus(status="ok", latency_ms=round(latency, 2))
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return ComponentStatus(status="unavailable", message=str(exc))


def _check_ml_models() -> ComponentStatus:
    try:
        registry = get_registry()
        reg_status = registry.status()
        available  = reg_status["available_models"]
        if not available:
            return ComponentStatus(
                status="unavailable",
                message="No trained models loaded. Run training scripts first.",
            )
        return ComponentStatus(
            status="ok",
            message=f"Available: {', '.join(available)}",
        )
    except Exception as exc:
        return ComponentStatus(status="unavailable", message=str(exc))


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application health check",
    description=(
        "Returns the health status of all application components:\n\n"
        "- **database**: PostgreSQL connectivity and query latency\n"
        "- **ml_models**: loaded model artefacts\n\n"
        "Returns HTTP 200 when fully healthy, HTTP 503 when degraded."
    ),
)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    db_status  = await _check_database(db)
    ml_status  = _check_ml_models()

    components = {
        "database":  db_status,
        "ml_models": ml_status,
    }

    overall = "ok"
    if db_status.status == "unavailable":
        overall = "unavailable"
    elif ml_status.status == "unavailable":
        overall = "degraded"

    body = HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.app_env,
        components=components,
    )

    http_status = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if overall == "unavailable"
        else status.HTTP_200_OK
    )

    return JSONResponse(
        status_code=http_status,
        content=body.model_dump(),
    )
