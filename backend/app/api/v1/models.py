"""
Model management endpoints.

GET /api/v1/models             — list registered model versions
GET /api/v1/model-performance  — training-time evaluation metrics
GET /api/v1/stats              — aggregate analysis statistics
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.models import ModelInfo, ModelPerformanceList
from app.schemas.stats  import StatsResponse
from app.services       import stats_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=List[ModelInfo],
    status_code=status.HTTP_200_OK,
    summary="List registered ML models",
    description=(
        "Returns metadata for every active model version registered in the database, "
        "including training dataset, algorithm, and version string."
    ),
)
async def list_models(
    db: AsyncSession = Depends(get_db),
) -> list[ModelInfo]:
    return await stats_service.get_model_versions(db)


@router.get(
    "/model-performance",
    response_model=ModelPerformanceList,
    status_code=status.HTTP_200_OK,
    summary="Model evaluation metrics",
    description=(
        "Returns training-time evaluation metrics (accuracy, F1, ROC-AUC) "
        "for each registered model, as recorded when the model was trained.\n\n"
        "These are real metrics computed on the held-out test set — "
        "not estimates or placeholders."
    ),
)
async def model_performance(
    db: AsyncSession = Depends(get_db),
) -> ModelPerformanceList:
    return await stats_service.get_model_performance(db)


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Aggregate analysis statistics",
    description=(
        "Returns real aggregate statistics computed from the PostgreSQL database:\n\n"
        "- Total analyses (completed / failed / pending)\n"
        "- Verdict distribution (FAKE / REAL / UNVERIFIED / MIXED)\n"
        "- Average confidence and average processing time"
    ),
)
async def get_stats(
    db: AsyncSession = Depends(get_db),
) -> StatsResponse:
    import traceback
    try:
        return await stats_service.get_stats(db)
    except Exception as e:
        logger.error("Stats endpoint error: %s\n%s", e, traceback.format_exc())
        raise
