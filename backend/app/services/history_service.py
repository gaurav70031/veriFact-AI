"""
History service.

Retrieves analysis records from PostgreSQL with pagination and filtering.
All queries are async via SQLAlchemy 2.x.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.models.analysis import Analysis, AnalysisStatus, FinalVerdict, InputType
from app.schemas.history import AnalysisListItem, PaginatedHistory
from app.schemas.analyze import AnalysisResponse, ModelPrediction
from app.models.prediction import Prediction
from app.models.model_version import ModelVersion

logger = logging.getLogger(__name__)


async def get_analysis_by_id(db: AsyncSession, analysis_id: int) -> AnalysisResponse:
    """
    Fetch a single analysis with all its model predictions.
    Raises NotFoundError if the record does not exist.
    """
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise NotFoundError(f"Analysis with id={analysis_id} not found.")

    # Fetch associated predictions
    pred_result = await db.execute(
        select(Prediction, ModelVersion.model_name)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.analysis_id == analysis_id)
    )
    pred_rows = pred_result.all()

    model_predictions = []
    for pred, model_name in pred_rows:
        model_predictions.append(ModelPrediction(
            model_id=model_name,
            model_name=model_name.replace("_", " ").title(),
            label=pred.label.value,
            is_fake=pred.label.value == "FAKE",
            confidence=pred.confidence,
            fake_probability=pred.fake_probability,
            real_probability=pred.real_probability,
            inference_time_ms=0.0,
        ))

    return AnalysisResponse(
        id=analysis.id,
        input_type=analysis.input_type.value,
        original_input=analysis.original_input,
        source_url=analysis.source_url,
        article_title=analysis.article_title,
        final_verdict=analysis.final_verdict.value if analysis.final_verdict else "UNVERIFIED",
        final_confidence=analysis.final_confidence,
        summary=analysis.summary,
        model_predictions=model_predictions,
        status=analysis.status.value,
        processing_time_ms=analysis.processing_time_ms,
        created_at=analysis.created_at,
    )


async def get_history(
    db:         AsyncSession,
    page:       int           = 1,
    page_size:  int           = 20,
    verdict:    Optional[str] = None,
    input_type: Optional[str] = None,
) -> PaginatedHistory:
    """
    Fetch paginated analysis history with optional filters.

    Parameters
    ----------
    page       : 1-based page number.
    page_size  : Records per page (max 100).
    verdict    : Optional filter: FAKE | REAL | UNVERIFIED | MIXED.
    input_type : Optional filter: text | url | claim.

    Returns
    -------
    PaginatedHistory with items, total, page, page_size, total_pages.
    """
    filters = []

    if verdict:
        try:
            filters.append(Analysis.final_verdict == FinalVerdict(verdict))
        except ValueError:
            pass

    if input_type:
        try:
            filters.append(Analysis.input_type == InputType(input_type))
        except ValueError:
            pass

    # Total count
    count_q = select(func.count(Analysis.id))
    if filters:
        count_q = count_q.where(and_(*filters))
    count_result = await db.execute(count_q)
    total = count_result.scalar_one()

    # Paginated rows
    offset = (page - 1) * page_size
    rows_q = (
        select(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    if filters:
        rows_q = rows_q.where(and_(*filters))

    rows_result = await db.execute(rows_q)
    analyses    = rows_result.scalars().all()

    items = [
        AnalysisListItem(
            id=a.id,
            input_type=a.input_type.value,
            original_input=a.original_input[:200],
            source_url=a.source_url,
            article_title=a.article_title,
            final_verdict=a.final_verdict.value if a.final_verdict else None,
            final_confidence=a.final_confidence,
            status=a.status.value,
            processing_time_ms=a.processing_time_ms,
            created_at=a.created_at,
        )
        for a in analyses
    ]

    total_pages = max(1, math.ceil(total / page_size))

    return PaginatedHistory(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
