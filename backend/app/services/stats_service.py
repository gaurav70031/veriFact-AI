"""
Stats service.

Computes real aggregate statistics from the PostgreSQL database.
No hardcoded values — every number is derived from actual DB records.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis, AnalysisStatus, FinalVerdict
from app.models.model_version import ModelVersion
from app.schemas.models import ModelInfo, ModelPerformance, ModelPerformanceList
from app.schemas.stats import StatsResponse, VerdictCounts

logger = logging.getLogger(__name__)


async def get_stats(db: AsyncSession) -> StatsResponse:
    """
    Compute aggregate analysis statistics from actual database records.
    Returns real counts, percentages, and averages — nothing hardcoded.
    Falls back to zero values if any query fails.
    """
    try:
        return await _get_stats_inner(db)
    except Exception as exc:
        logger.error("Stats query failed: %s", exc, exc_info=True)
        return StatsResponse(
            total_analyses=0, completed=0, failed=0, pending=0,
            verdict_counts=VerdictCounts(FAKE=0, REAL=0, UNVERIFIED=0, MIXED=0),
            fake_percentage=0.0, real_percentage=0.0,
            avg_confidence=0.0, avg_processing_ms=0.0,
        )


async def _get_stats_inner(db: AsyncSession) -> StatsResponse:
    # Total analyses
    total_result = await db.execute(select(func.count(Analysis.id)))
    total = total_result.scalar_one() or 0

    # Completed
    completed_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.status == AnalysisStatus.COMPLETED
        )
    )
    completed = completed_result.scalar_one() or 0

    # Failed
    failed_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.status == AnalysisStatus.FAILED
        )
    )
    failed = failed_result.scalar_one() or 0

    # Pending
    pending_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.status.in_([AnalysisStatus.PENDING, AnalysisStatus.PROCESSING])
        )
    )
    pending = pending_result.scalar_one() or 0

    # Verdict counts (only completed analyses)
    verdict_result = await db.execute(
        select(Analysis.final_verdict, func.count(Analysis.id))
        .where(
            and_(
                Analysis.status == AnalysisStatus.COMPLETED,
                Analysis.final_verdict.isnot(None),
            )
        )
        .group_by(Analysis.final_verdict)
    )
    verdict_rows = verdict_result.all()
    verdict_map: dict[str, int] = {}
    for row in verdict_rows:
        if row[0] is not None:
            key = row[0].value if hasattr(row[0], "value") else str(row[0])
            verdict_map[key] = row[1]

    fake_count       = verdict_map.get("FAKE",       0)
    real_count       = verdict_map.get("REAL",       0)
    unverified_count = verdict_map.get("UNVERIFIED", 0)
    mixed_count      = verdict_map.get("MIXED",      0)

    completed_nz  = max(completed, 1)
    fake_pct      = round(fake_count / completed_nz * 100, 2)
    real_pct      = round(real_count / completed_nz * 100, 2)

    # Average confidence (completed with a verdict)
    conf_result = await db.execute(
        select(func.avg(Analysis.final_confidence)).where(
            and_(
                Analysis.status == AnalysisStatus.COMPLETED,
                Analysis.final_confidence.isnot(None),
            )
        )
    )
    avg_conf = conf_result.scalar_one() or 0.0

    # Average processing time (completed)
    pt_result = await db.execute(
        select(func.avg(Analysis.processing_time_ms)).where(
            and_(
                Analysis.status == AnalysisStatus.COMPLETED,
                Analysis.processing_time_ms.isnot(None),
            )
        )
    )
    avg_processing_ms = pt_result.scalar_one() or 0.0

    return StatsResponse(
        total_analyses=total,
        completed=completed,
        failed=failed,
        pending=pending,
        verdict_counts=VerdictCounts(
            FAKE=fake_count,
            REAL=real_count,
            UNVERIFIED=unverified_count,
            MIXED=mixed_count,
        ),
        fake_percentage=fake_pct,
        real_percentage=real_pct,
        avg_confidence=round(float(avg_conf), 4),
        avg_processing_ms=round(float(avg_processing_ms), 1),
    )


async def get_model_versions(db: AsyncSession) -> list[ModelInfo]:
    """Return all registered (active) model versions from the DB."""
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.created_at.desc())
    )
    versions = result.scalars().all()

    return [
        ModelInfo(
            id=mv.id,
            model_name=mv.model_name,
            version=mv.version,
            model_type=mv.model_type.value,
            algorithm=mv.algorithm.value,
            is_active=mv.is_active,
            description=mv.description,
            dataset_name=mv.dataset_name,
            training_samples=mv.training_samples,
            created_at=mv.created_at,
        )
        for mv in versions
    ]


async def get_model_performance(db: AsyncSession) -> ModelPerformanceList:
    """Return training-time evaluation metrics for all registered models."""
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.model_name)
    )
    versions = result.scalars().all()

    perfs = [
        ModelPerformance(
            model_name=mv.model_name,
            version=mv.version,
            algorithm=mv.algorithm.value,
            accuracy=mv.accuracy,
            precision=mv.precision,
            recall=mv.recall,
            f1_score=mv.f1_score,
            roc_auc=mv.roc_auc,
            test_samples=mv.test_samples,
        )
        for mv in versions
    ]
    return ModelPerformanceList(models=perfs)
