"""
History service.

Retrieves analysis records from PostgreSQL with pagination and filtering.
User ownership is enforced: authenticated users only see their own analyses.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.models.analysis import Analysis, AnalysisStatus, FinalVerdict, InputType
from app.models.claim import Claim
from app.models.evidence_source import EvidenceSource
from app.schemas.history import AnalysisListItem, PaginatedHistory
from app.schemas.analyze import (
    AnalysisResponse, ModelPrediction,
    ClaimResult, EvidenceSourceResult, EvidenceSummary,
)
from app.models.prediction import Prediction
from app.models.model_version import ModelVersion

logger = logging.getLogger(__name__)


async def get_analysis_by_id(
    db:          AsyncSession,
    analysis_id: int,
    user_id:     Optional[int] = None,
) -> AnalysisResponse:
    """
    Fetch a single analysis.

    Ownership check: if user_id is provided, the analysis must belong to
    that user (or must be anonymous with user_id=NULL).  If the analysis
    belongs to a different user, raise NotFoundError (same as missing —
    avoids leaking existence of other users' analyses).
    """
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise NotFoundError(f"Analysis {analysis_id} not found.")

    # Ownership: if a user_id is provided, they may only access their own
    # analyses (or public/anonymous ones).
    if user_id is not None and analysis.user_id is not None and analysis.user_id != user_id:
        raise NotFoundError(f"Analysis {analysis_id} not found.")

    # Fetch predictions
    pred_result = await db.execute(
        select(Prediction, ModelVersion.model_name)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.analysis_id == analysis_id)
    )
    model_predictions = [
        ModelPrediction(
            model_id=model_name,
            model_name=model_name.replace("_", " ").title(),
            label=pred.label.value,
            is_fake=pred.label.value == "FAKE",
            confidence=pred.confidence,
            fake_probability=pred.fake_probability,
            real_probability=pred.real_probability,
            inference_time_ms=0.0,
        )
        for pred, model_name in pred_result.all()
    ]

    # Fetch claims + evidence sources
    claims_result = await db.execute(
        select(Claim)
        .where(Claim.analysis_id == analysis_id)
        .order_by(Claim.position)
    )
    claims = claims_result.scalars().all()

    claim_responses: list[ClaimResult] = []
    total_sup = total_con = total_inc = total_norel = total_ev = 0

    for claim in claims:
        ev_result = await db.execute(
            select(EvidenceSource)
            .where(EvidenceSource.claim_id == claim.id)
            .order_by(EvidenceSource.rank)
        )
        sources = ev_result.scalars().all()

        sup   = sum(1 for s in sources if s.relationship_to_claim.value == "supporting")
        con   = sum(1 for s in sources if s.relationship_to_claim.value == "contradicting")
        inc   = sum(1 for s in sources if s.relationship_to_claim.value == "inconclusive")
        norel = sum(1 for s in sources if s.relationship_to_claim.value == "not_relevant")
        total_sup   += sup;  total_con  += con
        total_inc   += inc;  total_norel += norel
        total_ev    += len(sources)

        ev_source_results = [
            EvidenceSourceResult(
                source_name=s.source_name,
                title=s.title,
                url=s.url,
                snippet=s.snippet,
                source_type=s.source_type.value,
                published_at=s.published_at,
                retrieved_at=s.retrieved_at,
                relevance_score=s.relevance_score,
                comparison_score=getattr(s, "comparison_score", 0.0),
                rank=s.rank,
                relationship_to_claim=s.relationship_to_claim.value,
            )
            for s in sources
        ]

        claim_responses.append(ClaimResult(
            position=claim.position,
            claim_text=claim.claim_text,
            ml_verdict=claim.verdict.value if claim.verdict else None,
            ml_confidence=claim.confidence,
            evidence_verdict=claim.evidence_verdict.value if claim.evidence_verdict else None,
            evidence_explanation=claim.evidence_explanation,
            evidence_sources=ev_source_results,
            supporting_count=sup,
            contradicting_count=con,
            inconclusive_count=inc,
            not_relevant_count=norel,
            total_evidence=len(sources),
        ))

    ml_verdict_val = analysis.final_verdict.value if analysis.final_verdict else "UNVERIFIED"

    # Extract overall evidence verdict from summary if available
    ev_summary = EvidenceSummary(
        total_evidence=total_ev,
        supporting_count=total_sup,
        contradicting_count=total_con,
        inconclusive_count=total_inc,
        not_relevant_count=total_norel,
        providers_used=[],
        providers_failed=[],
        evidence_limitations=[],
    )

    return AnalysisResponse(
        id=analysis.id,
        input_type=analysis.input_type.value,
        original_input=analysis.original_input,
        source_url=analysis.source_url,
        article_title=analysis.article_title,
        ml_verdict=ml_verdict_val,
        ml_confidence=analysis.final_confidence,
        evidence_verdict=None,
        evidence_explanation=None,
        model_predictions=model_predictions,
        claims=claim_responses,
        evidence_summary=ev_summary,
        summary=analysis.summary,
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
    user_id:    Optional[int] = None,
) -> PaginatedHistory:
    """
    Fetch paginated analysis history.

    When user_id is provided (authenticated request) only that user's
    analyses are returned.  Anonymous analyses (user_id=NULL) are only
    visible to admins or when user_id is not specified.
    """
    filters = []

    # ── User ownership filter ────────────────────────────────────────────────
    if user_id is not None:
        filters.append(Analysis.user_id == user_id)

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
