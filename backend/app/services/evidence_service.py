"""
Evidence service — backend service layer.

Wraps the evidence module for use by the API layer.
Fetches stored evidence from PostgreSQL for a given analysis ID,
and provides a direct retrieval endpoint for fresh searches.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.evidence.evidence_service import retrieve_evidence as _retrieve
from app.models.analysis import Analysis
from app.models.claim import Claim
from app.models.evidence_source import EvidenceSource

logger = logging.getLogger(__name__)


async def get_evidence_for_analysis(
    db:          AsyncSession,
    analysis_id: int,
) -> dict:
    """
    Return stored evidence for a completed analysis from the database.
    Raises NotFoundError if analysis_id does not exist.
    """
    # Verify analysis exists
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise NotFoundError(f"Analysis {analysis_id} not found.")

    # Get claims for this analysis
    claims_result = await db.execute(
        select(Claim).where(Claim.analysis_id == analysis_id)
    )
    claims = claims_result.scalars().all()

    all_evidence = []
    for claim in claims:
        ev_result = await db.execute(
            select(EvidenceSource)
            .where(EvidenceSource.claim_id == claim.id)
            .order_by(EvidenceSource.rank)
        )
        sources = ev_result.scalars().all()
        for src in sources:
            all_evidence.append({
                "source_name":          src.source_name,
                "title":                src.title,
                "url":                  src.url,
                "snippet":              src.snippet,
                "source_type":          src.source_type.value,
                "published_at":         src.published_at.isoformat() if src.published_at else None,
                "retrieved_at":         src.retrieved_at.isoformat() if src.retrieved_at else None,
                "relevance_score":      src.relevance_score,
                "rank":                 src.rank,
                "relationship_to_claim": src.relationship_to_claim.value,
            })

    return {
        "analysis_id":  analysis_id,
        "evidence":     all_evidence,
        "total":        len(all_evidence),
        "status":       "found" if all_evidence else "INSUFFICIENT_EVIDENCE",
    }


async def search_evidence_fresh(
    claim:       str,
    max_results: int           = 10,
    from_days:   Optional[int] = 30,
) -> dict:
    """
    Perform a fresh evidence search (not from DB) for the given claim.
    Used by the direct /evidence/search endpoint.
    """
    result = await _retrieve(claim, max_results=max_results, from_days=from_days)
    return result.to_dict()
