"""
Evidence endpoints.

GET  /api/v1/evidence/{analysis_id}
    Retrieve stored evidence sources for a completed analysis from the DB.

POST /api/v1/evidence/search
    Perform a fresh evidence search for a given claim text.
    Uses real external providers (NewsAPI, GNews, RSS, Search).
    API keys must be set in environment variables — never exposed here.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.evidence_service import (
    get_evidence_for_analysis,
    search_evidence_fresh,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evidence", tags=["Evidence"])


class EvidenceSearchRequest(BaseModel):
    claim:       str = Field(
        ..., min_length=5, max_length=500,
        description="Factual claim to search evidence for.",
        examples=["The WHO declared COVID-19 a global pandemic in March 2020."],
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of evidence items to return.",
    )
    from_days:   Optional[int] = Field(
        default=30, ge=1, le=365,
        description="Restrict results to articles published within this many days.",
    )


@router.get(
    "/{analysis_id}",
    status_code=status.HTTP_200_OK,
    summary="Get stored evidence for an analysis",
    description=(
        "Returns evidence sources that were retrieved and stored during a "
        "completed analysis.  Evidence is fetched from the PostgreSQL database — "
        "no external API calls are made.\n\n"
        "Returns HTTP 404 if the analysis does not exist."
    ),
)
async def get_evidence(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    return await get_evidence_for_analysis(db, analysis_id)


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Fresh evidence search for a claim",
    description=(
        "Perform a real-time evidence search using configured news providers "
        "(NewsAPI, GNews, RSS feeds, Search API).\n\n"
        "**Production note:** This endpoint calls real external APIs. "
        "Configure provider API keys in environment variables.\n\n"
        "If no evidence is found, `status` is `INSUFFICIENT_EVIDENCE`. "
        "Absence of evidence does not indicate a claim is false.\n\n"
        "Results are ranked by relevance (keyword overlap + recency)."
    ),
)
async def search_evidence(
    body: EvidenceSearchRequest,
) -> dict:
    return await search_evidence_fresh(
        claim=body.claim,
        max_results=body.max_results,
        from_days=body.from_days,
    )
