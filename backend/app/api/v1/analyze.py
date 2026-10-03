"""
Analysis endpoints.

POST /api/v1/analyze/text
POST /api/v1/analyze/url
POST /api/v1/analyze/claim

All routes are thin:
  1. Validate input (Pydantic + security module).
  2. Delegate to analysis_service.
  3. Return structured response.

Error handling is done by the global exception handlers in main.py.
Routes never call ML models or DB directly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import sanitise_text, validate_url
from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.analyze import (
    AnalyzeTextRequest,
    AnalyzeUrlRequest,
    AnalyzeClaimRequest,
    AnalysisResponse,
)
from app.services import analysis_service

logger   = logging.getLogger(__name__)
router   = APIRouter(prefix="/analyze", tags=["Analysis"])
settings = get_settings()


@router.post(
    "/text",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse raw text or article body",
    description=(
        "Submit a news article body (and optional headline) for fake-news analysis.\n\n"
        "The text is processed by all available ML models.  "
        "A weighted ensemble verdict is returned along with per-model breakdowns.\n\n"
        "**Input limits:** 20–50,000 characters.  "
        "**Rate limit:** 10 requests / minute per IP (when rate-limiting middleware is enabled)."
    ),
)
async def analyse_text(
    body: AnalyzeTextRequest,
    db:   AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    text = sanitise_text(body.text, max_length=settings.max_text_length)
    return await analysis_service.analyse_text(db, text=text, title=body.title)


@router.post(
    "/url",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a public news article URL",
    description=(
        "Submit a public HTTPS URL.  The backend fetches the page, extracts "
        "clean article text using trafilatura / newspaper3k, then runs all "
        "available ML models.\n\n"
        "Returns HTTP 422 if the URL cannot be fetched or parsed.\n\n"
        "**Security:** private IP ranges and localhost are blocked (SSRF prevention)."
    ),
)
async def analyse_url(
    body: AnalyzeUrlRequest,
    db:   AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    url = validate_url(body.url, max_length=settings.max_url_length)
    return await analysis_service.analyse_url(db, url=url)


@router.post(
    "/claim",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a short factual claim",
    description=(
        "Submit a concise checkable claim (≤ 500 characters) with optional "
        "surrounding context.  Useful for targeted fact-checking of a single "
        "statement extracted from a larger article.\n\n"
        "Context, when provided, is prepended to the claim before inference "
        "to give the models additional signal."
    ),
)
async def analyse_claim(
    body: AnalyzeClaimRequest,
    db:   AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    claim   = sanitise_text(body.claim, max_length=500)
    context = sanitise_text(body.context, max_length=5000) if body.context else None
    return await analysis_service.analyse_claim(db, claim=claim, context=context)
