"""
Analysis endpoints.

POST /api/v1/analyze/text
POST /api/v1/analyze/url
POST /api/v1/analyze/claim

Authentication: optional.
  - When a valid JWT cookie is present, analyses are linked to the user.
  - When anonymous, analyses are stored with user_id = NULL.
  - Public analysis remains accessible without login.

All routes are thin:
  1. Validate input.
  2. Resolve optional user from cookie.
  3. Delegate to analysis_service with optional user_id.
  4. Return structured response.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security  import sanitise_text, validate_url
from app.core.config    import get_settings
from app.core.dependencies import get_optional_user
from app.db.session     import get_db
from app.models.user    import User
from app.schemas.analyze import (
    AnalyzeTextRequest, AnalyzeUrlRequest,
    AnalyzeClaimRequest, AnalysisResponse,
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
        "Submit a news article body for fake-news analysis.\n\n"
        "Works without authentication.  When a valid session cookie is present "
        "the analysis is linked to your account and visible in your history."
    ),
)
async def analyse_text(
    body:         AnalyzeTextRequest,
    db:           AsyncSession       = Depends(get_db),
    current_user: Optional[User]     = Depends(get_optional_user),
) -> AnalysisResponse:
    text = sanitise_text(body.text, max_length=settings.max_text_length)
    return await analysis_service.analyse_text(
        db, text=text, title=body.title,
        user_id=current_user.id if current_user else None,
    )


@router.post(
    "/url",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a public news article URL",
    description=(
        "Submit a public HTTPS URL.  The backend fetches and extracts the article, "
        "then runs all available ML models.\n\n"
        "Private IP ranges and localhost are blocked (SSRF prevention)."
    ),
)
async def analyse_url(
    body:         AnalyzeUrlRequest,
    db:           AsyncSession   = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> AnalysisResponse:
    url = validate_url(body.url, max_length=settings.max_url_length)
    return await analysis_service.analyse_url(
        db, url=url,
        user_id=current_user.id if current_user else None,
    )


@router.post(
    "/claim",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a short factual claim",
    description=(
        "Submit a concise checkable claim (≤ 500 characters) with optional context."
    ),
)
async def analyse_claim(
    body:         AnalyzeClaimRequest,
    db:           AsyncSession       = Depends(get_db),
    current_user: Optional[User]     = Depends(get_optional_user),
) -> AnalysisResponse:
    claim   = sanitise_text(body.claim, max_length=500)
    context = sanitise_text(body.context, max_length=5000) if body.context else None
    return await analysis_service.analyse_claim(
        db, claim=claim, context=context,
        user_id=current_user.id if current_user else None,
    )
