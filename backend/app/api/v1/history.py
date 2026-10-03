"""
History endpoints.

GET /api/v1/analysis/{id}   — fetch one analysis by primary key
GET /api/v1/history         — paginated list with optional filters
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.analyze import AnalysisResponse
from app.schemas.history import PaginatedHistory
from app.services import history_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["History"])


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch a single analysis by ID",
    description=(
        "Retrieve the full result of a previously completed analysis, "
        "including per-model predictions.\n\n"
        "Returns HTTP 404 if the ID does not exist."
    ),
)
async def get_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    return await history_service.get_analysis_by_id(db, analysis_id)


@router.get(
    "/history",
    response_model=PaginatedHistory,
    status_code=status.HTTP_200_OK,
    summary="Paginated analysis history",
    description=(
        "Returns a paginated list of past analyses, newest first.\n\n"
        "**Filters:**\n"
        "- `verdict`: `FAKE` | `REAL` | `UNVERIFIED` | `MIXED`\n"
        "- `input_type`: `text` | `url` | `claim`\n\n"
        "**Pagination:** use `page` and `page_size` query parameters."
    ),
)
async def get_history(
    page: int = Query(default=1, ge=1, description="Page number (1-based)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    verdict: Optional[str] = Query(
        default=None,
        description="Filter by verdict.",
        pattern="^(FAKE|REAL|UNVERIFIED|MIXED)$",
    ),
    input_type: Optional[str] = Query(
        default=None,
        description="Filter by input type.",
        pattern="^(text|url|claim)$",
    ),
    db: AsyncSession = Depends(get_db),
) -> PaginatedHistory:
    return await history_service.get_history(
        db,
        page=page,
        page_size=page_size,
        verdict=verdict,
        input_type=input_type,
    )
