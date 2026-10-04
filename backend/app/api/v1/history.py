"""
History endpoints.

GET /api/v1/analysis/{id}  — fetch one analysis (optional auth, ownership check)
GET /api/v1/history        — paginated list (requires auth, user sees only own)

Authentication:
  /history       requires a valid JWT cookie — users see only their own analyses.
  /analysis/{id} works for both anonymous and authenticated callers:
    - authenticated: enforces ownership (cannot read another user's private analysis)
    - anonymous: can only read analyses that have no user_id (public)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user, get_optional_user
from app.db.session import get_db
from app.models.user import User
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
        "Retrieve the full result of a completed analysis.\n\n"
        "**Ownership:** authenticated users may only access their own analyses "
        "or analyses with no owner (anonymous submissions). "
        "Accessing another user's analysis returns HTTP 404 — not 403 — to "
        "prevent leaking the existence of other users' analyses.\n\n"
        "Returns HTTP 404 if the ID does not exist or is not accessible."
    ),
)
async def get_analysis(
    analysis_id:  int,
    db:           AsyncSession       = Depends(get_db),
    current_user: Optional[User]     = Depends(get_optional_user),
) -> AnalysisResponse:
    user_id = current_user.id if current_user else None
    return await history_service.get_analysis_by_id(db, analysis_id, user_id=user_id)


@router.get(
    "/history",
    response_model=PaginatedHistory,
    status_code=status.HTTP_200_OK,
    summary="Paginated personal analysis history",
    description=(
        "Returns a paginated list of the **authenticated user's** analyses, "
        "newest first.\n\n"
        "**Requires authentication.** Returns HTTP 401 when not logged in.\n\n"
        "Users can only see their own analyses — not other users'.\n\n"
        "**Filters:**\n"
        "- `verdict`: `FAKE` | `REAL` | `UNVERIFIED` | `MIXED`\n"
        "- `input_type`: `text` | `url` | `claim`"
    ),
)
async def get_history(
    page: int = Query(default=1, ge=1, description="Page number (1-based)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    verdict: Optional[str] = Query(
        default=None,
        description="Filter by ML verdict.",
        pattern="^(FAKE|REAL|UNVERIFIED|MIXED)$",
    ),
    input_type: Optional[str] = Query(
        default=None,
        description="Filter by input type.",
        pattern="^(text|url|claim)$",
    ),
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
) -> PaginatedHistory:
    return await history_service.get_history(
        db,
        page=page,
        page_size=page_size,
        verdict=verdict,
        input_type=input_type,
        user_id=current_user.id,     # enforce ownership
    )
