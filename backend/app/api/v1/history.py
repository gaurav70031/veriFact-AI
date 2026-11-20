# -*- coding: utf-8 -*-
"""
History endpoints.

GET /api/v1/analysis/{id}    - fetch one analysis (optional auth, ownership check)
GET /api/v1/history          - paginated list (requires auth, user sees only own)
GET /api/v1/history/export    - export history as CSV
GET /api/v1/history/compare    - compare two analyses side by side
GET /api/v1/report/pdf/{id}   - generate PDF verification report
GET /api/v1/report/research   - generate model evaluation research report

Authentication:
  /history       requires a valid JWT cookie - users see only their own analyses.
  /analysis/{id} works for both anonymous and authenticated callers:
    - authenticated: enforces ownership (cannot read another user's private analysis)
    - anonymous: can only read analyses that have no user_id (public)
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse, Response
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
        "Returns a paginated list of the authenticated user's analyses, "
        "newest first. Requires authentication.\n\n"
        "Filters:\n"
        "- verdict: FAKE | REAL | UNVERIFIED | MIXED\n"
        "- input_type: text | url | claim\n"
        "- search: text search across input and title\n"
        "- date_from / date_to: ISO date range\n"
        "- model_id: filter by model used\n"
        "- confidence_min / confidence_max: confidence range"
    ),
)
async def get_history(
    page:      int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    verdict:   Optional[str] = Query(default=None, pattern="^(FAKE|REAL|UNVERIFIED|MIXED)$"),
    input_type: Optional[str] = Query(default=None, pattern="^(text|url|query)$"),
    search:     Optional[str]  = Query(default=None, max_length=200),
    date_from:  Optional[str]  = Query(default=None),
    date_to:    Optional[str]  = Query(default=None),
    model_id:   Optional[str]  = Query(default=None),
    confidence_min: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    confidence_max: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    db:         AsyncSession     = Depends(get_db),
    current_user: User           = Depends(get_current_user),
) -> PaginatedHistory:
    return await history_service.get_history(
        db,
        page=page,
        page_size=page_size,
        verdict=verdict,
        input_type=input_type,
        user_id=current_user.id,
        search=search,
        date_from=date_from,
        date_to=date_to,
        model_id=model_id,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
    )


@router.get(
    "/history/export",
    summary="Export verification history as CSV",
    description="Exports the user's analysis history as a CSV. Supports same filters as /history.",
)
async def export_history(
    verdict:   Optional[str] = Query(default=None, pattern="^(FAKE|REAL|UNVERIFIED|MIXED)$"),
    input_type: Optional[str] = Query(default=None, pattern="^(text|url|query)$"),
    search:     Optional[str]  = Query(default=None, max_length=200),
    date_from:  Optional[str]  = Query(default=None),
    date_to:    Optional[str]  = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    import csv
    rows = await history_service.get_all_history(
        db, user_id=current_user.id,
        verdict=verdict, input_type=input_type, search=search,
        date_from=date_from, date_to=date_to, limit=10000,
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "input_type", "original_input",
        "article_title", "final_verdict", "final_confidence",
        "processing_time_ms", "source_url"])
    for r in rows:
        writer.writerow([
            r["id"],
            r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
            r["input_type"],
            (r["original_input"] or "")[:200],
            r.get("article_title") or "",
            r.get("final_verdict") or "",
            r.get("final_confidence") or "",
            r.get("processing_time_ms") or "",
            r.get("source_url") or "",
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="verifact_history_{current_user.username}.csv"'},
    )


@router.get(
    "/history/compare",
    summary="Compare two analyses side by side",
    description="Returns both analyses for side-by-side comparison. Ownership enforced.",
)
async def compare_analyses(
    id1: int = Query(..., description="First analysis ID."),
    id2: int = Query(..., description="Second analysis ID."),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> dict:
    user_id = current_user.id if current_user else None
    a1 = await history_service.get_analysis_by_id(db, id1, user_id=user_id)
    a2 = await history_service.get_analysis_by_id(db, id2, user_id=user_id)
    return {
        "analysis_1": a1.model_dump(),
        "analysis_2": a2.model_dump(),
        "comparison": {
            "same_verdict":    (a1.ml_verdict == a2.ml_verdict),
            "confidence_diff": abs((a1.ml_confidence or 0) - (a2.ml_confidence or 0)),
            "same_input_type": (a1.input_type == a2.input_type),
            "verdict_1":       a1.ml_verdict,
            "verdict_2":       a2.ml_verdict,
            "confidence_1":    a1.ml_confidence,
            "confidence_2":    a2.ml_confidence,
        },
    }


@router.get(
    "/report/pdf/{analysis_id}",
    summary="Generate PDF verification report",
    description=(
        "Generates a professional PDF report containing verification ID, "
        "submitted content, predicted classification, model info, confidence, "
        "evidence sources, XAI results, supporting/contradicting evidence, "
        "final assessment, and limitations."
    ),
)
async def generate_pdf_report(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> Response:
    user_id = current_user.id if current_user else None
    analysis = await history_service.get_analysis_by_id(db, analysis_id, user_id=user_id)
    from app.services.report_service import generate_verification_pdf
    pdf_bytes = await generate_verification_pdf(db, analysis, analysis_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="verifact_report_{analysis_id}.pdf"'},
    )


@router.get(
    "/report/research",
    summary="Generate model evaluation research report",
    description=(
        "Generates a comprehensive research report covering dataset details, "
        "preprocessing, model architecture, training config, evaluation metrics, "
        "comparative results, and research observations. All from stored data."
    ),
)
async def generate_research_report(
    db: AsyncSession = Depends(get_db),
) -> Response:
    from app.services.report_service import generate_research_pdf
    pdf_bytes = await generate_research_pdf(db)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="verifact_research_report.pdf"'},
    )


@router.get(
    "/report/paper",
    summary="Download research paper (Markdown)",
    description=(
        "Returns the full academic research paper as a Markdown document.\n\n"
        "Title: 'Fake News Detection Using NLP and Transformer-Based Models'\n\n"
        "Contains: abstract, literature review, methodology, results, limitations, references."
    ),
)
async def download_research_paper() -> Response:
    from pathlib import Path
    paper_path = Path(__file__).resolve().parents[4] / "docs" / "RESEARCH_PAPER.md"
    if not paper_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Research paper not found.")
    content = paper_path.read_bytes()
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="VeriFact_AI_Research_Paper.md"'},
    )
