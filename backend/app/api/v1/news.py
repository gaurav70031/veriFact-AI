"""
Live news search endpoint.

GET /api/v1/news/search

Query parameters
----------------
q          : Search query / claim text (required, 3–500 chars)
from_days  : Restrict to articles published within N days (default 30, max 365)
max_results: Maximum articles to return (default 20, max 50)

Design
------
The frontend never calls news APIs directly.
This endpoint acts as the secure backend proxy:
  1. Receives the search query from the frontend.
  2. Calls the evidence service which queries configured providers
     (NewsAPI, GNews, RSS feeds, Search API).
  3. Returns normalised, deduplicated, ranked results.

API keys stay server-side.  Provider errors are surfaced in the
`providers_failed` field — never exposed as raw exceptions.

Response shape
--------------
Identical to POST /evidence/search — uses the same EvidenceSearchResult
structure so the frontend has a single response type to handle.

Status values in response body
------------------------------
  "found"                — ≥1 result returned
  "INSUFFICIENT_EVIDENCE"— providers responded but found nothing
  "all_providers_failed" — every configured provider errored

The frontend must not interpret "INSUFFICIENT_EVIDENCE" as proof
that a claim is false.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, status

from app.services.evidence_service import search_evidence_fresh

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/news", tags=["Live News"])


@router.get(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Live news search",
    description=(
        "Search current news articles via configured backend providers "
        "(NewsAPI, GNews, RSS feeds, web search).\n\n"
        "**The frontend must not call news APIs directly.** "
        "All provider API keys are stored server-side.\n\n"
        "**Result status values:**\n"
        "- `found` — at least one article returned\n"
        "- `INSUFFICIENT_EVIDENCE` — providers responded but found nothing\n"
        "- `all_providers_failed` — every configured provider errored\n\n"
        "`INSUFFICIENT_EVIDENCE` does **not** mean the query topic is false "
        "— absence of results is not evidence of falsehood.\n\n"
        "Results are deduplicated by URL and ranked by "
        "keyword overlap + recency."
    ),
)
async def search_news(
    q: str = Query(
        ...,
        min_length=3,
        max_length=500,
        description="Search query or claim text.",
        examples=["ECB raised interest rates 2024"],
    ),
    from_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Restrict to articles published within this many days.",
    ),
    max_results: int = Query(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of articles to return.",
    ),
) -> dict:
    """
    Live news search — proxies to backend evidence providers.

    Returns normalised, deduplicated results.
    API keys are never exposed to the frontend.
    """
    logger.info(
        "Live news search: q=%r  from_days=%d  max_results=%d",
        q[:80], from_days, max_results,
    )
    return await search_evidence_fresh(
        claim=q,
        max_results=max_results,
        from_days=from_days,
    )
