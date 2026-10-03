"""
Schemas for GET /api/history and GET /api/analysis/{id}.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class AnalysisListItem(BaseModel):
    """Compact row shown in the history list."""

    id:               int
    input_type:       str
    original_input:   str
    source_url:       Optional[str]
    article_title:    Optional[str]
    final_verdict:    Optional[str]
    final_confidence: Optional[float]
    status:           str
    processing_time_ms: Optional[int]
    created_at:       datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedHistory(BaseModel):
    """Paginated list of analysis records."""

    items:       list[AnalysisListItem]
    total:       int
    page:        int
    page_size:   int
    total_pages: int


class HistoryQueryParams(BaseModel):
    """Query parameters for GET /api/history."""

    page:      int = Field(default=1,  ge=1,  description="Page number (1-based).")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page.")
    verdict:   Optional[str] = Field(
        None,
        description="Filter by verdict: FAKE | REAL | UNVERIFIED | MIXED.",
        pattern="^(FAKE|REAL|UNVERIFIED|MIXED)$",
    )
    input_type: Optional[str] = Field(
        None,
        description="Filter by input type: text | url | claim.",
        pattern="^(text|url|claim)$",
    )
