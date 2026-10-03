"""
Schema for analytics / statistics endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel


class VerdictCounts(BaseModel):
    FAKE:       int = 0
    REAL:       int = 0
    UNVERIFIED: int = 0
    MIXED:      int = 0


class StatsResponse(BaseModel):
    total_analyses:     int
    completed:          int
    failed:             int
    pending:            int
    verdict_counts:     VerdictCounts
    fake_percentage:    float
    real_percentage:    float
    avg_confidence:     float
    avg_processing_ms:  float
