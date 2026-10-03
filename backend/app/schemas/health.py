"""
Schema for GET /api/health.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ComponentStatus(BaseModel):
    status:  str             # "ok" | "degraded" | "unavailable"
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status:     str          # "ok" | "degraded" | "unavailable"
    version:    str
    environment: str
    components: dict[str, ComponentStatus]
