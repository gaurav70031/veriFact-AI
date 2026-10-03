"""
Request and response schemas for the /api/analyze endpoints.

Three input types are supported:
  POST /api/analyze/text  — raw text or claim
  POST /api/analyze/url   — public article URL
  POST /api/analyze/claim — short factual claim with optional context
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ── Request schemas ───────────────────────────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    """Submit raw article text or a multi-sentence claim for analysis."""

    text: str = Field(
        ...,
        min_length=20,
        max_length=50_000,
        description="Article body or multi-sentence text to analyse.",
        examples=["Scientists claim a new study proves that vaccines cause autism..."],
    )
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional article headline — prepended to text before inference.",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v.strip()

    model_config = ConfigDict(str_strip_whitespace=True)


class AnalyzeUrlRequest(BaseModel):
    """Submit a public article URL for extraction and analysis."""

    url: str = Field(
        ...,
        max_length=2_000,
        description="Public HTTPS URL of a news article.",
        examples=["https://www.reuters.com/world/example-article-2024"],
    )

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    model_config = ConfigDict(str_strip_whitespace=True)


class AnalyzeClaimRequest(BaseModel):
    """Submit a short factual claim (≤ 500 chars) for targeted fact-checking."""

    claim: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="A concise, checkable factual claim.",
        examples=["The Eiffel Tower was built in 1889."],
    )
    context: Optional[str] = Field(
        None,
        max_length=5_000,
        description="Optional surrounding context (article excerpt, paragraph).",
    )

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim must not be blank.")
        return v.strip()

    model_config = ConfigDict(str_strip_whitespace=True)


# ── Per-model prediction sub-schema ──────────────────────────────────────────

class ModelPrediction(BaseModel):
    """Result from a single ML model."""

    model_id:         str
    model_name:       str
    label:            str            # "FAKE" | "REAL"
    is_fake:          bool
    confidence:       float          # 0.0–1.0
    fake_probability: float
    real_probability: float
    inference_time_ms: float


# ── Analysis response schemas ─────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """
    Full analysis result returned by all three /api/analyze/* endpoints.
    """

    id:               int
    input_type:       str            # "text" | "url" | "claim"
    original_input:   str
    source_url:       Optional[str]
    article_title:    Optional[str]

    # Final verdict (ensemble)
    final_verdict:    str            # "FAKE" | "REAL" | "UNVERIFIED" | "MIXED"
    final_confidence: Optional[float]
    summary:          Optional[str]

    # Per-model breakdown
    model_predictions: list[ModelPrediction]

    # Metadata
    status:           str
    processing_time_ms: Optional[int]
    created_at:       datetime

    model_config = ConfigDict(from_attributes=True)


class AnalysisErrorResponse(BaseModel):
    """Returned when an analysis fails (e.g., URL extraction error)."""

    id:            int
    status:        str   # "failed"
    error_message: str
    created_at:    datetime

    model_config = ConfigDict(from_attributes=True)
