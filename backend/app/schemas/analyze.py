"""
Request and response schemas for the /api/analyze endpoints.

Response design
---------------
The response deliberately separates two independent assessments:

  ml_verdict        — What the ML models predict (FAKE/REAL/UNVERIFIED/MIXED).
                      Derived from model probability scores only.
                      High confidence does NOT mean factual correctness.

  evidence_verdict  — What the evidence search found
                      (LIKELY_CREDIBLE / LIKELY_MISLEADING / CONTRADICTED /
                       UNVERIFIED / INSUFFICIENT_EVIDENCE).
                      Based on actual retrieved sources.
                      INSUFFICIENT_EVIDENCE ≠ false.

Both verdicts are always present in the response so clients can display
them together and explain the distinction to users.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# Request schemas
# =============================================================================

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


# =============================================================================
# Response sub-schemas
# =============================================================================

class ModelPrediction(BaseModel):
    """
    Result from a single ML model.
    This is a probability-based prediction, NOT a factual determination.
    """
    model_id:          str
    model_name:        str
    label:             str    # "FAKE" | "REAL"
    is_fake:           bool
    confidence:        float  # max(fake_prob, real_prob)
    fake_probability:  float
    real_probability:  float
    inference_time_ms: float


class EvidenceSourceResult(BaseModel):
    """
    One evidence source as returned in the API response.
    Full article text is never included — only metadata and a short snippet.
    """
    source_name:          str
    title:                str
    url:                  str
    snippet:              Optional[str]         # ≤ 500 chars
    source_type:          str                   # "news_api" | "rss_feed" | etc.
    published_at:         Optional[datetime]
    retrieved_at:         datetime
    relevance_score:      float                 # keyword+recency score from ranker
    comparison_score:     float                 # semantic/lexical similarity to claim
    rank:                 int
    relationship_to_claim: str                  # "supporting" | "contradicting" | ...

    model_config = ConfigDict(from_attributes=True)


class ClaimResult(BaseModel):
    """
    Full assessment for one extracted claim.

    Contains BOTH the ML-derived verdict AND the evidence-based assessment
    so clients can display and explain them independently.
    """
    position:    int
    claim_text:  str

    # ── ML-derived ────────────────────────────────────────────────────────────
    ml_verdict:   Optional[str]   # "FAKE" | "REAL" | "UNVERIFIED" | "MIXED"
    ml_confidence: Optional[float]

    # ── Evidence-based (separate from ML) ────────────────────────────────────
    evidence_verdict: Optional[str]   # "LIKELY_CREDIBLE" | "CONTRADICTED" | etc.
    evidence_explanation: Optional[str]

    # ── Evidence items ────────────────────────────────────────────────────────
    evidence_sources:    list[EvidenceSourceResult] = []

    # ── Evidence summary counts ───────────────────────────────────────────────
    supporting_count:    int = 0
    contradicting_count: int = 0
    inconclusive_count:  int = 0
    not_relevant_count:  int = 0
    total_evidence:      int = 0

    # ── Conflict flag ─────────────────────────────────────────────────────────
    # True when ML and evidence assessments disagree significantly
    assessment_conflict: bool = False

    # ── Limitations ───────────────────────────────────────────────────────────
    evidence_limitations: list[str] = []


class EvidenceSummary(BaseModel):
    """
    Aggregate evidence statistics across all claims in the analysis.
    Always present even when no evidence was found.
    """
    total_evidence:      int
    supporting_count:    int
    contradicting_count: int
    inconclusive_count:  int
    not_relevant_count:  int
    providers_used:      list[str]
    providers_failed:    list[str]
    evidence_limitations: list[str]

    # Whether absence of evidence was due to provider failure vs true no-results
    all_providers_failed: bool = False


# =============================================================================
# Full analysis response
# =============================================================================

class AnalysisResponse(BaseModel):
    """
    Complete analysis response.

    Two independent assessments are always returned:

    1. ml_verdict / ml_confidence
       — Ensemble output of TF-IDF models and/or DistilBERT.
       — High confidence means the text pattern matches known fake/real patterns.
       — Does NOT verify the factual content of any claim.

    2. evidence_verdict (per claim and overall)
       — Based on retrieved news/RSS/search evidence.
       — INSUFFICIENT_EVIDENCE means not enough was found, NOT that it's false.
       — CONTRADICTED means multiple independent sources dispute a claim.

    The claims list contains the full per-claim breakdown.
    """

    id:             int
    input_type:     str          # "text" | "url" | "claim"
    original_input: str
    source_url:     Optional[str]
    article_title:  Optional[str]

    # ── ML assessment ─────────────────────────────────────────────────────────
    ml_verdict:    str           # "FAKE" | "REAL" | "UNVERIFIED" | "MIXED"
    ml_confidence: Optional[float]

    # ── Evidence-based assessment ─────────────────────────────────────────────
    evidence_verdict: Optional[str]  # "LIKELY_CREDIBLE" | "CONTRADICTED" | etc.
    evidence_explanation: Optional[str]

    # ── Per-model ML breakdown ────────────────────────────────────────────────
    model_predictions: list[ModelPrediction]

    # ── Per-claim breakdown (includes evidence per claim) ─────────────────────
    claims: list[ClaimResult] = []

    # ── Aggregate evidence summary ────────────────────────────────────────────
    evidence_summary: Optional[EvidenceSummary] = None

    # ── Legacy field: plain-text summary (kept for backwards compat) ──────────
    summary: Optional[str] = None

    # ── Metadata ──────────────────────────────────────────────────────────────
    status:             str
    processing_time_ms: Optional[int]
    created_at:         datetime

    model_config = ConfigDict(from_attributes=True)


class AnalysisErrorResponse(BaseModel):
    """Returned when an analysis fails (e.g., URL extraction error)."""

    id:            int
    status:        str   # "failed"
    error_message: str
    created_at:    datetime

    model_config = ConfigDict(from_attributes=True)
