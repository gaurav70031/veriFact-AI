"""
Request and response schemas for the /api/analyze endpoints.

Response design
---------------
The response deliberately separates three independent signals:

  ml_verdict        — What the ML models predict (FAKE/REAL/UNVERIFIED/MIXED).
                      Derived from model probability scores only.
                      High confidence does NOT mean factual correctness.

  evidence_verdict  — What the evidence search found.
                      INSUFFICIENT_EVIDENCE ≠ false.

  explanation       — Which words the model focused on.
                      High token weight ≠ factually incorrect word.
                      This is a MODEL SIGNAL, not factual evidence.

All three are always returned so clients can display and explain each
independently to users.
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
        description="Optional headline — prepended to text before inference.",
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
# Explainability sub-schemas
# =============================================================================

class TokenWeight(BaseModel):
    """
    Attribution weight for a single token.

    weight > 0  — model associates this token with FAKE patterns
    weight < 0  — model associates this token with REAL patterns

    IMPORTANT: weight != factual incorrectness.
    This reflects statistical patterns, not fact-checking.
    """
    token:    str
    weight:   float   # normalised to [-1, +1]
    position: int     # 0-based index in the token list


class ModelExplanation(BaseModel):
    """
    Explainability output for a single model's prediction.

    method values
    -------------
    lime          — LIME perturbation-based feature importance (baseline models)
    attention     — Attention-based attribution (DistilBERT)
    tfidf_weights — TF-IDF raw feature weights (fallback when LIME unavailable)
    unavailable   — Explanation could not be computed

    Disclaimer
    ----------
    Always shown to users alongside the highlighted tokens.
    Never claims any word "proves" the article is fake.
    """
    model_id:    str
    model_name:  str
    method:      str        # "lime" | "attention" | "tfidf_weights" | "unavailable"
    label:       str        # predicted label this explanation is for
    top_tokens:  list[TokenWeight] = []
    plain_text:  str = ""   # human-readable, non-technical explanation
    disclaimer:  str = ""   # always shown — clarifies model signal vs fact
    error:       Optional[str] = None


# =============================================================================
# Response sub-schemas
# =============================================================================

class ModelPrediction(BaseModel):
    """
    Result from a single ML model.
    Probability-based prediction — NOT a factual determination.
    """
    model_id:          str
    model_name:        str
    label:             str    # "FAKE" | "REAL"
    is_fake:           bool
    confidence:        float
    fake_probability:  float
    real_probability:  float
    inference_time_ms: float
    # Inline explanation (compact; full detail via GET /explanation/{id})
    explanation:       Optional[ModelExplanation] = None


class EvidenceSourceResult(BaseModel):
    source_name:           str
    title:                 str
    url:                   str
    snippet:               Optional[str]
    source_type:           str
    published_at:          Optional[datetime]
    retrieved_at:          datetime
    relevance_score:       float
    comparison_score:      float
    rank:                  int
    relationship_to_claim: str

    model_config = ConfigDict(from_attributes=True)


class ClaimResult(BaseModel):
    position:    int
    claim_text:  str

    ml_verdict:    Optional[str]
    ml_confidence: Optional[float]

    evidence_verdict:     Optional[str]
    evidence_explanation: Optional[str]

    # Top tokens across all models for this claim (aggregated)
    top_tokens: list[TokenWeight] = []

    evidence_sources:    list[EvidenceSourceResult] = []
    supporting_count:    int = 0
    contradicting_count: int = 0
    inconclusive_count:  int = 0
    not_relevant_count:  int = 0
    total_evidence:      int = 0
    assessment_conflict: bool = False
    evidence_limitations: list[str] = []


class EvidenceSummary(BaseModel):
    total_evidence:      int
    supporting_count:    int
    contradicting_count: int
    inconclusive_count:  int
    not_relevant_count:  int
    providers_used:      list[str]
    providers_failed:    list[str]
    evidence_limitations: list[str]
    all_providers_failed: bool = False


# =============================================================================
# Full analysis response
# =============================================================================

class AnalysisResponse(BaseModel):
    """
    Complete analysis result.

    Three independent signals
    -------------------------
    1. ml_verdict        — statistical model prediction
    2. evidence_verdict  — evidence-based assessment
    3. explanations      — which words drove the model (model signal, not fact)

    All three are independent; do not conflate them.
    """

    id:             int
    input_type:     str
    original_input: str
    source_url:     Optional[str]
    article_title:  Optional[str]

    # ── ML assessment ─────────────────────────────────────────────────────────
    ml_verdict:    str
    ml_confidence: Optional[float]

    # ── Evidence-based assessment ─────────────────────────────────────────────
    evidence_verdict:     Optional[str]
    evidence_explanation: Optional[str]

    # ── Per-model predictions (with inline explanations) ─────────────────────
    model_predictions: list[ModelPrediction]

    # ── Per-claim breakdown ───────────────────────────────────────────────────
    claims: list[ClaimResult] = []

    # ── Aggregate evidence summary ────────────────────────────────────────────
    evidence_summary: Optional[EvidenceSummary] = None

    # ── Legacy plain-text summary ─────────────────────────────────────────────
    summary: Optional[str] = None

    # ── Metadata ──────────────────────────────────────────────────────────────
    status:             str
    processing_time_ms: Optional[int]
    created_at:         datetime

    model_config = ConfigDict(from_attributes=True)


class AnalysisErrorResponse(BaseModel):
    id:            int
    status:        str
    error_message: str
    created_at:    datetime

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Standalone explanation response (GET /api/v1/explanation/{analysis_id})
# =============================================================================

class ExplanationTokenWeight(BaseModel):
    """Token weight as returned by the explanation endpoint."""
    token:    str
    weight:   float
    position: int


class PerModelExplanation(BaseModel):
    """
    Full explanation for one model's prediction on one analysis.

    Caveats displayed to users
    --------------------------
    * Token weights reflect statistical training patterns.
    * They are NOT evidence that any word is factually wrong.
    * Absence of a word from the list does not mean it is neutral.
    * Two articles can have identical model signals with opposite facts.
    """
    model_id:     str
    model_name:   str
    method:       str
    label:        str
    confidence:   float
    tokens:       list[ExplanationTokenWeight] = []
    top_tokens:   list[ExplanationTokenWeight] = []
    plain_text:   str = ""
    disclaimer:   str = ""
    error:        Optional[str] = None


class AggregateTokenWeight(BaseModel):
    """Token weight averaged across all available models for one claim."""
    token:  str
    weight: float


class ClaimExplanation(BaseModel):
    """Explanation breakdown for one extracted claim."""
    position:          int
    claim_text:        str
    aggregate_tokens:  list[AggregateTokenWeight] = []


class ExplanationResponse(BaseModel):
    """
    Full explanation response for GET /api/v1/explanation/{analysis_id}.

    Contains per-model explanations, per-claim aggregate tokens,
    and a top-level plain-language description of what the model found.

    Signal vs evidence warning
    --------------------------
    This explanation shows which words the MODEL focused on.
    It does NOT show which words are factually incorrect.
    Model signals and factual evidence are separate — always display both.
    """
    analysis_id:           int
    ml_verdict:            str
    ml_confidence:         Optional[float]
    evidence_verdict:      Optional[str] = None

    # Per-model LIME/attention explanations
    model_explanations:    list[PerModelExplanation] = []

    # Per-claim aggregate token weights
    claim_explanations:    list[ClaimExplanation]    = []

    # Signal-vs-evidence warning (always display)
    signal_vs_evidence_warning: str = (
        "The highlighted words show which parts of the text the ML models "
        "associated with known fake or real content patterns during training. "
        "This is a MODEL SIGNAL — it does not constitute factual evidence "
        "that any word or phrase is incorrect. Always check the evidence "
        "section for actual corroborating or contradicting sources."
    )
