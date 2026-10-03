"""
Claim model.

A single analysis can contain one or more discrete claims extracted from
the submitted text.  Each claim is assessed independently with:

  verdict          — ML-model-derived classification (FAKE/REAL/UNVERIFIED/MIXED)
  evidence_verdict — evidence-based assessment (separate from model predictions)

Separating these two assessments is intentional: the model may say FAKE but
the evidence search may return INSUFFICIENT_EVIDENCE, which are meaningfully
different outcomes that must not be collapsed.

EvidenceAssessment values
--------------------------
LIKELY_CREDIBLE        — multiple supporting sources, no contradictions
LIKELY_MISLEADING      — model says fake AND evidence leans contradicting
CONTRADICTED           — at least one credible source directly contradicts
UNVERIFIED             — evidence found but inconclusive (no clear direction)
INSUFFICIENT_EVIDENCE  — not enough evidence to make any assessment
"""

import enum
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint, Enum as PgEnum,
    Float, ForeignKey, Index, Integer, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.evidence_source import EvidenceSource


# ── ML-derived verdict ────────────────────────────────────────────────────────
class ClaimVerdict(str, enum.Enum):
    FAKE       = "FAKE"
    REAL       = "REAL"
    UNVERIFIED = "UNVERIFIED"
    MIXED      = "MIXED"


# ── Evidence-based assessment (new — distinct from ML verdict) ────────────────
class EvidenceAssessment(str, enum.Enum):
    LIKELY_CREDIBLE       = "LIKELY_CREDIBLE"
    LIKELY_MISLEADING     = "LIKELY_MISLEADING"
    CONTRADICTED          = "CONTRADICTED"
    UNVERIFIED            = "UNVERIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Claim(TimestampMixin, Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Parent ────────────────────────────────────────────────────────────────
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # ── Content ───────────────────────────────────────────────────────────────
    claim_text: Mapped[str]  = mapped_column(Text, nullable=False)
    position:   Mapped[int]  = mapped_column(Integer, default=1, nullable=False)

    # ── ML-derived verdict ────────────────────────────────────────────────────
    verdict:    Mapped[ClaimVerdict | None] = mapped_column(
        PgEnum(ClaimVerdict, name="claim_verdict"), nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Evidence-based assessment (separate from ML) ──────────────────────────
    evidence_verdict: Mapped[EvidenceAssessment | None] = mapped_column(
        PgEnum(EvidenceAssessment, name="evidence_assessment"),
        nullable=True,
    )

    # ── Explainability ────────────────────────────────────────────────────────
    explanation:    Mapped[str | None] = mapped_column(Text, nullable=True)
    # Evidence-based explanation (separate from ML explanation)
    evidence_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Top contributing tokens: [{"token": "...", "weight": 0.43}, ...]
    top_tokens_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="claims")
    evidence_sources: Mapped[list["EvidenceSource"]] = relationship(
        "EvidenceSource", back_populates="claim", cascade="all, delete-orphan",
    )

    # ── Constraints & Indexes ─────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_claim_confidence_range",
        ),
        CheckConstraint("position >= 1", name="ck_claim_position_positive"),
        Index("ix_claims_analysis_position", "analysis_id", "position"),
        Index("ix_claims_evidence_verdict",  "evidence_verdict"),
    )

    def __repr__(self) -> str:
        return (
            f"<Claim id={self.id} analysis_id={self.analysis_id} "
            f"verdict={self.verdict} evidence={self.evidence_verdict} "
            f"pos={self.position}>"
        )
