"""
Claim model.

A single analysis can contain one or more discrete claims extracted from the
submitted text (e.g. "Claim 1: vaccines cause autism").
Each claim is assessed independently and can have its own evidence sources.
"""

import enum
from sqlalchemy import (
    String, Text, Float, Integer, Enum as PgEnum,
    ForeignKey, Index, CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.evidence_source import EvidenceSource


class ClaimVerdict(str, enum.Enum):
    FAKE = "FAKE"
    REAL = "REAL"
    UNVERIFIED = "UNVERIFIED"
    MIXED = "MIXED"


class Claim(TimestampMixin, Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Parent ───────────────────────────────────────────────────────────────
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # ── Content ──────────────────────────────────────────────────────────────
    # The extracted or isolated claim text
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Ordinal position within the analysis (1-based)
    position: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # ── Assessment ───────────────────────────────────────────────────────────
    verdict: Mapped[ClaimVerdict | None] = mapped_column(
        PgEnum(ClaimVerdict, name="claim_verdict"), nullable=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Explainability ───────────────────────────────────────────────────────
    # Top contributing tokens as JSON: [{"token": "vaccine", "weight": 0.43}, ...]
    # Stored as TEXT to avoid requiring pgJSON extension — parsed by app layer
    top_tokens_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ────────────────────────────────────────────────────────
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="claims")
    evidence_sources: Mapped[list["EvidenceSource"]] = relationship(
        "EvidenceSource", back_populates="claim", cascade="all, delete-orphan"
    )

    # ── Constraints & Indexes ────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0",
                        name="ck_claim_confidence_range"),
        CheckConstraint("position >= 1", name="ck_claim_position_positive"),
        Index("ix_claims_analysis_position", "analysis_id", "position"),
    )

    def __repr__(self) -> str:
        return (
            f"<Claim id={self.id} analysis_id={self.analysis_id} "
            f"verdict={self.verdict} pos={self.position}>"
        )
