"""
EvidenceSource model.

Stores metadata about an external article/document retrieved as evidence
for a specific claim.  Full article bodies are NOT stored — only title,
snippet (≤ 500 chars), URL, and scores.

EvidenceRelationship values
---------------------------
SUPPORTING    — evidence content aligns with / supports the claim
CONTRADICTING — evidence content contradicts the claim
INCONCLUSIVE  — relevant to the topic but neither supports nor contradicts
NOT_RELEVANT  — low semantic overlap; fetched but not meaningful for this claim
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint, DateTime, Enum as PgEnum,
    Float, ForeignKey, Index, Integer, String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.claim import Claim


class SourceType(str, enum.Enum):
    NEWS_API   = "news_api"
    RSS_FEED   = "rss_feed"
    WEB_SEARCH = "web_search"
    OFFICIAL   = "official"
    FACT_CHECK = "fact_check"
    ACADEMIC   = "academic"
    USER_URL   = "user_url"


class EvidenceRelationship(str, enum.Enum):
    SUPPORTING    = "supporting"     # evidence supports the claim
    CONTRADICTING = "contradicting"  # evidence contradicts the claim
    INCONCLUSIVE  = "inconclusive"   # relevant but neither supports nor contradicts
    NOT_RELEVANT  = "not_relevant"   # too low similarity to be meaningful


class EvidenceSource(TimestampMixin, Base):
    __tablename__ = "evidence_sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Parent ────────────────────────────────────────────────────────────────
    claim_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # ── Source metadata ───────────────────────────────────────────────────────
    source_name:  Mapped[str]        = mapped_column(String(200), nullable=False)
    title:        Mapped[str]        = mapped_column(String(500), nullable=False)
    url:          Mapped[str]        = mapped_column(String(2000), nullable=False)
    snippet:      Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_type:  Mapped[SourceType] = mapped_column(
        PgEnum(SourceType, name="source_type"), nullable=False,
    )

    # ── Dates ─────────────────────────────────────────────────────────────────
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Keyword-overlap + recency relevance score from the evidence ranker
    relevance_score:  Mapped[float]      = mapped_column(Float, nullable=False)
    # Semantic similarity between claim text and this snippet (0–1).
    # Set by the evidence comparator; 0.0 when comparator not run.
    comparison_score: Mapped[float]      = mapped_column(Float, nullable=False, default=0.0)
    # Rank within the result set for this claim (1 = most relevant)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # ── Relationship to claim ─────────────────────────────────────────────────
    relationship_to_claim: Mapped[EvidenceRelationship] = mapped_column(
        PgEnum(EvidenceRelationship, name="evidence_relationship"),
        default=EvidenceRelationship.INCONCLUSIVE,
        nullable=False,
    )

    # ── ORM ───────────────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="evidence_sources")

    # ── Constraints & Indexes ─────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint(
            "relevance_score  >= 0.0 AND relevance_score  <= 1.0",
            name="ck_evidence_relevance_range",
        ),
        CheckConstraint(
            "comparison_score >= 0.0 AND comparison_score <= 1.0",
            name="ck_evidence_comparison_range",
        ),
        CheckConstraint("rank >= 1", name="ck_evidence_rank_positive"),
        Index("ix_evidence_claim_rank",    "claim_id", "rank"),
        Index("ix_evidence_source_type",   "source_type"),
        Index("ix_evidence_relationship",  "relationship_to_claim"),
    )

    def __repr__(self) -> str:
        return (
            f"<EvidenceSource id={self.id} claim_id={self.claim_id} "
            f"source={self.source_name!r} rel={self.relationship_to_claim}>"
        )
