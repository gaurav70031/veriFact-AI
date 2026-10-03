"""
EvidenceSource model.

Stores metadata about an external article/document retrieved as evidence
for a specific claim.  Full article bodies are NOT stored here to respect
copyright — only title, snippet (≤ 500 chars), URL, and scores.

Each evidence source belongs to exactly one Claim.
"""

import enum
from sqlalchemy import (
    String, Text, Float, Integer, DateTime, Enum as PgEnum,
    ForeignKey, Index, CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from typing import TYPE_CHECKING

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.claim import Claim


class SourceType(str, enum.Enum):
    NEWS_API = "news_api"         # Fetched via NewsAPI
    RSS_FEED = "rss_feed"         # Fetched from an RSS feed
    WEB_SEARCH = "web_search"     # Fetched via search provider (SerpAPI etc.)
    OFFICIAL = "official"         # Government / institutional source
    FACT_CHECK = "fact_check"     # Dedicated fact-check site
    ACADEMIC = "academic"         # Journal / preprint
    USER_URL = "user_url"         # Supplied directly by the user


class EvidenceRelationship(str, enum.Enum):
    SUPPORTING = "supporting"         # Evidence supports the claim being real
    CONTRADICTING = "contradicting"   # Evidence contradicts the claim
    INCONCLUSIVE = "inconclusive"     # Relevant but neutral


class EvidenceSource(TimestampMixin, Base):
    __tablename__ = "evidence_sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Parent ───────────────────────────────────────────────────────────────
    claim_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # ── Source metadata ──────────────────────────────────────────────────────
    source_name: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    # Snippet ≤ 500 chars — enough for display; respects copyright
    snippet: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_type: Mapped[SourceType] = mapped_column(
        PgEnum(SourceType, name="source_type"), nullable=False
    )

    # ── Dates ────────────────────────────────────────────────────────────────
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # ── Scoring ──────────────────────────────────────────────────────────────
    # Cosine similarity or BM25 score between claim text and this article
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    # Rank within the result set for this claim (1 = most relevant)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # ── Relationship to claim ─────────────────────────────────────────────────
    relationship_to_claim: Mapped[EvidenceRelationship] = mapped_column(
        PgEnum(EvidenceRelationship, name="evidence_relationship"),
        default=EvidenceRelationship.INCONCLUSIVE,
        nullable=False,
    )

    # ── ORM ──────────────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="evidence_sources")

    # ── Constraints & Indexes ────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint("relevance_score >= 0.0 AND relevance_score <= 1.0",
                        name="ck_evidence_relevance_range"),
        CheckConstraint("rank >= 1", name="ck_evidence_rank_positive"),
        Index("ix_evidence_claim_rank", "claim_id", "rank"),
        Index("ix_evidence_source_type", "source_type"),
        Index("ix_evidence_relationship", "relationship_to_claim"),
    )

    def __repr__(self) -> str:
        return (
            f"<EvidenceSource id={self.id} claim_id={self.claim_id} "
            f"source={self.source_name!r} rel={self.relationship_to_claim}>"
        )
