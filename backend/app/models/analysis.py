"""
Analysis model.

One analysis = one user submission (text / URL / search query).
It is the top-level entity that owns claims, predictions, and model runs.
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
    from app.models.user import User
    from app.models.claim import Claim
    from app.models.prediction import Prediction
    from app.models.model_run import ModelRun


class InputType(str, enum.Enum):
    TEXT = "text"       # raw text / claim pasted by user
    URL = "url"         # public article URL
    QUERY = "query"     # news search query string


class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FinalVerdict(str, enum.Enum):
    FAKE = "FAKE"
    REAL = "REAL"
    UNVERIFIED = "UNVERIFIED"   # insufficient evidence
    MIXED = "MIXED"             # conflicting signals


class Analysis(TimestampMixin, Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Ownership ────────────────────────────────────────────────────────────
    # NULL allowed: support anonymous / unauthenticated usage
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # ── Input ────────────────────────────────────────────────────────────────
    input_type: Mapped[InputType] = mapped_column(
        PgEnum(InputType, name="input_type"), nullable=False
    )
    original_input: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    article_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    article_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Output ───────────────────────────────────────────────────────────────
    status: Mapped[AnalysisStatus] = mapped_column(
        PgEnum(AnalysisStatus, name="analysis_status"),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
    )
    final_verdict: Mapped[FinalVerdict | None] = mapped_column(
        PgEnum(FinalVerdict, name="final_verdict"), nullable=True
    )
    final_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Performance ──────────────────────────────────────────────────────────
    # Total wall-clock time in milliseconds from request receipt to response
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Relationships ────────────────────────────────────────────────────────
    user: Mapped["User | None"] = relationship("User", back_populates="analyses")
    claims: Mapped[list["Claim"]] = relationship(
        "Claim", back_populates="analysis", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="analysis", cascade="all, delete-orphan"
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        "ModelRun", back_populates="analysis", cascade="all, delete-orphan"
    )

    # ── Constraints & Indexes ────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint("final_confidence >= 0.0 AND final_confidence <= 1.0",
                        name="ck_analysis_confidence_range"),
        Index("ix_analyses_user_created", "user_id", "created_at"),
        Index("ix_analyses_status_created", "status", "created_at"),
        Index("ix_analyses_verdict", "final_verdict"),
    )

    def __repr__(self) -> str:
        return (
            f"<Analysis id={self.id} type={self.input_type} "
            f"verdict={self.final_verdict} status={self.status}>"
        )
