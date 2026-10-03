"""
ModelRun model.

Execution log for each time a model was invoked.
Records latency, whether it succeeded, and the model version used.
Useful for monitoring model performance in production over time.

One analysis → many model runs (one per model invoked).
"""

import enum
from sqlalchemy import (
    String, Integer, Boolean, Text, Enum as PgEnum,
    ForeignKey, Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.model_version import ModelVersion


class RunStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ModelRun(TimestampMixin, Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Foreign keys ─────────────────────────────────────────────────────────
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False
    )

    # ── Execution details ─────────────────────────────────────────────────────
    status: Mapped[RunStatus] = mapped_column(
        PgEnum(RunStatus, name="run_status"),
        default=RunStatus.SUCCESS,
        nullable=False,
    )
    # Wall-clock inference time in milliseconds
    inference_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Input token/character count fed to this model
    input_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ────────────────────────────────────────────────────────
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="model_runs")
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion", back_populates="model_runs"
    )

    # ── Indexes ──────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_model_runs_analysis", "analysis_id"),
        Index("ix_model_runs_status", "status"),
        Index("ix_model_runs_model_version", "model_version_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRun id={self.id} analysis_id={self.analysis_id} "
            f"status={self.status} time_ms={self.inference_time_ms}>"
        )
