"""
Prediction model.

One row per model per analysis.  If three models run on a single analysis,
three Prediction rows are created — one for LR, one for SVM, one for
DistilBERT.  A fourth row with model_type=ENSEMBLE stores the final verdict.

This normalised design allows per-model accuracy analytics over time.
"""

import enum
from sqlalchemy import (
    String, Float, Integer, Text, Enum as PgEnum,
    ForeignKey, Index, CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.model_version import ModelVersion


class PredictionLabel(str, enum.Enum):
    FAKE = "FAKE"
    REAL = "REAL"


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Foreign keys ─────────────────────────────────────────────────────────
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False
    )

    # ── Result ───────────────────────────────────────────────────────────────
    label: Mapped[PredictionLabel] = mapped_column(
        PgEnum(PredictionLabel, name="prediction_label"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    real_probability: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Explainability ───────────────────────────────────────────────────────
    # Serialised token/feature weights JSON produced by LIME or attention
    explanation_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ────────────────────────────────────────────────────────
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion", back_populates="predictions"
    )

    # ── Constraints & Indexes ────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0",
                        name="ck_prediction_confidence_range"),
        CheckConstraint("fake_probability >= 0.0 AND fake_probability <= 1.0",
                        name="ck_prediction_fake_prob_range"),
        CheckConstraint("real_probability >= 0.0 AND real_probability <= 1.0",
                        name="ck_prediction_real_prob_range"),
        Index("ix_predictions_analysis", "analysis_id"),
        Index("ix_predictions_model_version", "model_version_id"),
        Index("ix_predictions_label", "label"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} analysis_id={self.analysis_id} "
            f"label={self.label} confidence={self.confidence:.3f}>"
        )
