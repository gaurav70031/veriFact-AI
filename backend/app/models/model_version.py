"""
ModelVersion model.

Registry of every ML model artifact that has been trained and deployed.
Predictions and model runs reference a row here so results are always
traceable to the exact artifact that produced them.
"""

import enum
from sqlalchemy import String, Text, Float, Boolean, Enum as PgEnum, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.prediction import Prediction
    from app.models.model_run import ModelRun


class ModelType(str, enum.Enum):
    TRADITIONAL = "traditional"   # TF-IDF + classical classifier
    TRANSFORMER = "transformer"   # DistilBERT / BERT variant
    ENSEMBLE = "ensemble"         # Combined output of multiple models


class ModelAlgorithm(str, enum.Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVM = "linear_svm"
    NAIVE_BAYES = "naive_bayes"
    DISTILBERT = "distilbert"
    ENSEMBLE = "ensemble"


class ModelVersion(TimestampMixin, Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ── Identity ─────────────────────────────────────────────────────────────
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)   # e.g. "1.0.0"
    model_type: Mapped[ModelType] = mapped_column(
        PgEnum(ModelType, name="model_type"), nullable=False
    )
    algorithm: Mapped[ModelAlgorithm] = mapped_column(
        PgEnum(ModelAlgorithm, name="model_algorithm"), nullable=False
    )

    # ── Artifact location ────────────────────────────────────────────────────
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    vectorizer_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # ── Training metadata ────────────────────────────────────────────────────
    dataset_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    training_samples: Mapped[int | None] = mapped_column(nullable=True)
    test_samples: Mapped[int | None] = mapped_column(nullable=True)

    # ── Evaluation metrics (recorded once at training time) ──────────────────
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Status ───────────────────────────────────────────────────────────────
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ────────────────────────────────────────────────────────
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="model_version"
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        "ModelRun", back_populates="model_version"
    )

    # ── Constraints & Indexes ────────────────────────────────────────────────
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_name_version"),
        Index("ix_model_versions_active", "is_active"),
        Index("ix_model_versions_algorithm", "algorithm"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelVersion id={self.id} name={self.model_name!r} "
            f"version={self.version!r} active={self.is_active}>"
        )
