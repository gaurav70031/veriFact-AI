"""
Import all ORM models so SQLAlchemy's mapper registry is fully populated
before any query runs, and Alembic can detect all tables.
"""

from app.models.user           import User, UserRole
from app.models.analysis       import Analysis, InputType, AnalysisStatus, FinalVerdict
from app.models.claim          import Claim, ClaimVerdict, EvidenceAssessment
from app.models.model_version  import ModelVersion, ModelType, ModelAlgorithm
from app.models.prediction     import Prediction, PredictionLabel
from app.models.evidence_source import (
    EvidenceSource, SourceType, EvidenceRelationship,
)
from app.models.model_run      import ModelRun, RunStatus

__all__ = [
    "User",          "Analysis",      "Claim",          "ModelVersion",
    "Prediction",    "EvidenceSource", "ModelRun",
    "UserRole",      "InputType",     "AnalysisStatus",  "FinalVerdict",
    "ClaimVerdict",  "EvidenceAssessment",
    "ModelType",     "ModelAlgorithm",
    "PredictionLabel",
    "SourceType",    "EvidenceRelationship",
    "RunStatus",
]
