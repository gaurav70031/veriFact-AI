"""
Import all ORM models here so that:
 1. SQLAlchemy's mapper registry is fully populated before any query runs.
 2. Alembic's env.py only needs to import this module to detect all tables.
"""

from app.models.user import User, UserRole
from app.models.analysis import Analysis, InputType, AnalysisStatus, FinalVerdict
from app.models.claim import Claim, ClaimVerdict
from app.models.model_version import ModelVersion, ModelType, ModelAlgorithm
from app.models.prediction import Prediction, PredictionLabel
from app.models.evidence_source import EvidenceSource, SourceType, EvidenceRelationship
from app.models.model_run import ModelRun, RunStatus

__all__ = [
    # Models
    "User",
    "Analysis",
    "Claim",
    "ModelVersion",
    "Prediction",
    "EvidenceSource",
    "ModelRun",
    # Enums (exported so schemas can import from one place)
    "UserRole",
    "InputType",
    "AnalysisStatus",
    "FinalVerdict",
    "ClaimVerdict",
    "ModelType",
    "ModelAlgorithm",
    "PredictionLabel",
    "SourceType",
    "EvidenceRelationship",
    "RunStatus",
]
