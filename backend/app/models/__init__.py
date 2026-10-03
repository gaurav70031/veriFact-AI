"""
Import every ORM model here so that:
  1. SQLAlchemy mapper registry is fully populated before any query runs.
  2. Alembic env.py only needs to import this module to detect all tables.
  3. app/db/session.py's create_all_tables() picks up every table.
"""

from app.models.user           import User, UserRole
from app.models.analysis       import Analysis, InputType, AnalysisStatus, FinalVerdict
from app.models.claim          import Claim, ClaimVerdict
from app.models.model_version  import ModelVersion, ModelType, ModelAlgorithm
from app.models.prediction     import Prediction, PredictionLabel
from app.models.evidence_source import EvidenceSource, SourceType, EvidenceRelationship
from app.models.model_run      import ModelRun, RunStatus

__all__ = [
    # Models
    "User",
    "Analysis",
    "Claim",
    "ModelVersion",
    "Prediction",
    "EvidenceSource",
    "ModelRun",
    # Enums
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
