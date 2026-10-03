"""
Schemas for GET /api/models and GET /api/model-performance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    """Metadata about a registered ML model."""

    id:              int
    model_name:      str
    version:         str
    model_type:      str   # "traditional" | "transformer" | "ensemble"
    algorithm:       str
    is_active:       bool
    description:     Optional[str]
    dataset_name:    Optional[str]
    training_samples: Optional[int]
    created_at:      datetime

    model_config = ConfigDict(from_attributes=True)


class PerClassMetrics(BaseModel):
    precision: float
    recall:    float
    f1:        float
    support:   int


class ModelPerformance(BaseModel):
    """Evaluation metrics for a single model (from training-time evaluation)."""

    model_name:      str
    version:         str
    algorithm:       str
    accuracy:        Optional[float]
    precision:       Optional[float]
    recall:          Optional[float]
    f1_score:        Optional[float]
    roc_auc:         Optional[float]
    test_samples:    Optional[int]

    model_config = ConfigDict(from_attributes=True)


class ModelPerformanceList(BaseModel):
    models: list[ModelPerformance]
