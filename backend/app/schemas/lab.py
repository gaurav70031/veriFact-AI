"""
Schemas for the Model Laboratory API.

Endpoints:
  GET  /api/v1/lab/models                  — all registered models with full details
  GET  /api/v1/lab/models/{id}             — single model full detail
  GET  /api/v1/lab/models/{id}/curves      — training/validation loss curves
  GET  /api/v1/lab/models/{id}/confusion   — confusion matrix data
  GET  /api/v1/lab/models/{id}/roc         — ROC curve data points
  GET  /api/v1/lab/comparison              — side-by-side metrics for all models
  GET  /api/v1/lab/report                  — automated research comparison report
  POST /api/v1/lab/train                   — trigger training job (baseline)
  GET  /api/v1/lab/train/{job_id}/status   — training job status
  GET  /api/v1/lab/train/{job_id}/logs     — training log lines
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Full model detail ─────────────────────────────────────────────────────────

class PerClassDetail(BaseModel):
    precision: Optional[float]
    recall:    Optional[float]
    f1:        Optional[float]
    support:   Optional[int]


class LossCurvePoint(BaseModel):
    epoch:      int
    train_loss: float
    val_loss:   Optional[float]
    val_acc:    Optional[float]


class ModelDetailResponse(BaseModel):
    id:               int
    model_name:       str
    version:          str
    model_type:       str
    algorithm:        str
    is_active:        bool
    description:      Optional[str]
    dataset_name:     Optional[str]
    training_samples: Optional[int]
    test_samples:     Optional[int]
    random_seed:      Optional[int]

    # Timing
    train_time_s:      Optional[float]
    inference_time_ms: Optional[float]

    # Size
    model_size_bytes: Optional[int]

    # Core metrics
    accuracy:  Optional[float]
    precision: Optional[float]
    recall:    Optional[float]
    f1_score:  Optional[float]
    roc_auc:   Optional[float]

    # Extended
    confusion_matrix:  Optional[list[list[int]]]      # [[TN,FP],[FN,TP]]
    training_history:  Optional[list[LossCurvePoint]]
    per_class_metrics: Optional[dict[str, PerClassDetail]]
    hyperparameters:   Optional[dict]

    created_at: datetime


# ── Comparison ────────────────────────────────────────────────────────────────

class ModelComparisonRow(BaseModel):
    model_name:        str
    model_type:        str
    version:           str
    accuracy:          Optional[float]
    precision:         Optional[float]
    recall:            Optional[float]
    f1_score:          Optional[float]
    roc_auc:           Optional[float]
    train_time_s:      Optional[float]
    inference_time_ms: Optional[float]
    model_size_bytes:  Optional[int]
    test_samples:      Optional[int]


class ComparisonResponse(BaseModel):
    models:          list[ModelComparisonRow]
    dataset_name:    Optional[str]
    test_split_pct:  float = 10.0
    evaluation_note: str = (
        "All metrics computed on the same held-out test set (10% stratified split). "
        "No test data was used during training or hyperparameter selection."
    )


# ── ROC curve ─────────────────────────────────────────────────────────────────

class RocPoint(BaseModel):
    fpr: float   # false positive rate
    tpr: float   # true positive rate


class RocResponse(BaseModel):
    model_id:  int
    model_name: str
    roc_auc:   Optional[float]
    points:    list[RocPoint]
    note:      str = (
        "ROC curve reconstructed from stored AUC. "
        "Re-run evaluate_baseline.py to get exact threshold-sweep data."
    )


# ── Confusion matrix ──────────────────────────────────────────────────────────

class ConfusionResponse(BaseModel):
    model_id:    int
    model_name:  str
    matrix:      list[list[int]]   # [[TN,FP],[FN,TP]]
    labels:      list[str] = ["FAKE", "REAL"]
    test_samples: Optional[int]


# ── Research report ───────────────────────────────────────────────────────────

class BestModelEntry(BaseModel):
    metric:      str
    model_name:  str
    value:       Optional[float]


class ModelGroupSummary(BaseModel):
    group:          str    # "Traditional NLP" | "Transformer"
    models:         list[str]
    avg_accuracy:   Optional[float]
    avg_f1:         Optional[float]
    avg_train_time: Optional[float]


class ResearchReport(BaseModel):
    generated_at:    str
    dataset:         str
    test_split:      str
    total_models:    int

    best_by_accuracy:   BestModelEntry
    best_by_precision:  BestModelEntry
    best_by_recall:     BestModelEntry
    best_by_f1:         BestModelEntry
    best_by_roc_auc:    BestModelEntry
    fastest_training:   BestModelEntry
    fastest_inference:  BestModelEntry

    traditional_summary:  ModelGroupSummary
    transformer_summary:  ModelGroupSummary

    narrative:   list[str]   # paragraph strings — derived, not generated
    limitations: list[str]

    evaluation_conditions: dict


# ── Training job ──────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="logistic_regression | linear_svm | naive_bayes",
        pattern="^(logistic_regression|linear_svm|naive_bayes)$",
    )
    data_dir: str = Field(
        default="ml/datasets/processed",
        description="Path to directory containing train.csv and val.csv",
    )


class TrainJobStatus(BaseModel):
    job_id:     str
    model_id:   str
    status:     str   # "queued" | "running" | "completed" | "failed"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error:      Optional[str] = None
    result_model_version_id: Optional[int] = None
    log_lines:  list[str] = []
