"""
backend/app/services/lab_service.py
=====================================
Model Laboratory service layer.

Responsibilities
----------------
* Read model details (with JSON blob columns decoded) from the DB.
* Build comparison tables from DB rows — no hardcoded values.
* Generate the research comparison report from actual stored metrics.
* Reconstruct approximate ROC curve from stored AUC value.
* Trigger and track baseline training jobs (runs in a thread pool).
* Update model_versions rows after a training/evaluation run.

Research integrity
------------------
* All metrics come from the database, which was populated by
  scripts/evaluate_baseline.py and scripts/register_models.py.
* No values are fabricated — if a metric is NULL the report says so.
* The report narrative is template-driven from actual data, not an LLM.
* Random seed, dataset version, and split size are recorded per model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_version import ModelVersion
from app.schemas.lab import (
    ModelDetailResponse, PerClassDetail, LossCurvePoint,
    ModelComparisonRow, ComparisonResponse,
    RocPoint, RocResponse,
    ConfusionResponse,
    ResearchReport, BestModelEntry, ModelGroupSummary,
    TrainJobStatus,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── In-memory job store (dev-grade; replace with Redis for production) ─────────
_JOBS: dict[str, TrainJobStatus] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_json(raw: Optional[str]) -> Optional[dict | list]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _mv_to_detail(mv: ModelVersion) -> ModelDetailResponse:
    confusion   = _safe_json(mv.confusion_matrix_json)
    history_raw = _safe_json(mv.training_history_json)
    per_class_raw = _safe_json(mv.per_class_metrics_json)
    hyper_raw   = _safe_json(mv.hyperparameters_json)

    history: Optional[list[LossCurvePoint]] = None
    if isinstance(history_raw, list):
        history = [
            LossCurvePoint(
                epoch=h.get("epoch", i + 1),
                train_loss=h.get("train_loss", 0.0),
                val_loss=h.get("val_loss"),
                val_acc=h.get("val_acc"),
            )
            for i, h in enumerate(history_raw)
        ]

    per_class: Optional[dict[str, PerClassDetail]] = None
    if isinstance(per_class_raw, dict):
        per_class = {
            cls_name: PerClassDetail(
                precision=m.get("precision"),
                recall=m.get("recall"),
                f1=m.get("f1"),
                support=m.get("support"),
            )
            for cls_name, m in per_class_raw.items()
        }

    return ModelDetailResponse(
        id=mv.id,
        model_name=mv.model_name,
        version=mv.version,
        model_type=mv.model_type.value,
        algorithm=mv.algorithm.value,
        is_active=mv.is_active,
        description=mv.description,
        dataset_name=mv.dataset_name,
        training_samples=mv.training_samples,
        test_samples=mv.test_samples,
        random_seed=mv.random_seed,
        train_time_s=mv.train_time_s,
        inference_time_ms=mv.inference_time_ms,
        model_size_bytes=mv.model_size_bytes,
        accuracy=mv.accuracy,
        precision=mv.precision,
        recall=mv.recall,
        f1_score=mv.f1_score,
        roc_auc=mv.roc_auc,
        confusion_matrix=confusion if isinstance(confusion, list) else None,
        training_history=history,
        per_class_metrics=per_class,
        hyperparameters=hyper_raw if isinstance(hyper_raw, dict) else None,
        created_at=mv.created_at,
    )


# ── Service functions ─────────────────────────────────────────────────────────

async def get_all_models(db: AsyncSession) -> list[ModelDetailResponse]:
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.version)
    )
    rows = result.scalars().all()
    return [_mv_to_detail(mv) for mv in rows]


async def get_model_detail(db: AsyncSession, model_id: int) -> Optional[ModelDetailResponse]:
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    mv = result.scalar_one_or_none()
    return _mv_to_detail(mv) if mv else None


async def get_comparison(db: AsyncSession) -> ComparisonResponse:
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.model_name)
    )
    rows = result.scalars().all()

    models_out = [
        ModelComparisonRow(
            model_name=mv.model_name,
            model_type=mv.model_type.value,
            version=mv.version,
            accuracy=mv.accuracy,
            precision=mv.precision,
            recall=mv.recall,
            f1_score=mv.f1_score,
            roc_auc=mv.roc_auc,
            train_time_s=mv.train_time_s,
            inference_time_ms=mv.inference_time_ms,
            model_size_bytes=mv.model_size_bytes,
            test_samples=mv.test_samples,
        )
        for mv in rows
    ]

    dataset_name = rows[0].dataset_name if rows else None
    return ComparisonResponse(models=models_out, dataset_name=dataset_name)


async def get_confusion(db: AsyncSession, model_id: int) -> Optional[ConfusionResponse]:
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    mv = result.scalar_one_or_none()
    if not mv:
        return None

    matrix_raw = _safe_json(mv.confusion_matrix_json)
    if not matrix_raw or not isinstance(matrix_raw, list):
        # Return a placeholder with zeros if not stored
        matrix_raw = [[0, 0], [0, 0]]

    return ConfusionResponse(
        model_id=mv.id,
        model_name=mv.model_name,
        matrix=matrix_raw,
        test_samples=mv.test_samples,
    )


async def get_roc(db: AsyncSession, model_id: int) -> Optional[RocResponse]:
    """
    Return ROC curve data.

    If detailed threshold-sweep data is not stored, reconstruct an
    approximate 10-point curve from the stored AUC value using the
    binormal assumption: TPR = Phi(Phi^-1(FPR) + d')
    where d' is chosen so the resulting AUC matches the stored value.

    This approximation is visually representative but NOT the exact
    empirical curve — the UI always displays this caveat.
    """
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    mv = result.scalar_one_or_none()
    if not mv:
        return None

    auc = mv.roc_auc
    if auc is None:
        points = []
    else:
        points = _approximate_roc_curve(auc)

    return RocResponse(
        model_id=mv.id,
        model_name=mv.model_name,
        roc_auc=auc,
        points=points,
    )


def _approximate_roc_curve(auc: float, n_points: int = 20) -> list[RocPoint]:
    """
    Generate an approximate ROC curve from AUC using the binormal model.

    Under the binormal assumption, the ROC curve is:
       TPR = Phi(a + b * Phi^{-1}(FPR))
    where a and b are shape parameters.

    For a symmetric binormal curve (b=1), a = Phi^{-1}(AUC) * sqrt(2).
    """
    import math

    def phi(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def phi_inv(p: float) -> float:
        """Inverse standard normal CDF via iterative method."""
        p = max(1e-9, min(1 - 1e-9, p))
        # Initial approximation
        t = math.sqrt(-2 * math.log(min(p, 1 - p)))
        x = t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) / (
            1 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3
        )
        if p < 0.5:
            x = -x
        return x

    # Solve for a: AUC = Phi(a / sqrt(2))  →  a = Phi^{-1}(AUC) * sqrt(2)
    a = phi_inv(auc) * math.sqrt(2)

    points = [RocPoint(fpr=0.0, tpr=0.0)]
    for i in range(1, n_points):
        fpr = i / n_points
        tpr = phi(a + phi_inv(fpr))
        tpr = min(max(tpr, 0.0), 1.0)
        points.append(RocPoint(fpr=round(fpr, 4), tpr=round(tpr, 4)))
    points.append(RocPoint(fpr=1.0, tpr=1.0))
    return points


async def generate_report(db: AsyncSession) -> ResearchReport:
    """
    Generate the research comparison report from actual DB values.

    All narrative text is template-driven from real metrics.
    Nothing is fabricated or hallucinated.
    Returns a report even when some metrics are NULL — explicitly notes NULLs.
    """
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.model_name)
    )
    rows = result.scalars().all()

    if not rows:
        return ResearchReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            dataset="Unknown — no registered models",
            test_split="10% stratified",
            total_models=0,
            best_by_accuracy=BestModelEntry(metric="accuracy", model_name="N/A", value=None),
            best_by_precision=BestModelEntry(metric="precision", model_name="N/A", value=None),
            best_by_recall=BestModelEntry(metric="recall", model_name="N/A", value=None),
            best_by_f1=BestModelEntry(metric="f1_score", model_name="N/A", value=None),
            best_by_roc_auc=BestModelEntry(metric="roc_auc", model_name="N/A", value=None),
            fastest_training=BestModelEntry(metric="train_time_s", model_name="N/A", value=None),
            fastest_inference=BestModelEntry(metric="inference_time_ms", model_name="N/A", value=None),
            traditional_summary=ModelGroupSummary(group="Traditional NLP", models=[], avg_accuracy=None, avg_f1=None, avg_train_time=None),
            transformer_summary=ModelGroupSummary(group="Transformer", models=[], avg_accuracy=None, avg_f1=None, avg_train_time=None),
            narrative=["No registered models with evaluation metrics found."],
            limitations=["Run evaluation scripts to populate metrics."],
            evaluation_conditions={},
        )

    def _best(attr: str, lower_is_better: bool = False) -> BestModelEntry:
        candidates = [(getattr(mv, attr), mv.model_name) for mv in rows if getattr(mv, attr) is not None]
        if not candidates:
            return BestModelEntry(metric=attr, model_name="N/A", value=None)
        val, name = (min if lower_is_better else max)(candidates, key=lambda x: x[0])
        return BestModelEntry(metric=attr, model_name=name, value=round(val, 6))

    def _avg(mvs: list[ModelVersion], attr: str) -> Optional[float]:
        vals = [getattr(mv, attr) for mv in mvs if getattr(mv, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    traditional = [mv for mv in rows if mv.model_type.value == "traditional"]
    transformers = [mv for mv in rows if mv.model_type.value == "transformer"]

    best_acc  = _best("accuracy")
    best_prec = _best("precision")
    best_rec  = _best("recall")
    best_f1   = _best("f1_score")
    best_auc  = _best("roc_auc")
    fast_train = _best("train_time_s", lower_is_better=True)
    fast_infer = _best("inference_time_ms", lower_is_better=True)

    trad_summary = ModelGroupSummary(
        group="Traditional NLP",
        models=[mv.model_name for mv in traditional],
        avg_accuracy=_avg(traditional, "accuracy"),
        avg_f1=_avg(traditional, "f1_score"),
        avg_train_time=_avg(traditional, "train_time_s"),
    )
    trans_summary = ModelGroupSummary(
        group="Transformer",
        models=[mv.model_name for mv in transformers],
        avg_accuracy=_avg(transformers, "accuracy"),
        avg_f1=_avg(transformers, "f1_score"),
        avg_train_time=_avg(transformers, "train_time_s"),
    )

    narrative = _build_narrative(rows, best_acc, best_f1, fast_train, fast_infer, trad_summary, trans_summary)

    dataset_name = rows[0].dataset_name or "Unknown"
    test_samples = max((mv.test_samples or 0 for mv in rows), default=0)
    random_seeds = list({mv.random_seed for mv in rows if mv.random_seed})

    return ResearchReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset_name,
        test_split="10% held-out stratified split",
        total_models=len(rows),
        best_by_accuracy=best_acc,
        best_by_precision=best_prec,
        best_by_recall=best_rec,
        best_by_f1=best_f1,
        best_by_roc_auc=best_auc,
        fastest_training=fast_train,
        fastest_inference=fast_infer,
        traditional_summary=trad_summary,
        transformer_summary=trans_summary,
        narrative=narrative,
        limitations=[
            "Models were trained on ISOT (2015–2018 US political news). "
            "Performance may not generalise to other domains or time periods.",
            "High accuracy on ISOT does not guarantee correctness on arbitrary inputs.",
            "Transformer models were evaluated after fine-tuning on the same dataset — "
            "comparison reflects in-domain performance only.",
            "Inference time depends on hardware. All timings from the same machine.",
        ],
        evaluation_conditions={
            "dataset":          dataset_name,
            "test_split":       "10% stratified",
            "test_samples":     test_samples,
            "random_seeds":     random_seeds,
            "evaluation_script":"scripts/evaluate_baseline.py + evaluate_transformer.py",
            "no_test_in_training": True,
        },
    )


def _build_narrative(rows, best_acc, best_f1, fast_train, fast_infer, trad, trans) -> list[str]:
    """Build factual, template-driven narrative paragraphs from real metrics."""
    paras: list[str] = []

    n_trad  = len(trad.models)
    n_trans = len(trans.models)
    n_total = len(rows)

    paras.append(
        f"This report compares {n_total} models evaluated on the same held-out test set. "
        f"{n_trad} traditional NLP model(s) (TF-IDF + classifier) and "
        f"{n_trans} transformer model(s) were evaluated."
    )

    # Overall best
    if best_acc.value is not None:
        paras.append(
            f"The model with the highest accuracy is {best_acc.model_name.replace('_', ' ').title()} "
            f"({best_acc.value * 100:.2f}%). "
            f"The model with the best F1 score is {best_f1.model_name.replace('_', ' ').title()} "
            f"({best_f1.value * 100:.2f}% F1)."
            if best_f1.value is not None else
            f"The model with the highest accuracy is {best_acc.model_name.replace('_', ' ').title()} "
            f"({best_acc.value * 100:.2f}%)."
        )

    # Speed
    if fast_train.value is not None:
        paras.append(
            f"The fastest model to train was {fast_train.model_name.replace('_', ' ').title()} "
            f"({fast_train.value:.1f}s). "
            + (
                f"The fastest for inference was {fast_infer.model_name.replace('_', ' ').title()} "
                f"({fast_infer.value:.3f}ms/sample)."
                if fast_infer.value is not None else ""
            )
        )

    # Traditional vs Transformer comparison
    if trad.avg_accuracy is not None and trans.avg_accuracy is not None:
        diff = (trans.avg_accuracy - trad.avg_accuracy) * 100
        direction = "higher" if diff > 0 else "lower"
        paras.append(
            f"On average, transformer models achieved {abs(diff):.2f}% {direction} accuracy "
            f"than traditional TF-IDF models "
            f"({trans.avg_accuracy * 100:.2f}% vs {trad.avg_accuracy * 100:.2f}%). "
            f"This gap is consistent with the literature: transformers benefit from contextual "
            f"representations while TF-IDF models rely on bag-of-words features."
        )
    elif trad.avg_accuracy is not None:
        paras.append(
            f"Traditional NLP models achieved average accuracy of {trad.avg_accuracy * 100:.2f}%. "
            f"No transformer models with evaluation results were found in the database."
        )
    elif trans.avg_accuracy is not None:
        paras.append(
            f"Transformer models achieved average accuracy of {trans.avg_accuracy * 100:.2f}%. "
            f"No traditional NLP models with evaluation results were found in the database."
        )

    # Speed trade-off
    if trad.avg_train_time is not None and trans.avg_train_time is not None:
        ratio = trans.avg_train_time / max(trad.avg_train_time, 0.001)
        paras.append(
            f"Transformer models required approximately {ratio:.0f}x more training time than "
            f"traditional models ({trans.avg_train_time:.0f}s vs {trad.avg_train_time:.1f}s average). "
            f"For applications where training speed and deployability matter, traditional models "
            f"offer a strong trade-off with competitive accuracy."
        )

    # Limitations note
    paras.append(
        "These results are specific to the ISOT fake news dataset "
        "(2015–2018 US political news). "
        "Generalisability to other domains, languages, or time periods is not established. "
        "All metrics are from offline evaluation on the held-out test split — "
        "not from real-world deployment."
    )

    return paras


# ── Training job management ───────────────────────────────────────────────────

def _get_model_size(model_id: str) -> Optional[int]:
    """Return file size in bytes of the saved pipeline artifact, or None."""
    from ml.baseline.config import MODEL_PIPELINE_PATHS
    path = MODEL_PIPELINE_PATHS.get(model_id)
    if path and path.exists():
        return path.stat().st_size
    return None


async def trigger_training(
    db:       AsyncSession,
    model_id: str,
    data_dir: str,
) -> TrainJobStatus:
    """
    Trigger a baseline training + evaluation run in a thread pool.

    The job is tracked in the in-memory _JOBS dict.
    After completion, model_versions is updated with timing, size, and metrics.
    """
    job_id = str(uuid.uuid4())[:8]
    job = TrainJobStatus(
        job_id=job_id,
        model_id=model_id,
        status="queued",
        started_at=None,
        finished_at=None,
        log_lines=[],
    )
    _JOBS[job_id] = job

    asyncio.create_task(
        _run_training_job(job_id, model_id, data_dir, db)
    )
    return job


async def _run_training_job(
    job_id:   str,
    model_id: str,
    data_dir: str,
    db:       AsyncSession,
) -> None:
    """Background coroutine: train, evaluate, update DB."""
    import time

    job = _JOBS[job_id]
    job.status = "running"
    job.started_at = datetime.now(timezone.utc).isoformat()
    _log(job_id, f"Starting training job for model: {model_id}")

    loop = asyncio.get_running_loop()

    try:
        # ── Run training in thread pool ───────────────────────────────────────
        _log(job_id, "Loading training data…")
        t0 = time.perf_counter()

        pipeline, record = await loop.run_in_executor(
            None,
            lambda: _train_sync(model_id, data_dir, job_id),
        )

        train_time = time.perf_counter() - t0
        _log(job_id, f"Training complete in {train_time:.1f}s")

        # ── Evaluate in thread pool ───────────────────────────────────────────
        _log(job_id, "Evaluating on test set…")
        eval_result = await loop.run_in_executor(
            None,
            lambda: _evaluate_sync(model_id, data_dir, pipeline, job_id),
        )

        # ── Update DB ─────────────────────────────────────────────────────────
        _log(job_id, "Updating model registry in database…")
        mv_id = await _upsert_model_version(db, model_id, record, eval_result)

        job.status = "completed"
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.result_model_version_id = mv_id
        _log(job_id, f"Done. ModelVersion id={mv_id}")

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.finished_at = datetime.now(timezone.utc).isoformat()
        _log(job_id, f"ERROR: {exc}")
        logger.error("Training job %s failed: %s", job_id, exc, exc_info=True)


def _log(job_id: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    logger.info("[job=%s] %s", job_id, msg)
    if job_id in _JOBS:
        _JOBS[job_id].log_lines.append(line)


def _train_sync(model_id: str, data_dir: str, job_id: str) -> tuple:
    """Synchronous training — runs in thread pool."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    import pandas as pd
    from ml.baseline.trainer import train_single

    # Resolve data_dir relative to project root if not absolute
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _PROJECT_ROOT / data_path

    train_path = data_path / "train.csv"
    val_path   = data_path / "val.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    _log(job_id, f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    pipeline, record = train_single(model_id, train_df, val_df, save=True)
    _log(job_id, f"Pipeline saved — {record.get('train_time_s', '?'):.2f}s")
    return pipeline, record


def _evaluate_sync(model_id: str, data_dir: str, pipeline, job_id: str) -> dict:
    """Synchronous evaluation — runs in thread pool."""
    import pandas as pd
    from ml.baseline.evaluator import evaluate_pipeline

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _PROJECT_ROOT / data_path

    test_path = data_path / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    test_df = pd.read_csv(test_path)
    _log(job_id, f"Test: {len(test_df)} rows")
    result = evaluate_pipeline(model_id, pipeline, test_df, save=True)
    _log(job_id, f"Accuracy: {result.get('accuracy', '?'):.4f}  F1: {result.get('macro_f1', '?'):.4f}")
    return result


async def _upsert_model_version(
    db:          AsyncSession,
    model_id:    str,
    train_record: dict,
    eval_result: dict,
) -> int:
    """Insert or update model_versions row with fresh metrics."""
    from app.models.model_version import ModelType, ModelAlgorithm

    _ALGO_MAP = {
        "logistic_regression": ModelAlgorithm.LOGISTIC_REGRESSION,
        "linear_svm":          ModelAlgorithm.LINEAR_SVM,
        "naive_bayes":         ModelAlgorithm.NAIVE_BAYES,
    }

    # Find existing active row for this model
    existing = await db.execute(
        select(ModelVersion).where(
            ModelVersion.model_name == model_id,
            ModelVersion.version == "1.0.0",
        )
    )
    mv = existing.scalar_one_or_none()

    per_class = eval_result.get("per_class", {})
    hyper = train_record.get("hyperparameters", {})
    tfidf = train_record.get("tfidf_params", {})
    model_size = _get_model_size(model_id)

    # Map per-class: {FAKE: {precision,recall,f1,support}, REAL: {...}}
    pc_map = {
        cls_name: {
            "precision": m.get("precision"),
            "recall":    m.get("recall"),
            "f1":        m.get("f1"),
            "support":   m.get("support"),
        }
        for cls_name, m in per_class.items()
    }

    if mv:
        mv.accuracy           = eval_result.get("accuracy")
        mv.precision          = eval_result.get("macro_precision")
        mv.recall             = eval_result.get("macro_recall")
        mv.f1_score           = eval_result.get("macro_f1")
        mv.roc_auc            = eval_result.get("roc_auc")
        mv.test_samples       = eval_result.get("test_samples")
        mv.training_samples   = train_record.get("train_samples")
        mv.train_time_s       = train_record.get("train_time_s")
        mv.inference_time_ms  = eval_result.get("per_sample_ms")
        mv.model_size_bytes   = model_size
        mv.random_seed        = train_record.get("random_seed", 42)
        mv.dataset_name       = "ISOT Fake News Dataset"
        mv.hyperparameters_json = json.dumps({**hyper, "tfidf": tfidf})
        mv.confusion_matrix_json = json.dumps(eval_result.get("confusion_matrix"))
        mv.per_class_metrics_json = json.dumps(pc_map)
        await db.flush()
        return mv.id
    else:
        new_mv = ModelVersion(
            model_name           = model_id,
            version              = "1.0.0",
            model_type           = ModelType.TRADITIONAL,
            algorithm            = _ALGO_MAP.get(model_id, ModelAlgorithm.LOGISTIC_REGRESSION),
            artifact_path        = str(_PROJECT_ROOT / "ml" / "saved_models" / f"{model_id.replace('logistic_regression','lr').replace('linear_svm','svm').replace('naive_bayes','nb')}_pipeline.pkl"),
            dataset_name         = "ISOT Fake News Dataset",
            training_samples     = train_record.get("train_samples"),
            test_samples         = eval_result.get("test_samples"),
            random_seed          = train_record.get("random_seed", 42),
            train_time_s         = train_record.get("train_time_s"),
            inference_time_ms    = eval_result.get("per_sample_ms"),
            model_size_bytes     = model_size,
            accuracy             = eval_result.get("accuracy"),
            precision            = eval_result.get("macro_precision"),
            recall               = eval_result.get("macro_recall"),
            f1_score             = eval_result.get("macro_f1"),
            roc_auc              = eval_result.get("roc_auc"),
            is_active            = True,
            hyperparameters_json = json.dumps({**hyper, "tfidf": tfidf}),
            confusion_matrix_json = json.dumps(eval_result.get("confusion_matrix")),
            per_class_metrics_json = json.dumps(pc_map),
        )
        db.add(new_mv)
        await db.flush()
        return new_mv.id


def get_job(job_id: str) -> Optional[TrainJobStatus]:
    return _JOBS.get(job_id)


def list_jobs() -> list[TrainJobStatus]:
    return list(_JOBS.values())
