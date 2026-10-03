"""
Baseline model trainer.

Trains three sklearn Pipeline objects — each bundles a TF-IDF vectorizer
with one classifier — and saves them as .pkl files.

Pipeline structure (same for all three models):
    Step 1: TfidfVectorizer   (fitted on training data only)
    Step 2: Classifier        (LR | CalibratedSVM | MultinomialNB)

Using sklearn Pipelines means:
  * The vectorizer and classifier are always saved/loaded together.
  * Calling pipeline.predict(raw_text_list) applies both steps in order.
  * No accidental leakage — fit() only touches the data passed to it.

Data leakage prevention:
  * TF-IDF is fitted inside Pipeline.fit(X_train, y_train).
  * Val and test sets are only transformed (never used to fit).
  * Processed splits are loaded from disk (built by prepare_dataset.py).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.baseline.config import (
    RANDOM_SEED,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_ANALYZER,
    LR_C, LR_MAX_ITER, LR_SOLVER, LR_CLASS_WEIGHT,
    SVM_C, SVM_MAX_ITER, SVM_CLASS_WEIGHT, SVM_CALIBRATION_CV,
    NB_ALPHA,
    MODEL_PIPELINE_PATHS, SAVED_MODELS,
)

logger = logging.getLogger(__name__)

# ── Shared TF-IDF factory ─────────────────────────────────────────────────────

def _make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        analyzer=TFIDF_ANALYZER,
        strip_accents="unicode",
        lowercase=True,         # belt-and-suspenders: text already lowercased
    )


# ── Pipeline factories ────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression (supports predict_proba natively)."""
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf",   LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
            class_weight=LR_CLASS_WEIGHT,
            random_state=RANDOM_SEED,
        )),
    ])


def build_svm_pipeline() -> Pipeline:
    """
    TF-IDF + LinearSVC wrapped in CalibratedClassifierCV.

    LinearSVC is faster than SVC(kernel='linear') but does not output
    probabilities.  CalibratedClassifierCV adds Platt scaling (sigmoid
    calibration) so predict_proba() returns meaningful confidence scores.
    """
    base_svm = LinearSVC(
        C=SVM_C,
        max_iter=SVM_MAX_ITER,
        class_weight=SVM_CLASS_WEIGHT,
        random_state=RANDOM_SEED,
    )
    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        cv=SVM_CALIBRATION_CV,
        method="sigmoid",
    )
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf",   calibrated_svm),
    ])


def build_nb_pipeline() -> Pipeline:
    """
    TF-IDF + MultinomialNB.

    MultinomialNB supports predict_proba natively via smoothed log-likelihoods.
    TF-IDF features are always >= 0, satisfying MultinomialNB's requirement.
    """
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf",   MultinomialNB(alpha=NB_ALPHA)),
    ])


# ── Training record ───────────────────────────────────────────────────────────

def _make_training_record(
    model_id:       str,
    pipeline:       Pipeline,
    train_size:     int,
    val_size:       int,
    train_time_s:   float,
    hyperparams:    dict,
) -> dict:
    """Build a machine-readable training configuration record."""
    return {
        "model_id":        model_id,
        "algorithm":       type(pipeline.named_steps["clf"]).__name__,
        "vectorizer":      type(pipeline.named_steps["tfidf"]).__name__,
        "train_samples":   train_size,
        "val_samples":     val_size,
        "train_time_s":    round(train_time_s, 3),
        "random_seed":     RANDOM_SEED,
        "hyperparameters": hyperparams,
        "tfidf_params": {
            "max_features":  TFIDF_MAX_FEATURES,
            "ngram_range":   list(TFIDF_NGRAM_RANGE),
            "sublinear_tf":  TFIDF_SUBLINEAR_TF,
            "min_df":        TFIDF_MIN_DF,
            "max_df":        TFIDF_MAX_DF,
        },
    }


# ── Save / load ───────────────────────────────────────────────────────────────

def save_pipeline(pipeline: Pipeline, model_id: str) -> Path:
    """Serialise a fitted pipeline to disk with joblib."""
    path = MODEL_PIPELINE_PATHS[model_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    size_kb = path.stat().st_size / 1024
    logger.info("Saved %s pipeline → %s  (%.0f KB)", model_id, path, size_kb)
    return path


def load_pipeline(model_id: str) -> Pipeline:
    """Load a fitted pipeline from disk."""
    path = MODEL_PIPELINE_PATHS[model_id]
    if not path.exists():
        raise FileNotFoundError(
            f"No saved pipeline for '{model_id}' at {path}.\n"
            "Run: python scripts/train_baseline.py"
        )
    pipeline = joblib.load(path)
    logger.info("Loaded %s pipeline from %s", model_id, path)
    return pipeline


# ── Public training API ───────────────────────────────────────────────────────

def train_single(
    model_id:  str,
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    save:      bool = True,
) -> tuple[Pipeline, dict]:
    """
    Train one model and optionally save it.

    Parameters
    ----------
    model_id  : "logistic_regression" | "linear_svm" | "naive_bayes"
    train_df  : DataFrame with 'cleaned_text' and 'label' columns.
    val_df    : Validation DataFrame (not used for fitting — logged only).
    save      : Write the fitted pipeline to disk.

    Returns
    -------
    (fitted_pipeline, training_record)
    """
    builders = {
        "logistic_regression": build_lr_pipeline,
        "linear_svm":          build_svm_pipeline,
        "naive_bayes":         build_nb_pipeline,
    }
    hyperparams_map = {
        "logistic_regression": {"C": LR_C, "max_iter": LR_MAX_ITER, "solver": LR_SOLVER},
        "linear_svm":          {"C": SVM_C, "max_iter": SVM_MAX_ITER, "calibration_cv": SVM_CALIBRATION_CV},
        "naive_bayes":         {"alpha": NB_ALPHA},
    }

    if model_id not in builders:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Choose from: {list(builders.keys())}"
        )

    text_col = "cleaned_text" if "cleaned_text" in train_df.columns else "text"
    X_train = train_df[text_col].fillna("").tolist()
    y_train = train_df["label"].tolist()

    logger.info(
        "Training %s on %d samples (text_col='%s')...",
        model_id, len(X_train), text_col,
    )

    pipeline = builders[model_id]()

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    logger.info("  Training complete in %.2f s", train_time)

    record = _make_training_record(
        model_id=model_id,
        pipeline=pipeline,
        train_size=len(X_train),
        val_size=len(val_df),
        train_time_s=train_time,
        hyperparams=hyperparams_map[model_id],
    )

    if save:
        artifact_path = save_pipeline(pipeline, model_id)
        record["artifact_path"] = str(artifact_path)

    # Save training record alongside model
    record_path = SAVED_MODELS / f"{model_id}_training_record.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("  Training record → %s", record_path)

    return pipeline, record


def train_all(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    save:     bool = True,
) -> dict[str, tuple[Pipeline, dict]]:
    """
    Train all three baseline models.

    Returns
    -------
    dict mapping model_id → (fitted_pipeline, training_record)
    """
    results: dict[str, tuple[Pipeline, dict]] = {}
    for model_id in ["logistic_regression", "linear_svm", "naive_bayes"]:
        pipeline, record = train_single(model_id, train_df, val_df, save=save)
        results[model_id] = (pipeline, record)

    logger.info("All baseline models trained.")
    return results
