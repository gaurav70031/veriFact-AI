"""
Baseline model inference.

This module is the single interface used by the FastAPI backend.
It:
  1. Loads saved pipelines lazily (once, on first call).
  2. Applies the same text cleaning used at training time.
  3. Returns a structured prediction dict with label + confidence.

Design
------
* Lazy singleton loading via module-level cache — avoids reloading on every
  request without requiring application-level DI.
* `predict_single()` is the primary public function called by the backend.
* `predict_batch()` is available for bulk processing.
* All models use `predict_proba()` — SVM has been calibrated during training.

Returned structure
------------------
{
    "label":            "FAKE" | "REAL",
    "is_fake":          bool,
    "confidence":       float,          # max(fake_prob, real_prob)
    "fake_probability": float,
    "real_probability": float,
    "model_id":         str,
}
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import numpy as np

from ml.baseline.config import MODEL_IDS, ID2LABEL
from ml.baseline.trainer import load_pipeline
from ml.preprocessing.cleaner import clean_for_baseline

logger = logging.getLogger(__name__)

# ── Pipeline cache ────────────────────────────────────────────────────────────
# Populated lazily on first call to predict_single() or get_pipeline().
_pipeline_cache: dict[str, Any] = {}


def get_pipeline(model_id: str):
    """Return (and cache) a loaded pipeline for `model_id`."""
    if model_id not in _pipeline_cache:
        _pipeline_cache[model_id] = load_pipeline(model_id)
    return _pipeline_cache[model_id]


def load_all_pipelines() -> None:
    """
    Eagerly load all three pipelines into the cache.
    Call this at FastAPI startup to avoid first-request latency.
    """
    for model_id in MODEL_IDS:
        get_pipeline(model_id)
    logger.info("All baseline pipelines loaded into cache.")


def clear_cache() -> None:
    """Evict all cached pipelines (useful for testing)."""
    _pipeline_cache.clear()


# ── Core prediction function ──────────────────────────────────────────────────

def predict_single(
    text:     str,
    model_id: str = "logistic_regression",
    clean:    bool = True,
) -> dict:
    """
    Predict fake/real for a single text string.

    Parameters
    ----------
    text     : Raw input text (article body, claim, etc.).
    model_id : Which model to use. One of MODEL_IDS.
    clean    : Apply baseline text cleaning before inference.
               Set False only if text is already cleaned.

    Returns
    -------
    dict with keys: label, is_fake, confidence,
                    fake_probability, real_probability, model_id,
                    inference_time_ms.
    """
    if model_id not in MODEL_IDS:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Choose from: {MODEL_IDS}"
        )

    if not text or not isinstance(text, str):
        raise ValueError("text must be a non-empty string.")

    processed = clean_for_baseline(text) if clean else text
    if not processed:
        # Text collapsed to empty after cleaning — return uncertain prediction
        logger.warning(
            "Input text collapsed to empty after cleaning for model '%s'.", model_id
        )
        return {
            "label":            "UNVERIFIED",
            "is_fake":          None,
            "confidence":       0.5,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "model_id":         model_id,
            "inference_time_ms": 0.0,
            "warning":          "Input text too short or empty after cleaning.",
        }

    pipeline = get_pipeline(model_id)

    t0 = time.perf_counter()
    proba = pipeline.predict_proba([processed])[0]   # shape (2,)
    inference_ms = (time.perf_counter() - t0) * 1000

    classes     = list(pipeline.classes_)
    fake_idx    = classes.index(0) if 0 in classes else 0
    real_idx    = classes.index(1) if 1 in classes else 1

    fake_prob   = float(proba[fake_idx])
    real_prob   = float(proba[real_idx])
    is_fake     = fake_prob > real_prob
    label       = "FAKE" if is_fake else "REAL"
    confidence  = max(fake_prob, real_prob)

    return {
        "label":            label,
        "is_fake":          is_fake,
        "confidence":       round(confidence,  6),
        "fake_probability": round(fake_prob,   6),
        "real_probability": round(real_prob,   6),
        "model_id":         model_id,
        "inference_time_ms": round(inference_ms, 3),
    }


def predict_all_models(
    text:  str,
    clean: bool = True,
) -> dict[str, dict]:
    """
    Run all three baseline models on the same text.
    Returns {model_id: prediction_dict}.
    Useful for ensemble voting and for the backend's analysis pipeline.
    """
    results: dict[str, dict] = {}
    for model_id in MODEL_IDS:
        results[model_id] = predict_single(text, model_id=model_id, clean=clean)
    return results


def predict_batch(
    texts:    list[str],
    model_id: str = "logistic_regression",
    clean:    bool = True,
) -> list[dict]:
    """
    Predict a list of texts in one vectorise call (faster than looping).

    Returns a list of prediction dicts in the same order as `texts`.
    """
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model_id '{model_id}'.")

    if not texts:
        return []

    processed = [clean_for_baseline(t) if clean else t for t in texts]
    pipeline  = get_pipeline(model_id)
    classes   = list(pipeline.classes_)
    fake_idx  = classes.index(0) if 0 in classes else 0
    real_idx  = classes.index(1) if 1 in classes else 1

    t0    = time.perf_counter()
    proba = pipeline.predict_proba(processed)   # shape (n, 2)
    total_ms = (time.perf_counter() - t0) * 1000
    per_ms   = total_ms / len(texts)

    results = []
    for i, text in enumerate(texts):
        fake_p = float(proba[i, fake_idx])
        real_p = float(proba[i, real_idx])
        is_fake = fake_p > real_p
        results.append({
            "label":            "FAKE" if is_fake else "REAL",
            "is_fake":          is_fake,
            "confidence":       round(max(fake_p, real_p), 6),
            "fake_probability": round(fake_p, 6),
            "real_probability": round(real_p, 6),
            "model_id":         model_id,
            "inference_time_ms": round(per_ms, 3),
        })
    return results
