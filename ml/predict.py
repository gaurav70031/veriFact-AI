"""
ml/predict.py
=============
Unified prediction interface for the VeriFact AI ML engine.

This is the primary public API used by the FastAPI backend (via
backend/app/ml/model_registry.py) and by scripts that need predictions.

It wraps the separate baseline (sklearn) and transformer (DistilBERT/BERT)
inference modules behind a single, consistent interface.

Confidence threshold
--------------------
Predictions below CONFIDENCE_THRESHOLD are labelled UNVERIFIED instead
of FAKE or REAL.  The threshold can be overridden per-call or changed
globally via the VERIFACT_CONFIDENCE_THRESHOLD environment variable.

Default threshold: 0.55  (5pp above chance — conservative)

Returned structure (PredictionResult)
--------------------------------------
{
    "label":             "FAKE" | "REAL" | "UNVERIFIED",
    "is_fake":           bool | None,
    "confidence":        float,          # max(fake_prob, real_prob)
    "fake_probability":  float,
    "real_probability":  float,
    "model_id":          str,
    "inference_time_ms": float,
    "below_threshold":   bool,           # True when confidence < threshold
    "threshold_used":    float,
    "preprocessing":     str,            # "baseline" | "transformer" | "none"
    "limitations":       list[str],      # always populated
}

Usage
-----
    from ml.predict import predict, predict_all, predict_ensemble

    # Single model
    result = predict("5G towers caused COVID-19", model_id="logistic_regression")

    # All models
    all_results = predict_all("5G towers caused COVID-19")

    # Weighted ensemble verdict
    verdict = predict_ensemble("5G towers caused COVID-19")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Confidence threshold ──────────────────────────────────────────────────────
# Loaded from environment variable if set, otherwise use default.
_DEFAULT_THRESHOLD: float = 0.55

def _load_threshold() -> float:
    raw = os.getenv("VERIFACT_CONFIDENCE_THRESHOLD", "")
    if raw:
        try:
            val = float(raw)
            if 0.0 < val < 1.0:
                return val
            logger.warning(
                "VERIFACT_CONFIDENCE_THRESHOLD=%s is out of range (0, 1) — using default %.2f",
                raw, _DEFAULT_THRESHOLD,
            )
        except ValueError:
            logger.warning(
                "VERIFACT_CONFIDENCE_THRESHOLD=%s is not a valid float — using default %.2f",
                raw, _DEFAULT_THRESHOLD,
            )
    return _DEFAULT_THRESHOLD

CONFIDENCE_THRESHOLD: float = _load_threshold()

# ── Standard prediction limitations ──────────────────────────────────────────
_STANDARD_LIMITATIONS = [
    "Model predictions are based on statistical patterns from training data, "
    "not factual knowledge.",
    "Confidence scores reflect pattern matching, not factual certainty.",
    "Models were trained on ISOT (2015–2018 US political news) and may not "
    "generalise to other topics, languages, or time periods.",
    "High model confidence does not mean an article is factually correct or incorrect.",
]

_UNVERIFIED_LIMITATIONS = [
    "Model confidence was below the threshold — prediction withheld to avoid misleading output.",
] + _STANDARD_LIMITATIONS


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_threshold(
    result: dict,
    threshold: float,
) -> dict:
    """
    Apply confidence threshold to a raw prediction dict.
    If confidence < threshold, replace label with UNVERIFIED and set
    below_threshold=True.
    """
    conf = result.get("confidence", 0.5)
    below = conf < threshold

    result["below_threshold"]  = below
    result["threshold_used"]   = threshold
    result["limitations"]      = (
        _UNVERIFIED_LIMITATIONS if below else _STANDARD_LIMITATIONS
    )

    if below:
        result["label"]   = "UNVERIFIED"
        result["is_fake"] = None
        logger.debug(
            "Prediction below threshold (%.3f < %.3f): label→UNVERIFIED",
            conf, threshold,
        )

    return result


def _add_preprocessing_key(result: dict, preprocessing: str) -> dict:
    result["preprocessing"] = preprocessing
    return result


# ── Baseline prediction ───────────────────────────────────────────────────────

def predict_baseline(
    text:       str,
    model_id:   str   = "logistic_regression",
    threshold:  float = CONFIDENCE_THRESHOLD,
    clean:      bool  = True,
) -> dict:
    """
    Run one TF-IDF baseline model.

    Parameters
    ----------
    text      : Input text (article, claim, extracted article body).
    model_id  : "logistic_regression" | "linear_svm" | "naive_bayes"
    threshold : Confidence threshold below which label→UNVERIFIED.
    clean     : Apply baseline text cleaning before inference.

    Returns
    -------
    PredictionResult dict (see module docstring).
    """
    from ml.baseline.inference import predict_single
    result = predict_single(text, model_id=model_id, clean=clean)
    result = _apply_threshold(result, threshold)
    result = _add_preprocessing_key(result, "baseline")
    return result


def predict_all_baseline(
    text:      str,
    threshold: float = CONFIDENCE_THRESHOLD,
    clean:     bool  = True,
) -> dict[str, dict]:
    """
    Run all three baseline models on the same text.

    Returns
    -------
    {model_id: PredictionResult}
    """
    from ml.baseline.inference import predict_all_models
    raw_results = predict_all_models(text, clean=clean)
    return {
        mid: _add_preprocessing_key(
            _apply_threshold(pred, threshold),
            "baseline",
        )
        for mid, pred in raw_results.items()
    }


# ── Transformer prediction ────────────────────────────────────────────────────

def predict_transformer(
    text:       str,
    threshold:  float = CONFIDENCE_THRESHOLD,
    model_dir:  Optional[str] = None,
) -> dict:
    """
    Run the DistilBERT transformer model.

    Long articles are automatically chunked (overlapping 510-token windows)
    and aggregated via weighted mean.  See ml/transformer/inference.py for details.

    Parameters
    ----------
    text      : Raw article text — transformer handles its own light cleaning.
    threshold : Confidence threshold.
    model_dir : Path to versioned model directory. Uses default if None.

    Returns
    -------
    PredictionResult dict (see module docstring).
    """
    from ml.transformer.inference import predict_text, VERSIONED_MODEL_DIR
    from pathlib import Path

    dir_path = Path(model_dir) if model_dir else VERSIONED_MODEL_DIR
    result   = predict_text(text, model_dir=dir_path)
    result   = _apply_threshold(result, threshold)
    result   = _add_preprocessing_key(result, "transformer")
    result["model_id"] = "distilbert"
    return result


# ── Unified predict() ─────────────────────────────────────────────────────────

def predict(
    text:       str,
    model_id:   str   = "logistic_regression",
    threshold:  float = CONFIDENCE_THRESHOLD,
    clean:      bool  = True,
) -> dict:
    """
    Predict for any model by model_id.

    Parameters
    ----------
    text      : Input text.
    model_id  : Any of "logistic_regression", "linear_svm", "naive_bayes",
                "distilbert", "bert" (aliases to distilbert if only one is trained).
    threshold : Confidence threshold.
    clean     : Apply cleaning before baseline inference (ignored for transformer).

    Returns
    -------
    PredictionResult dict.
    """
    transformer_ids = {"distilbert", "bert", "distilbert-base-uncased", "bert-base-uncased"}
    baseline_ids    = {"logistic_regression", "linear_svm", "naive_bayes"}

    if model_id in transformer_ids:
        return predict_transformer(text, threshold=threshold)
    elif model_id in baseline_ids:
        return predict_baseline(text, model_id=model_id, threshold=threshold, clean=clean)
    else:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Valid IDs: {sorted(baseline_ids | transformer_ids)}"
        )


# ── Ensemble prediction ───────────────────────────────────────────────────────

# Ensemble weights — must match backend/app/services/analysis_service.py
_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "logistic_regression": 0.20,
    "linear_svm":          0.30,
    "naive_bayes":         0.10,
    "distilbert":          0.40,
}


def predict_ensemble(
    text:                str,
    threshold:           float          = CONFIDENCE_THRESHOLD,
    include_transformer: bool           = True,
    custom_weights:      Optional[dict] = None,
) -> dict:
    """
    Run all available models and produce a weighted ensemble verdict.

    Parameters
    ----------
    text                : Input text.
    threshold           : Confidence threshold for the ensemble verdict.
    include_transformer : Whether to include DistilBERT (may not be loaded).
    custom_weights      : Override default ensemble weights.

    Returns
    -------
    {
        "label":             str,
        "is_fake":           bool | None,
        "confidence":        float,
        "fake_probability":  float,
        "real_probability":  float,
        "model_predictions": {model_id: PredictionResult},
        "ensemble_weights":  dict,
        "below_threshold":   bool,
        "threshold_used":    float,
        "limitations":       list[str],
    }
    """
    weights = custom_weights or _ENSEMBLE_WEIGHTS
    model_predictions: dict[str, dict] = {}

    # Baseline models
    try:
        model_predictions.update(predict_all_baseline(text, threshold=0.0, clean=True))
    except Exception as exc:
        logger.warning("Baseline models unavailable: %s", exc)

    # Transformer
    if include_transformer:
        try:
            model_predictions["distilbert"] = predict_transformer(
                text, threshold=0.0
            )
        except Exception as exc:
            logger.warning("Transformer unavailable: %s", exc)

    if not model_predictions:
        return {
            "label":            "UNVERIFIED",
            "is_fake":          None,
            "confidence":       0.5,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "model_predictions": {},
            "ensemble_weights":  {},
            "below_threshold":   True,
            "threshold_used":    threshold,
            "limitations":       _UNVERIFIED_LIMITATIONS + [
                "No trained models were found. Run the training scripts first."
            ],
        }

    # Weighted average
    weighted_fake = weighted_real = total_weight = 0.0
    for mid, pred in model_predictions.items():
        w = weights.get(mid, 0.1)
        weighted_fake += w * pred.get("fake_probability", 0.5)
        weighted_real += w * pred.get("real_probability", 0.5)
        total_weight  += w

    if total_weight == 0:
        avg_fake = avg_real = 0.5
    else:
        avg_fake = weighted_fake / total_weight
        avg_real = weighted_real / total_weight

    confidence = max(avg_fake, avg_real)

    # Determine label
    if confidence < threshold:
        label   = "UNVERIFIED"
        is_fake = None
    elif avg_fake > avg_real:
        label   = "FAKE"
        is_fake = True
    else:
        label   = "REAL"
        is_fake = False

    # Also handle MIXED (close to 50/50)
    if confidence < 0.55 and threshold <= 0.55:
        label   = "MIXED"
        is_fake = None

    limitations = _UNVERIFIED_LIMITATIONS if confidence < threshold else _STANDARD_LIMITATIONS

    return {
        "label":             label,
        "is_fake":           is_fake,
        "confidence":        round(confidence,  6),
        "fake_probability":  round(avg_fake,    6),
        "real_probability":  round(avg_real,    6),
        "model_predictions": model_predictions,
        "ensemble_weights":  {mid: weights.get(mid, 0.1) for mid in model_predictions},
        "below_threshold":   confidence < threshold,
        "threshold_used":    threshold,
        "limitations":       limitations,
    }


# ── Predict from URL (convenience wrapper) ────────────────────────────────────

def predict_url(
    url:       str,
    model_id:  str   = "logistic_regression",
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Extract article text from a URL and predict.

    Uses the backend's article extractor when available, otherwise
    falls back to a simple requests-based fetch.

    Parameters
    ----------
    url      : Public HTTPS URL of a news article.
    model_id : Model to use (or "ensemble" for all models).
    threshold: Confidence threshold.

    Returns
    -------
    PredictionResult dict with an extra "source_url" key.
    """
    # Try backend extractor first
    text: Optional[str] = None
    try:
        import sys
        from pathlib import Path
        backend_root = Path(__file__).resolve().parents[1] / "backend"
        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))

        import asyncio
        from app.extraction.article_extractor import extract_article
        article = asyncio.run(extract_article(url))
        text = f"{article.title or ''} {article.text}".strip()
        logger.info("Extracted article from %s (%d chars)", url, len(text))
    except Exception as exc:
        logger.warning("Backend extractor failed (%s), trying requests…", exc)

    # Simple fallback
    if not text:
        try:
            import requests
            resp = requests.get(url, timeout=15, headers={"User-Agent": "VeriFact/1.0"})
            resp.raise_for_status()
            # Strip HTML
            import re, html as html_module
            raw = html_module.unescape(re.sub(r"<[^>]+>", " ", resp.text))
            text = re.sub(r"\s+", " ", raw).strip()[:10_000]
        except Exception as exc2:
            raise RuntimeError(f"Could not fetch URL '{url}': {exc2}") from exc2

    if model_id == "ensemble":
        result = predict_ensemble(text, threshold=threshold)
    else:
        result = predict(text, model_id=model_id, threshold=threshold)

    result["source_url"] = url
    return result


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(
    texts:     list[str],
    model_id:  str   = "logistic_regression",
    threshold: float = CONFIDENCE_THRESHOLD,
    clean:     bool  = True,
) -> list[dict]:
    """
    Predict a list of texts using one baseline model (vectorised, fast).

    Parameters
    ----------
    texts     : List of raw input texts.
    model_id  : Baseline model ID (transformer not supported in batch mode
                because each article may require different chunk counts).
    threshold : Confidence threshold.
    clean     : Apply cleaning before inference.

    Returns
    -------
    List of PredictionResult dicts in the same order as `texts`.
    """
    from ml.baseline.inference import predict_batch as _predict_batch
    raw_results = _predict_batch(texts, model_id=model_id, clean=clean)
    return [
        _add_preprocessing_key(
            _apply_threshold(r, threshold),
            "baseline",
        )
        for r in raw_results
    ]
