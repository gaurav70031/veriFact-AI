"""
ml/evaluation.py
================
Unified evaluation interface for the ML engine.

Wraps ml.baseline.evaluator and ml.transformer.evaluator behind a single
consistent API so evaluation scripts, the model registry, and the backend
API endpoint can all call one place.

Returned evaluation record
--------------------------
{
    "model_id":          str,
    "model_type":        "baseline" | "transformer",
    "test_samples":      int,
    "accuracy":          float,
    "precision":         float,   # macro
    "recall":            float,   # macro
    "f1_score":          float,   # macro
    "weighted_f1":       float,
    "roc_auc":           float | None,
    "per_class": {
        "FAKE": {"precision", "recall", "f1", "support"},
        "REAL": {"precision", "recall", "f1", "support"},
    },
    "confusion_matrix":  [[int, int], [int, int]],   # [[TN,FP],[FN,TP]]
    "inference_time_ms": float,
    "per_sample_ms":     float,
}

Public API
----------
  evaluate_baseline(model_id, test_df)   → dict
  evaluate_transformer(model_dir, test_csv) → dict
  evaluate_all_baseline(test_df)         → dict[model_id, dict]
  load_saved_results()                   → dict
  format_summary(results)                → str
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ml.baseline.config import (
    MODEL_IDS, MODEL_PIPELINE_PATHS, ALL_EVAL_PATH, EVAL_RESULTS_DIR,
)
from ml.transformer.config import TRANSFORMER_EVAL_PATH, VERSIONED_MODEL_DIR

logger = logging.getLogger(__name__)


# ── Internal normaliser ───────────────────────────────────────────────────────

def _normalise_baseline_result(result: dict, model_id: str) -> dict:
    """
    Normalise a baseline evaluator result to the unified schema.
    Renames "macro_f1" → "f1_score", "macro_precision" → "precision", etc.
    """
    return {
        "model_id":          model_id,
        "model_type":        "baseline",
        "test_samples":      result.get("test_samples", 0),
        "accuracy":          result.get("accuracy", 0.0),
        "precision":         result.get("macro_precision", 0.0),
        "recall":            result.get("macro_recall", 0.0),
        "f1_score":          result.get("macro_f1", 0.0),
        "weighted_f1":       result.get("weighted_f1", 0.0),
        "roc_auc":           result.get("roc_auc"),
        "per_class":         result.get("per_class", {}),
        "confusion_matrix":  result.get("confusion_matrix", [[0,0],[0,0]]),
        "inference_time_ms": result.get("inference_time_ms", 0.0),
        "per_sample_ms":     result.get("per_sample_ms", 0.0),
    }


def _normalise_transformer_result(result: dict) -> dict:
    """Normalise a transformer evaluator result to the unified schema."""
    return {
        "model_id":          "distilbert",
        "model_type":        "transformer",
        "test_samples":      result.get("test_samples", 0),
        "accuracy":          result.get("accuracy", 0.0),
        "precision":         result.get("macro_precision", 0.0),
        "recall":            result.get("macro_recall", 0.0),
        "f1_score":          result.get("macro_f1", 0.0),
        "weighted_f1":       result.get("weighted_f1", 0.0),
        "roc_auc":           result.get("roc_auc"),
        "per_class":         result.get("per_class", {}),
        "confusion_matrix":  result.get("confusion_matrix", [[0,0],[0,0]]),
        "inference_time_ms": result.get("inference_time_ms", 0.0),
        "per_sample_ms":     result.get("per_sample_ms", 0.0),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_baseline(
    model_id: str,
    test_df:  pd.DataFrame,
    save:     bool = True,
) -> dict:
    """
    Evaluate a single trained baseline model on the held-out test set.

    Parameters
    ----------
    model_id : "logistic_regression" | "linear_svm" | "naive_bayes"
    test_df  : DataFrame with 'cleaned_text' (or 'text') and 'label'.
    save     : Write JSON result to ml/saved_models/eval/.

    Returns
    -------
    Unified evaluation dict.
    """
    from ml.baseline.trainer   import load_pipeline
    from ml.baseline.evaluator import evaluate_pipeline

    if model_id not in MODEL_IDS:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Valid IDs: {MODEL_IDS}"
        )

    pipeline = load_pipeline(model_id)
    raw      = evaluate_pipeline(model_id, pipeline, test_df, save=save)
    return _normalise_baseline_result(raw, model_id)


def evaluate_all_baseline(
    test_df: pd.DataFrame,
    save:    bool = True,
) -> dict[str, dict]:
    """
    Evaluate all three baseline models on the held-out test set.

    Returns
    -------
    {model_id: unified_eval_dict}
    """
    from ml.baseline.trainer   import load_pipeline
    from ml.baseline.evaluator import evaluate_all

    pipelines = {mid: load_pipeline(mid) for mid in MODEL_IDS}
    raw_all   = evaluate_all(pipelines, test_df, save=save)
    return {
        mid: _normalise_baseline_result(result, mid)
        for mid, result in raw_all.items()
    }


def evaluate_transformer(
    model_dir: Optional[Union[str, Path]] = None,
    test_csv:  Optional[Union[str, Path]] = None,
    save:      bool = True,
) -> dict:
    """
    Evaluate the trained DistilBERT model on the held-out test set.

    Parameters
    ----------
    model_dir : Path to versioned model directory.  Uses VERSIONED_MODEL_DIR if None.
    test_csv  : Path to test.csv.  Uses ml/datasets/processed/test.csv if None.
    save      : Write JSON result to ml/saved_models/eval/distilbert_eval.json.

    Returns
    -------
    Unified evaluation dict.
    """
    from ml.transformer.evaluator import evaluate_from_csv

    dir_path = Path(model_dir) if model_dir else VERSIONED_MODEL_DIR
    csv_path = Path(test_csv)  if test_csv  else Path("ml/datasets/processed/test.csv")

    raw = evaluate_from_csv(model_dir=dir_path, test_csv=csv_path, save=save)
    return _normalise_transformer_result(raw)


def load_saved_results() -> dict[str, dict]:
    """
    Load all previously saved evaluation results from disk.

    Returns
    -------
    Dict mapping model_id → unified evaluation dict.
    Returns {} if no saved results exist.
    """
    results: dict[str, dict] = {}

    # Baseline — combined JSON
    if ALL_EVAL_PATH.exists():
        with open(ALL_EVAL_PATH) as f:
            baseline_raw = json.load(f)
        for mid, result in baseline_raw.items():
            results[mid] = _normalise_baseline_result(result, mid)
    else:
        # Try per-model files
        from ml.baseline.config import MODEL_EVAL_PATHS
        for mid, path in MODEL_EVAL_PATHS.items():
            if path.exists():
                with open(path) as f:
                    results[mid] = _normalise_baseline_result(json.load(f), mid)

    # Transformer
    if TRANSFORMER_EVAL_PATH.exists():
        with open(TRANSFORMER_EVAL_PATH) as f:
            tf_raw = json.load(f)
        results["distilbert"] = _normalise_transformer_result(tf_raw)

    return results


def format_summary(results: dict[str, dict]) -> str:
    """
    Return a human-readable comparison table as a string.

    Parameters
    ----------
    results : {model_id: unified_eval_dict} from load_saved_results() or evaluate_*().
    """
    if not results:
        return "No evaluation results available."

    lines = [
        "",
        "=" * 72,
        "  MODEL EVALUATION SUMMARY",
        "=" * 72,
        f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'ms/s':>7}",
        "-" * 72,
    ]

    for mid, r in results.items():
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "  N/A "
        lines.append(
            f"  {mid:<25}"
            f"{r['accuracy']:>7.4f}"
            f"{r['precision']:>7.4f}"
            f"{r['recall']:>7.4f}"
            f"{r['f1_score']:>7.4f}"
            f"{auc:>8}"
            f"{r['per_sample_ms']:>7.4f}"
        )

    lines.append("=" * 72)

    # Per-class detail
    for mid, r in results.items():
        lines.append(f"\n  {mid}")
        lines.append(f"  {'Class':<6}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Support':>9}")
        lines.append("  " + "-" * 44)
        for cls_name, cm in r.get("per_class", {}).items():
            lines.append(
                f"  {cls_name:<6}  "
                f"{cm['precision']:>8.4f}  "
                f"{cm['recall']:>8.4f}  "
                f"{cm['f1']:>8.4f}  "
                f"{cm['support']:>9}"
            )

    lines.append("")
    return "\n".join(lines)
