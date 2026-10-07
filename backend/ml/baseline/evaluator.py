"""
Baseline model evaluator.

Computes all metrics from actual test-set predictions.
Nothing is hardcoded — every number comes from sklearn metric functions
applied to real model outputs on real (held-out) test data.

Metrics computed
----------------
  accuracy          — overall fraction correct
  precision         — per-class and macro average
  recall            — per-class and macro average
  f1_score          — per-class and macro average
  macro_f1          — unweighted mean F1 across classes
  weighted_f1       — support-weighted mean F1
  confusion_matrix  — [[TN, FP], [FN, TP]]
  roc_auc           — area under ROC curve (requires predict_proba)
  inference_time_ms — wall-clock ms for predict() on full test set
  per_sample_ms     — average ms per sample

Results are returned as plain Python dicts and saved as JSON.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ml.baseline.config import (
    ID2LABEL,
    MODEL_EVAL_PATHS,
    ALL_EVAL_PATH,
    EVAL_RESULTS_DIR,
)

logger = logging.getLogger(__name__)


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_pipeline(
    model_id:  str,
    pipeline:  Pipeline,
    test_df:   pd.DataFrame,
    save:      bool = True,
) -> dict:
    """
    Evaluate a fitted pipeline on the held-out test set.

    Parameters
    ----------
    model_id : Identifier string (used for saving results).
    pipeline : Fitted sklearn Pipeline.
    test_df  : Test DataFrame with 'cleaned_text' (or 'text') and 'label'.
    save     : Write JSON results to disk.

    Returns
    -------
    dict with all metrics — fully machine-readable.
    """
    text_col = "cleaned_text" if "cleaned_text" in test_df.columns else "text"
    X_test = test_df[text_col].fillna("").tolist()
    y_true = test_df["label"].tolist()

    if len(X_test) == 0:
        raise ValueError("test_df is empty — cannot evaluate.")

    # ── Inference (timed) ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_time_s = time.perf_counter() - t0
    inference_time_ms = inference_time_s * 1000

    # ── Probabilities (for ROC AUC) ──────────────────────────────────────────
    y_prob_fake = None
    y_prob_real = None
    roc_auc = None

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)   # shape (n, 2)
            # Column order matches pipeline.classes_
            classes = list(pipeline.classes_)
            fake_idx = classes.index(0) if 0 in classes else 0
            real_idx = classes.index(1) if 1 in classes else 1
            y_prob_fake = proba[:, fake_idx].tolist()
            y_prob_real = proba[:, real_idx].tolist()
            roc_auc = float(roc_auc_score(y_true, proba[:, real_idx]))
        except Exception as exc:
            logger.warning("Could not compute ROC AUC for %s: %s", model_id, exc)

    # ── Core metrics ─────────────────────────────────────────────────────────
    accuracy   = float(accuracy_score(y_true, y_pred))
    macro_f1   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1= float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    ).tolist()
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    ).tolist()
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    ).tolist()

    macro_precision = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    macro_recall    = float(recall_score   (y_true, y_pred, average="macro",    zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # ── Per-class dict ────────────────────────────────────────────────────────
    per_class: dict[str, dict] = {}
    for label_id, label_name in ID2LABEL.items():
        support = int(sum(1 for y in y_true if y == label_id))
        per_class[label_name] = {
            "precision": round(precision_per_class[label_id], 6),
            "recall":    round(recall_per_class[label_id],    6),
            "f1":        round(f1_per_class[label_id],        6),
            "support":   support,
        }

    # ── Assemble result dict ──────────────────────────────────────────────────
    result = {
        "model_id":            model_id,
        "test_samples":        len(X_test),
        "accuracy":            round(accuracy,    6),
        "macro_f1":            round(macro_f1,    6),
        "weighted_f1":         round(weighted_f1, 6),
        "macro_precision":     round(macro_precision, 6),
        "macro_recall":        round(macro_recall,    6),
        "roc_auc":             round(roc_auc, 6) if roc_auc is not None else None,
        "per_class":           per_class,
        "confusion_matrix":    cm,           # [[TN, FP], [FN, TP]]
        "inference_time_ms":   round(inference_time_ms, 3),
        "per_sample_ms":       round(inference_time_ms / len(X_test), 4),
    }

    logger.info(
        "%s  |  acc=%.4f  macro_f1=%.4f  roc_auc=%s  t=%.1f ms",
        model_id,
        accuracy,
        macro_f1,
        f"{roc_auc:.4f}" if roc_auc else "N/A",
        inference_time_ms,
    )

    if save:
        _save_result(model_id, result)

    return result


def _save_result(model_id: str, result: dict) -> None:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_EVAL_PATHS[model_id]
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Evaluation results saved → %s", path)


# ── Evaluate all models ───────────────────────────────────────────────────────

def evaluate_all(
    pipelines: dict[str, Pipeline],
    test_df:   pd.DataFrame,
    save:      bool = True,
) -> dict[str, dict]:
    """
    Evaluate every model in `pipelines` dict on the same test set.

    Parameters
    ----------
    pipelines : {model_id: fitted_pipeline}
    test_df   : Held-out test DataFrame.
    save      : Save individual + combined results to disk.

    Returns
    -------
    {model_id: metrics_dict}
    """
    all_results: dict[str, dict] = {}
    for model_id, pipeline in pipelines.items():
        result = evaluate_pipeline(model_id, pipeline, test_df, save=save)
        all_results[model_id] = result

    if save:
        with open(ALL_EVAL_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Combined evaluation results → %s", ALL_EVAL_PATH)

    return all_results


# ── Human-readable summary ────────────────────────────────────────────────────

def print_evaluation_summary(results: dict[str, dict]) -> None:
    """Print a compact comparison table to stdout."""
    header = f"\n{'Model':<25} {'Acc':>7} {'MacroF1':>9} {'ROC-AUC':>9} {'ms/sample':>10}"
    print("\n" + "=" * 65)
    print("  BASELINE MODEL EVALUATION SUMMARY")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for model_id, r in results.items():
        roc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A "
        print(
            f"  {model_id:<23} "
            f"{r['accuracy']:>7.4f} "
            f"{r['macro_f1']:>9.4f} "
            f"{roc:>9} "
            f"{r['per_sample_ms']:>9.4f}"
        )
    print("=" * 65)

    # Per-class details
    for model_id, r in results.items():
        print(f"\n  {model_id}")
        print(f"  {'Class':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>9}")
        print("  " + "-" * 42)
        for cls_name, cls_metrics in r["per_class"].items():
            print(
                f"  {cls_name:<8} "
                f"{cls_metrics['precision']:>8.4f} "
                f"{cls_metrics['recall']:>8.4f} "
                f"{cls_metrics['f1']:>8.4f} "
                f"{cls_metrics['support']:>9}"
            )
        cm = r["confusion_matrix"]
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"             FAKE   REAL")
        print(f"  FAKE   {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"  REAL   {cm[1][0]:>6}  {cm[1][1]:>6}")
    print()
