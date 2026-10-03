"""
Transformer model evaluator.

Computes all metrics from actual test-set predictions.
Nothing is hardcoded.  Every number is computed by sklearn metric functions
applied to real model outputs on the held-out test set.

Metrics
-------
  accuracy, macro_precision, macro_recall, macro_f1, weighted_f1,
  per-class precision/recall/F1/support, confusion matrix,
  ROC-AUC, inference_time_ms, per_sample_ms

Results written to: ml/saved_models/eval/distilbert_eval.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.transformer.config import (
    ID2LABEL,
    EVAL_BATCH_SIZE,
    TRANSFORMER_EVAL_PATH,
    EVAL_RESULTS_DIR,
    DATALOADER_WORKERS,
)
from ml.transformer.dataset import FakeNewsDataset
from ml.transformer.inference import load_model_and_tokenizer

logger = logging.getLogger(__name__)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model_dir:  Path,
    test_texts: list[str],
    test_labels: list[int],
    save:       bool = True,
) -> dict:
    """
    Evaluate a saved DistilBERT model on a list of texts and labels.

    Parameters
    ----------
    model_dir   : Path to the saved model directory (versioned).
    test_texts  : List of cleaned article texts.
    test_labels : List of integer labels (0=FAKE, 1=REAL).
    save        : Write JSON results to TRANSFORMER_EVAL_PATH.

    Returns
    -------
    dict with all metrics.
    """
    if len(test_texts) == 0:
        raise ValueError("test_texts is empty — cannot evaluate.")
    if len(test_texts) != len(test_labels):
        raise ValueError("test_texts and test_labels must be the same length.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.to(device)
    model.eval()

    dataset = FakeNewsDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
    )

    all_preds:  list[int]   = []
    all_probs:  list[float] = []   # probability of REAL (class 1)
    all_labels: list[int]   = list(test_labels)

    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1)
            preds   = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # P(REAL)

    inference_ms     = (time.perf_counter() - t0) * 1000
    per_sample_ms    = inference_ms / len(test_texts)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = all_labels
    y_pred = all_preds
    y_prob = all_probs

    accuracy      = float(accuracy_score(y_true, y_pred))
    macro_f1      = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec    = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall  = float(recall_score   (y_true, y_pred, average="macro", zero_division=0))

    prec_per  = precision_score(y_true, y_pred, average=None, labels=[0,1], zero_division=0).tolist()
    rec_per   = recall_score   (y_true, y_pred, average=None, labels=[0,1], zero_division=0).tolist()
    f1_per    = f1_score       (y_true, y_pred, average=None, labels=[0,1], zero_division=0).tolist()
    cm        = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = None

    per_class = {}
    for label_id, label_name in ID2LABEL.items():
        per_class[label_name] = {
            "precision": round(prec_per[label_id], 6),
            "recall":    round(rec_per[label_id],  6),
            "f1":        round(f1_per[label_id],   6),
            "support":   int(sum(1 for y in y_true if y == label_id)),
        }

    result = {
        "model_dir":          str(model_dir),
        "test_samples":       len(test_texts),
        "accuracy":           round(accuracy,     6),
        "macro_f1":           round(macro_f1,     6),
        "weighted_f1":        round(weighted_f1,  6),
        "macro_precision":    round(macro_prec,   6),
        "macro_recall":       round(macro_recall, 6),
        "roc_auc":            round(roc_auc, 6) if roc_auc is not None else None,
        "per_class":          per_class,
        "confusion_matrix":   cm,
        "inference_time_ms":  round(inference_ms,  3),
        "per_sample_ms":      round(per_sample_ms, 4),
    }

    logger.info(
        "DistilBERT eval  |  acc=%.4f  macro_f1=%.4f  roc_auc=%s  t=%.1fms",
        accuracy, macro_f1,
        f"{roc_auc:.4f}" if roc_auc else "N/A",
        inference_ms,
    )

    if save:
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRANSFORMER_EVAL_PATH, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Evaluation results saved → %s", TRANSFORMER_EVAL_PATH)

    return result


def evaluate_from_csv(
    model_dir: Path,
    test_csv:  Path,
    save:      bool = True,
) -> dict:
    """Convenience wrapper: load test.csv and call evaluate()."""
    import pandas as pd
    df = pd.read_csv(test_csv)
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    texts  = df[text_col].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return evaluate(model_dir, texts, labels, save=save)


def print_evaluation_summary(result: dict) -> None:
    print("\n" + "=" * 60)
    print("  DISTILBERT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model dir     : {result['model_dir']}")
    print(f"  Test samples  : {result['test_samples']:,}")
    print(f"  Accuracy      : {result['accuracy']:.4f}")
    print(f"  Macro F1      : {result['macro_f1']:.4f}")
    print(f"  Weighted F1   : {result['weighted_f1']:.4f}")
    print(f"  ROC-AUC       : {result['roc_auc']:.4f}" if result['roc_auc'] else "  ROC-AUC       : N/A")
    print(f"  Inference     : {result['inference_time_ms']:.1f} ms total  "
          f"({result['per_sample_ms']:.3f} ms/sample)")
    print()
    print(f"  {'Class':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 42)
    for cls_name, m in result["per_class"].items():
        print(f"  {cls_name:<8} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>9}")
    cm = result["confusion_matrix"]
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"             FAKE   REAL")
    print(f"  FAKE   {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  REAL   {cm[1][0]:>6}  {cm[1][1]:>6}")
    print("=" * 60 + "\n")
