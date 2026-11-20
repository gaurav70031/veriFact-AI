"""
register_models.py
==================
Register trained models and their full evaluation metrics into the PostgreSQL
model_versions table so the Model Laboratory page shows real data.

Reads from:
  ml/saved_models/eval/all_models_eval.json  (baseline metrics)
  ml/saved_models/eval/distilbert_eval.json  (transformer metrics)
  ml/saved_models/*_training_record.json     (training time, hyperparams)
  ml/saved_models/distilbert/v*/training_record.json (loss curves)
  ml/saved_models/*.pkl                      (file size)

Writes to: model_versions table in PostgreSQL

Columns populated:
  accuracy, precision, recall, f1_score, roc_auc (core metrics)
  train_time_s, inference_time_ms (timing)
  model_size_bytes (artifact file size)
  confusion_matrix_json, per_class_metrics_json (extended eval)
  training_history_json (loss curves — transformer only)
  hyperparameters_json, random_seed, dataset_name, training_samples

Run after:
  python scripts/prepare_dataset.py --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv
  python scripts/train_baseline.py
  python scripts/evaluate_baseline.py
  [optional] python scripts/train_transformer.py
  [optional] python scripts/evaluate_transformer.py

Usage:
  python scripts/register_models.py                # register all with eval JSONs
  python scripts/register_models.py --overwrite    # overwrite existing metrics
  python scripts/register_models.py --dry-run      # print without DB changes
  python scripts/register_models.py --transformer-version 1.1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))   # must come AFTER backend so ml.* resolves to top-level ml/

from ml.baseline.config     import MODEL_IDS, MODEL_PIPELINE_PATHS, EVAL_RESULTS_DIR, ALL_EVAL_PATH
from ml.transformer.config  import TRANSFORMER_EVAL_PATH, TRANSFORMER_MODEL_DIR, MODEL_VERSION as DEFAULT_TF_VERSION


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Load eval results ─────────────────────────────────────────────────────────

def _load_baseline_evals() -> dict[str, dict]:
    if ALL_EVAL_PATH.exists():
        with open(ALL_EVAL_PATH) as f:
            return json.load(f)
    # Fall back to per-model files
    results = {}
    for mid in MODEL_IDS:
        for name in [mid, mid.replace("logistic_regression","lr").replace("linear_svm","svm").replace("naive_bayes","nb")]:
            path = EVAL_RESULTS_DIR / f"{name}_eval.json"
            if path.exists():
                with open(path) as f:
                    results[mid] = json.load(f)
                break
    return results


def _load_transformer_eval(version: str) -> dict | None:
    if TRANSFORMER_EVAL_PATH.exists():
        with open(TRANSFORMER_EVAL_PATH) as f:
            return json.load(f)
    return None


def _load_training_record(mid: str) -> dict:
    path = PROJECT_ROOT / "ml" / "saved_models" / f"{mid}_training_record.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_transformer_training_record(version: str) -> dict:
    path = TRANSFORMER_MODEL_DIR / f"v{version}" / "training_record.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _get_model_size(mid: str) -> int | None:
    path = MODEL_PIPELINE_PATHS.get(mid)
    if path and path.exists():
        return path.stat().st_size
    return None


def _get_transformer_size(version: str) -> int | None:
    model_dir = TRANSFORMER_MODEL_DIR / f"v{version}"
    total = 0
    for ext in ["*.bin", "*.safetensors"]:
        for f in model_dir.glob(ext):
            total += f.stat().st_size
    return total if total > 0 else None


# ── Database registration ─────────────────────────────────────────────────────

async def _register(
    baseline_evals:      dict[str, dict],
    transformer_eval:    dict | None,
    transformer_version: str,
    overwrite:           bool,
    dry_run:             bool,
) -> None:
    from sqlalchemy import select
    from app.db.session import AsyncSessionLocal
    from app.models.model_version import ModelVersion, ModelType, ModelAlgorithm

    _ALGO_MAP = {
        "logistic_regression": ModelAlgorithm.LOGISTIC_REGRESSION,
        "linear_svm":          ModelAlgorithm.LINEAR_SVM,
        "naive_bayes":         ModelAlgorithm.NAIVE_BAYES,
        "distilbert":          ModelAlgorithm.DISTILBERT,
    }
    _TYPE_MAP = {
        "logistic_regression": ModelType.TRADITIONAL,
        "linear_svm":          ModelType.TRADITIONAL,
        "naive_bayes":         ModelType.TRADITIONAL,
        "distilbert":          ModelType.TRANSFORMER,
    }
    _DESCRIPTIONS = {
        "logistic_regression": (
            "TF-IDF (50k features, 1-2 ngrams, sublinear TF) + "
            "Logistic Regression (C=1.0, balanced class weights)"
        ),
        "linear_svm": (
            "TF-IDF (50k features, 1-2 ngrams, sublinear TF) + "
            "LinearSVC (C=1.0) wrapped in CalibratedClassifierCV (Platt scaling, cv=5)"
        ),
        "naive_bayes": (
            "TF-IDF (50k features, 1-2 ngrams, sublinear TF) + "
            "MultinomialNB (alpha=0.1)"
        ),
        "distilbert": (
            "DistilBERT-base-uncased fine-tuned on ISOT. "
            "Long-article chunking: 510-token windows with 50-token overlap, "
            "weighted-mean aggregation."
        ),
    }

    async with AsyncSessionLocal() as session:
        async with session.begin():

            # ── Baseline models ───────────────────────────────────────────────
            for mid, raw_eval in baseline_evals.items():
                train_rec  = _load_training_record(mid)
                model_size = _get_model_size(mid)

                # Core metrics
                accuracy   = raw_eval.get("accuracy")
                precision  = raw_eval.get("macro_precision")
                recall     = raw_eval.get("macro_recall")
                f1_score   = raw_eval.get("macro_f1")
                roc_auc    = raw_eval.get("roc_auc")

                # Extended
                confusion  = raw_eval.get("confusion_matrix")  # [[TN,FP],[FN,TP]]
                per_class  = raw_eval.get("per_class", {})

                # Timing
                train_time_s      = train_rec.get("train_time_s")
                inference_time_ms = raw_eval.get("per_sample_ms")

                # Training meta
                training_samples  = train_rec.get("train_samples") or raw_eval.get("test_samples")
                test_samples_n    = raw_eval.get("test_samples")
                hyper             = train_rec.get("hyperparameters", {})
                tfidf             = train_rec.get("tfidf_params", {})

                print(
                    f"  [{mid}]  "
                    f"acc={round(accuracy, 4) if accuracy is not None else 'N/A'}  "
                    f"f1={round(f1_score, 4) if f1_score is not None else 'N/A'}  "
                    f"auc={round(roc_auc, 4) if roc_auc is not None else 'N/A'}  "
                    f"size={str(round(model_size/1024))+'KB' if model_size else 'N/A'}  "
                    f"train={str(round(train_time_s, 2))+'s' if train_time_s else 'N/A'}"
                )

                if dry_run:
                    continue

                existing = await session.execute(
                    select(ModelVersion).where(
                        ModelVersion.model_name == mid,
                        ModelVersion.version == "1.0.0",
                    )
                )
                mv = existing.scalar_one_or_none()

                if mv and not overwrite:
                    print(f"    -> Already exists (use --overwrite to update)")
                    continue

                fields = dict(
                    dataset_name          = "ISOT Fake News Dataset",
                    training_samples      = train_rec.get("train_samples"),
                    test_samples          = test_samples_n,
                    random_seed           = train_rec.get("random_seed", 42),
                    accuracy              = accuracy,
                    precision             = precision,
                    recall                = recall,
                    f1_score              = f1_score,
                    roc_auc               = roc_auc,
                    train_time_s          = train_time_s,
                    inference_time_ms     = inference_time_ms,
                    model_size_bytes      = model_size,
                    confusion_matrix_json = json.dumps(confusion) if confusion else None,
                    per_class_metrics_json = json.dumps(per_class) if per_class else None,
                    hyperparameters_json  = json.dumps({**hyper, "tfidf": tfidf}) if hyper else None,
                    description           = _DESCRIPTIONS.get(mid),
                )

                if mv:
                    for k, v in fields.items():
                        setattr(mv, k, v)
                    print(f"    -> Updated existing row (id={mv.id})")
                else:
                    mv = ModelVersion(
                        model_name    = mid,
                        version       = "1.0.0",
                        model_type    = _TYPE_MAP[mid],
                        algorithm     = _ALGO_MAP[mid],
                        artifact_path = str(MODEL_PIPELINE_PATHS.get(mid, "")),
                        is_active     = True,
                        **fields,
                    )
                    session.add(mv)
                    print(f"    -> Inserted new row")

            # ── Transformer model ─────────────────────────────────────────────
            if transformer_eval:
                mid = "distilbert"
                train_rec  = _load_transformer_training_record(transformer_version)
                model_size = _get_transformer_size(transformer_version)

                accuracy   = transformer_eval.get("accuracy")
                precision  = transformer_eval.get("macro_precision")
                recall     = transformer_eval.get("macro_recall")
                f1_score   = transformer_eval.get("macro_f1")
                roc_auc    = transformer_eval.get("roc_auc")
                confusion  = transformer_eval.get("confusion_matrix")
                per_class  = transformer_eval.get("per_class", {})
                train_time_s      = train_rec.get("total_time_s")
                inference_time_ms = transformer_eval.get("per_sample_ms")
                test_samples_n    = transformer_eval.get("test_samples")
                training_samples  = train_rec.get("train_samples")

                # Training history (loss curves)
                history = train_rec.get("history", [])

                print(
                    f"  [{mid}]  "
                    f"acc={round(accuracy, 4) if accuracy is not None else 'N/A'}  "
                    f"f1={round(f1_score, 4) if f1_score is not None else 'N/A'}  "
                    f"auc={round(roc_auc, 4) if roc_auc is not None else 'N/A'}  "
                    f"size={str(round(model_size/1024/1024, 1))+'MB' if model_size else 'N/A'}"
                )

                if dry_run:
                    return

                hyper = train_rec.get("hyperparameters", {})
                fields = dict(
                    dataset_name            = "ISOT Fake News Dataset",
                    training_samples        = training_samples,
                    test_samples            = test_samples_n,
                    random_seed             = 42,
                    accuracy                = accuracy,
                    precision               = precision,
                    recall                  = recall,
                    f1_score                = f1_score,
                    roc_auc                 = roc_auc,
                    train_time_s            = train_time_s,
                    inference_time_ms       = inference_time_ms,
                    model_size_bytes        = model_size,
                    confusion_matrix_json   = json.dumps(confusion) if confusion else None,
                    per_class_metrics_json  = json.dumps(per_class) if per_class else None,
                    training_history_json   = json.dumps(history) if history else None,
                    hyperparameters_json    = json.dumps(hyper) if hyper else None,
                    description             = _DESCRIPTIONS.get(mid),
                )

                existing = await session.execute(
                    select(ModelVersion).where(
                        ModelVersion.model_name == mid,
                        ModelVersion.version == transformer_version,
                    )
                )
                mv = existing.scalar_one_or_none()

                if mv and not overwrite:
                    print(f"    -> Already exists (use --overwrite to update)")
                else:
                    if mv:
                        for k, v in fields.items():
                            setattr(mv, k, v)
                        print(f"    -> Updated existing row (id={mv.id})")
                    else:
                        mv = ModelVersion(
                            model_name    = mid,
                            version       = transformer_version,
                            model_type    = ModelType.TRANSFORMER,
                            algorithm     = ModelAlgorithm.DISTILBERT,
                            artifact_path = str(TRANSFORMER_MODEL_DIR / f"v{transformer_version}"),
                            is_active     = True,
                            **fields,
                        )
                        session.add(mv)
                        print(f"    -> Inserted new row")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="register_models",
        description="Register trained models and full eval metrics into model_versions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite existing metric values.")
    parser.add_argument("--dry-run", action="store_true",
        help="Print what would be registered without touching the DB.")
    parser.add_argument("--transformer-version", default=DEFAULT_TF_VERSION, metavar="STR",
        help=f"Transformer model version to register (default: {DEFAULT_TF_VERSION}).")
    parser.add_argument("--skip-transformer", action="store_true",
        help="Skip transformer model registration.")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.")
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("  Model Registry Updater")
    print("=" * 60)

    baseline_evals = _load_baseline_evals()
    if not baseline_evals:
        print("[WARN] No baseline evaluation JSON files found.\nRun: python scripts/evaluate_baseline.py\n")
    else:
        print(f"[INFO] Found baseline evals for: {list(baseline_evals.keys())}")

    transformer_eval = None
    if not args.skip_transformer:
        transformer_eval = _load_transformer_eval(args.transformer_version)
        if transformer_eval:
            print(f"[INFO] Found transformer eval (v{args.transformer_version})")
        else:
            print(f"[INFO] No transformer eval found — skipping")

    if not baseline_evals and not transformer_eval:
        print("[ERROR] Nothing to register. Run evaluation scripts first.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY-RUN] Models that would be registered:")
    else:
        print("\nRegistering models...")

    asyncio.run(_register(
        baseline_evals       = baseline_evals,
        transformer_eval     = transformer_eval,
        transformer_version  = args.transformer_version,
        overwrite            = args.overwrite,
        dry_run              = args.dry_run,
    ))

    if args.dry_run:
        print("\n[DRY-RUN complete — no DB changes made]")
    else:
        print("\nDone. Restart the backend for the registry to pick up new artifacts.")


if __name__ == "__main__":
    main()
