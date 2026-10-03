"""
evaluate_baseline.py
====================
Evaluate trained baseline models on the held-out test set.
Loads saved pipelines and computes all metrics from actual predictions.

Usage examples
--------------
# Evaluate all trained models
python scripts/evaluate_baseline.py

# Evaluate one model
python scripts/evaluate_baseline.py --model logistic_regression

# Show results for a model already evaluated (no re-inference)
python scripts/evaluate_baseline.py --show-saved

# Evaluate on a different test file
python scripts/evaluate_baseline.py --test-file ml/datasets/processed/test.csv

Run from the project root (fake-news-detection/).
Requires trained models from: python scripts/train_baseline.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.baseline.trainer   import load_pipeline
from ml.baseline.evaluator import evaluate_pipeline, evaluate_all, print_evaluation_summary
from ml.baseline.config    import MODEL_IDS, MODEL_PIPELINE_PATHS, ALL_EVAL_PATH


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_test_split(test_path: Path):
    import pandas as pd
    if not test_path.exists():
        print(
            f"[ERROR] Test split not found: {test_path}\n"
            "Run first: python scripts/prepare_dataset.py "
            "--files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv",
            file=sys.stderr,
        )
        sys.exit(1)
    df = pd.read_csv(test_path)
    print(f"[INFO] Test split: {len(df):,} rows from {test_path}")
    return df


def _check_model_exists(model_id: str) -> bool:
    path = MODEL_PIPELINE_PATHS[model_id]
    if not path.exists():
        print(
            f"[ERROR] No saved pipeline for '{model_id}' at {path}\n"
            f"Run first: python scripts/train_baseline.py --model {model_id}",
            file=sys.stderr,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_baseline",
        description=(
            "Evaluate trained TF-IDF baseline models on the held-out test set.\n\n"
            "Computes:\n"
            "  accuracy, precision, recall, F1, macro F1, ROC-AUC,\n"
            "  confusion matrix, inference time per sample\n\n"
            "Results are saved as JSON to ml/saved_models/eval/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m",
        choices=MODEL_IDS + ["all"],
        default="all",
        help="Which model(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("ml/datasets/processed/test.csv"),
        metavar="PATH",
        help="Path to test CSV (default: ml/datasets/processed/test.csv).",
    )
    parser.add_argument(
        "--show-saved",
        action="store_true",
        help="Print previously saved evaluation results without running inference.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run evaluation but do not overwrite saved JSON results.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _show_saved_results() -> None:
    if not ALL_EVAL_PATH.exists():
        print(
            f"[ERROR] No saved evaluation results at {ALL_EVAL_PATH}\n"
            "Run: python scripts/evaluate_baseline.py",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(ALL_EVAL_PATH) as f:
        results = json.load(f)
    print_evaluation_summary(results)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    print("\n" + "=" * 55)
    print("  Baseline Model Evaluation")
    print("=" * 55)

    if args.show_saved:
        _show_saved_results()
        return

    test_df = _load_test_split(args.test_file)
    save    = not args.no_save

    if args.model == "all":
        # Verify all pipelines exist before starting
        missing = [m for m in MODEL_IDS if not _check_model_exists(m)]
        if missing:
            sys.exit(1)

        pipelines = {m: load_pipeline(m) for m in MODEL_IDS}
        results   = evaluate_all(pipelines, test_df, save=save)
        print_evaluation_summary(results)

    else:
        if not _check_model_exists(args.model):
            sys.exit(1)
        pipeline = load_pipeline(args.model)
        result   = evaluate_pipeline(args.model, pipeline, test_df, save=save)
        print_evaluation_summary({args.model: result})

    if save:
        print(f"[INFO] JSON results saved to ml/saved_models/eval/")


if __name__ == "__main__":
    main()
