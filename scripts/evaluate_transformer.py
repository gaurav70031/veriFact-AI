"""
evaluate_transformer.py
=======================
Evaluate a saved DistilBERT checkpoint on the held-out test set.

Usage examples
--------------
# Evaluate the default version
python scripts/evaluate_transformer.py

# Evaluate a specific version
python scripts/evaluate_transformer.py --version 1.1.0

# Use a custom test file
python scripts/evaluate_transformer.py --test-file ml/datasets/processed/test.csv

# Print previously saved results without re-running inference
python scripts/evaluate_transformer.py --show-saved

# Save results to a custom path
python scripts/evaluate_transformer.py --output-json results/distilbert_eval.json

Run from the project root (fake-news-detection/).

Prerequisites
-------------
  python scripts/train_transformer.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.transformer.config import (
    MODEL_VERSION,
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_EVAL_PATH,
)
from ml.transformer.evaluator import evaluate_from_csv, print_evaluation_summary


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("filelock", "urllib3", "transformers.modeling_utils",
                  "transformers.configuration_utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_transformer",
        description=(
            "Evaluate a saved DistilBERT model on the held-out test set.\n\n"
            "Computes:\n"
            "  accuracy, precision, recall, F1, macro F1,\n"
            "  ROC-AUC, confusion matrix, inference time per sample\n\n"
            "Results saved to: ml/saved_models/eval/distilbert_eval.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", default=MODEL_VERSION, metavar="STR",
        help=f"Model version to evaluate (default: {MODEL_VERSION}).",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None, metavar="PATH",
        help="Explicit path to model directory (overrides --version).",
    )
    parser.add_argument(
        "--test-file", type=Path,
        default=Path("ml/datasets/processed/test.csv"),
        metavar="PATH",
        help="Test CSV file (default: ml/datasets/processed/test.csv).",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, metavar="PATH",
        help="Override output JSON path (default: ml/saved_models/eval/distilbert_eval.json).",
    )
    parser.add_argument(
        "--show-saved", action="store_true",
        help="Print previously saved evaluation results without running inference.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run evaluation but do not write JSON results to disk.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _show_saved(path: Path) -> None:
    if not path.exists():
        print(
            f"[ERROR] No saved evaluation results at {path}\n"
            "Run: python scripts/evaluate_transformer.py",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        result = json.load(f)
    print_evaluation_summary(result)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("  DistilBERT Evaluation")
    print("=" * 60)

    if args.show_saved:
        _show_saved(TRANSFORMER_EVAL_PATH)
        return

    # Resolve model directory
    model_dir = args.model_dir or (TRANSFORMER_MODEL_DIR / f"v{args.version}")

    if not model_dir.exists():
        print(
            f"[ERROR] Model not found at: {model_dir}\n"
            f"Run first: python scripts/train_transformer.py --version {args.version}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.test_file.exists():
        print(
            f"[ERROR] Test file not found: {args.test_file}\n"
            "Run first: python scripts/prepare_dataset.py "
            "--files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Model dir  : {model_dir}")
    print(f"  Test file  : {args.test_file}")

    save = not args.no_save
    result = evaluate_from_csv(
        model_dir=model_dir,
        test_csv=args.test_file,
        save=save,
    )

    # Override output path if requested
    if args.output_json and save:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Results also written to {args.output_json}")

    print_evaluation_summary(result)

    if save:
        print(f"[INFO] JSON results saved to ml/saved_models/eval/distilbert_eval.json")


if __name__ == "__main__":
    main()
