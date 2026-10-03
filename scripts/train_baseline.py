"""
train_baseline.py
=================
Train one or all TF-IDF baseline models on the processed dataset.

Usage examples
--------------
# Train all three models
python scripts/train_baseline.py

# Train a single model
python scripts/train_baseline.py --model logistic_regression
python scripts/train_baseline.py --model linear_svm
python scripts/train_baseline.py --model naive_bayes

# Dry-run: verify data loads and pipelines build, but do not save artifacts
python scripts/train_baseline.py --dry-run

# Use custom processed data directory
python scripts/train_baseline.py --data-dir ml/datasets/processed

Run from the project root (fake-news-detection/).
Requires processed splits from: python scripts/prepare_dataset.py
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.baseline.trainer  import train_single, train_all
from ml.baseline.config   import MODEL_IDS
from ml.preprocessing.config import TRAIN_CSV, VAL_CSV


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_splits(data_dir: Path):
    import pandas as pd

    train_path = data_dir / "train.csv"
    val_path   = data_dir / "val.csv"

    for p in (train_path, val_path):
        if not p.exists():
            print(
                f"[ERROR] Processed split not found: {p}\n"
                "Run first: python scripts/prepare_dataset.py "
                "--files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv",
                file=sys.stderr,
            )
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # Accept either 'cleaned_text' or 'text' column
    text_col = "cleaned_text" if "cleaned_text" in train_df.columns else "text"
    print(f"[INFO] Text column: '{text_col}'")
    print(f"[INFO] Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    return train_df, val_df


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_baseline",
        description=(
            "Train TF-IDF baseline models for fake news detection.\n\n"
            "Available models:\n"
            "  logistic_regression — TF-IDF + Logistic Regression\n"
            "  linear_svm          — TF-IDF + LinearSVC (calibrated)\n"
            "  naive_bayes         — TF-IDF + MultinomialNB\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m",
        choices=MODEL_IDS + ["all"],
        default="all",
        help="Which model to train (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ml/datasets/processed"),
        metavar="PATH",
        help="Directory containing train.csv and val.csv (default: ml/datasets/processed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build pipelines and load data but do NOT save artifacts.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    print("\n" + "=" * 55)
    print("  Baseline Model Training")
    print("=" * 55)

    train_df, val_df = _load_splits(args.data_dir)

    save = not args.dry_run
    if args.dry_run:
        print("[INFO] DRY-RUN mode — pipelines will NOT be saved.\n")

    if args.model == "all":
        results = train_all(train_df, val_df, save=save)
        print(f"\n[INFO] Trained {len(results)} models.")
        for model_id, (_, record) in results.items():
            print(
                f"  {model_id:<25}  "
                f"train_time={record['train_time_s']:.2f}s  "
                f"samples={record['train_samples']:,}"
            )
    else:
        pipeline, record = train_single(args.model, train_df, val_df, save=save)
        print(
            f"\n[INFO] {args.model} trained in {record['train_time_s']:.2f}s  "
            f"({record['train_samples']:,} samples)"
        )

    if save:
        print("\nNext step: python scripts/evaluate_baseline.py")
    print()


if __name__ == "__main__":
    main()
