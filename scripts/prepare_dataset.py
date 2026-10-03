"""
prepare_dataset.py
==================
CLI entry point for the preprocessing pipeline.

Usage examples
--------------
# ISOT dataset (two files)
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv

# Single generic CSV
python scripts/prepare_dataset.py --files ml/datasets/raw/news.csv

# JSONL dataset with transformer-mode cleaning
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/articles.jsonl \
    --mode transformer

# Rebuild even if processed files already exist
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv \
    --force

# Custom split ratios
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv \
    --test-size 0.15 --val-size 0.10

# Inspect a raw file without running the pipeline
python scripts/prepare_dataset.py --inspect ml/datasets/raw/Fake.csv

Run from the project root (fake-news-detection/).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the project root without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.pipeline import run_pipeline
from ml.preprocessing.loader   import describe_file
from ml.preprocessing.config   import (
    RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    TRAIN_CSV, VAL_CSV, TEST_CSV, STATS_JSON,
)


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description=(
            "Preprocess raw fake-news datasets into cleaned train/val/test splits.\n\n"
            "Supported datasets:\n"
            "  ISOT   — Fake.csv + True.csv (kaggle: clmentbisaillon/fake-and-real-news-dataset)\n"
            "  LIAR   — train/valid/test .tsv (huggingface: liar)\n"
            "  Generic — any CSV/JSON with detectable text + label columns\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ────────────────────────────────────────────────────────────────
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--files", "-f",
        nargs="+",
        metavar="PATH",
        help="One or more raw dataset file paths (CSV, TSV, JSON, JSONL).",
    )
    input_group.add_argument(
        "--inspect",
        metavar="PATH",
        help=(
            "Inspect a single raw file: print detected columns and label distribution "
            "without running the full pipeline."
        ),
    )

    # ── Pipeline options ──────────────────────────────────────────────────────
    pipeline_group = parser.add_argument_group("Pipeline options")
    pipeline_group.add_argument(
        "--mode",
        choices=["baseline", "transformer"],
        default="baseline",
        help=(
            "Text cleaning mode.\n"
            "  baseline    — heavy normalisation for TF-IDF models (default)\n"
            "  transformer — light cleaning for DistilBERT (preserves punctuation/casing)"
        ),
    )
    pipeline_group.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        metavar="FLOAT",
        help=f"Fraction of data for the test split (default: {TEST_SIZE}).",
    )
    pipeline_group.add_argument(
        "--val-size",
        type=float,
        default=VAL_SIZE,
        metavar="FLOAT",
        help=f"Fraction of data for the validation split (default: {VAL_SIZE}).",
    )
    pipeline_group.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        metavar="INT",
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    pipeline_group.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed splits even if they already exist.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip printing dataset statistics to stdout.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations before running anything."""
    if args.inspect:
        return   # inspect mode needs no other args

    if not args.files:
        print(
            "[ERROR] --files is required unless --inspect is used.\n"
            "Example: python scripts/prepare_dataset.py "
            "--files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(
                f"[ERROR] File not found: {p}\n"
                "Place dataset files in ml/datasets/raw/ before running.",
                file=sys.stderr,
            )
            sys.exit(1)

    total_held_out = args.test_size + args.val_size
    if total_held_out >= 0.5:
        print(
            f"[ERROR] test_size ({args.test_size}) + val_size ({args.val_size}) = "
            f"{total_held_out:.2f} — this leaves less than 50% for training.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.test_size <= 0 or args.val_size <= 0:
        print("[ERROR] --test-size and --val-size must be > 0.", file=sys.stderr)
        sys.exit(1)


# ── Inspect mode ──────────────────────────────────────────────────────────────

def _run_inspect(path: str) -> None:
    print(f"\nInspecting: {path}\n")
    info = describe_file(path)
    if "error" in info:
        print(f"[ERROR] {info['error']}")
        sys.exit(1)
    print(f"  Format           : {info['format']}")
    print(f"  Columns detected : {info['columns']}")
    print(f"  Label column     : {info['detected_label_col'] or 'NOT FOUND'}")
    print(f"  Sample rows read : {info['sample_rows']}")
    if info["sample_label_distribution"]:
        print("  Sample label distribution:")
        for lbl, cnt in sorted(info["sample_label_distribution"].items()):
            print(f"    {lbl!r:20s}: {cnt}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    _setup_logging(args.verbose)
    _validate_args(args)

    # ── Inspect mode ──────────────────────────────────────────────────────────
    if args.inspect:
        _run_inspect(args.inspect)
        return

    # ── Pipeline mode ─────────────────────────────────────────────────────────
    file_paths = [Path(f) for f in args.files]

    try:
        train_df, val_df, test_df, stats = run_pipeline(
            paths         = file_paths,
            cleaning_mode = args.mode,
            test_size     = args.test_size,
            val_size      = args.val_size,
            random_seed   = args.seed,
            force         = args.force,
            verbose       = not args.no_stats,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("Output files:")
    for path in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  {path.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")

    if STATS_JSON.exists():
        print(f"  {STATS_JSON.relative_to(PROJECT_ROOT)}")

    print("\nDone. Next step: python scripts/train_baseline.py")


if __name__ == "__main__":
    main()
