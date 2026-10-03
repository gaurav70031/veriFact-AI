"""
train_transformer.py
====================
Fine-tune DistilBERT on the processed fake-news dataset.

Usage examples
--------------
# Basic training (uses config.py defaults)
python scripts/train_transformer.py

# Custom hyperparameters
python scripts/train_transformer.py --epochs 5 --lr 3e-5 --batch-size 8

# Use a different base model
python scripts/train_transformer.py --base-model bert-base-uncased

# Pin a model version string
python scripts/train_transformer.py --version 1.1.0

# Overwrite an existing checkpoint
python scripts/train_transformer.py --force

# Point at custom data directory
python scripts/train_transformer.py --data-dir ml/datasets/processed

Run from the project root (fake-news-detection/).

Prerequisites
-------------
  python scripts/prepare_dataset.py --files ml/datasets/raw/Fake.csv \\
                                             ml/datasets/raw/True.csv

Hardware note
-------------
  CPU training is supported but slow (~1–2 hrs per epoch on ISOT).
  Use a GPU (or Google Colab / Kaggle) for realistic training times.
  Set CUDA_VISIBLE_DEVICES=0 to restrict to a single GPU.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.transformer.config import (
    BASE_MODEL, MODEL_VERSION,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    VERSIONED_MODEL_DIR, TRANSFORMER_MODEL_DIR,
)
from ml.transformer.trainer import train


# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_transformer",
        description=(
            "Fine-tune DistilBERT for fake news detection.\n\n"
            "The model is saved to:\n"
            "  ml/saved_models/distilbert/v{VERSION}/\n\n"
            "Training record (hyperparams + per-epoch metrics) is written to:\n"
            "  ml/saved_models/distilbert/v{VERSION}/training_record.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data-dir", type=Path,
        default=Path("ml/datasets/processed"),
        metavar="PATH",
        help="Directory containing train.csv and val.csv (default: ml/datasets/processed).",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--base-model", default=BASE_MODEL, metavar="HF_ID",
        help=f"HuggingFace model identifier (default: {BASE_MODEL}).",
    )
    model.add_argument(
        "--version", default=MODEL_VERSION, metavar="STR",
        help=f"Model version string, used as directory name (default: {MODEL_VERSION}).",
    )

    # ── Hyperparameters ───────────────────────────────────────────────────────
    hp = parser.add_argument_group("Hyperparameters")
    hp.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, metavar="N",
        help=f"Number of training epochs (default: {NUM_EPOCHS}).",
    )
    hp.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, metavar="N",
        help=f"Training batch size (default: {BATCH_SIZE}).",
    )
    hp.add_argument(
        "--lr", type=float, default=LEARNING_RATE, metavar="FLOAT",
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )

    # ── Flags ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing checkpoint for this version.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    train_path = args.data_dir / "train.csv"
    val_path   = args.data_dir / "val.csv"

    for p in (train_path, val_path):
        if not p.exists():
            print(
                f"[ERROR] Processed split not found: {p}\n"
                "Run first:\n"
                "  python scripts/prepare_dataset.py "
                "--files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.lr <= 0:
        print(f"[ERROR] --lr must be > 0, got {args.lr}", file=sys.stderr)
        sys.exit(1)
    if args.epochs < 1:
        print(f"[ERROR] --epochs must be >= 1, got {args.epochs}", file=sys.stderr)
        sys.exit(1)

    output_dir = TRANSFORMER_MODEL_DIR / f"v{args.version}"
    return train_path, val_path, output_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("  DistilBERT Fine-Tuning")
    print("=" * 60)
    print(f"  Base model   : {args.base_model}")
    print(f"  Version      : {args.version}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    train_path, val_path, output_dir = _validate(args)
    print(f"  Output dir   : {output_dir}")

    try:
        record = train(
            train_path = train_path,
            val_path   = val_path,
            output_dir = output_dir,
            base_model = args.base_model,
            num_epochs = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            force      = args.force,
        )
    except FileExistsError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Training complete.")
    print(f"  Best val_loss : {record['best_val_loss']:.4f}")
    print(f"  Best val_acc  : {record['best_val_acc']:.4f}")
    print(f"  Total time    : {record['total_time_s']:.1f}s")
    print(f"  Checkpoint    : {output_dir}")
    print(f"\nNext step: python scripts/evaluate_transformer.py --version {args.version}")


if __name__ == "__main__":
    main()
