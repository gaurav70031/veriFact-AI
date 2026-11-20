"""
train_bert.py
=============
Fine-tune BERT (bert-base-uncased) for fake news detection.

This script is identical to train_transformer.py but defaults to
bert-base-uncased instead of distilbert-base-uncased.

BERT vs DistilBERT
------------------
  BERT-base-uncased   : 110M parameters, 12 transformer layers
  DistilBERT-base     :  66M parameters,  6 transformer layers (40% smaller)
  Quality trade-off   : DistilBERT reaches ~97% of BERT's GLUE score
  Speed trade-off     : DistilBERT is ~60% faster at inference

Both are supported by the same training/inference/evaluation code.
The base model is just a HuggingFace identifier string passed to
AutoModelForSequenceClassification.from_pretrained().

Usage examples
--------------
# Train BERT (default settings)
python scripts/train_bert.py

# Custom hyperparameters
python scripts/train_bert.py --epochs 4 --lr 2e-5 --batch-size 8

# Use a different HuggingFace model (e.g. RoBERTa)
python scripts/train_bert.py --base-model roberta-base --version roberta-1.0.0

# Overwrite existing checkpoint
python scripts/train_bert.py --force

Hardware note
-------------
BERT is larger than DistilBERT.  On CPU: ~2–4 hrs/epoch on ISOT.
On a T4 GPU (Colab): ~15–30 min/epoch with batch_size=16.
Consider reducing --batch-size to 4 or 8 if you get OOM errors.

Prerequisites
-------------
  python scripts/prepare_dataset.py --files ml/datasets/raw/Fake.csv \\
                                             ml/datasets/raw/True.csv
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.transformer.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    TRANSFORMER_MODEL_DIR,
)
from ml.transformer.trainer import train

BERT_BASE_MODEL = "bert-base-uncased"
BERT_DEFAULT_VERSION = "bert-1.0.0"


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
        prog="train_bert",
        description=(
            "Fine-tune BERT (bert-base-uncased) for fake news detection.\n\n"
            "Uses the same training/evaluation code as train_transformer.py.\n"
            "The model is saved to:\n"
            "  ml/saved_models/distilbert/v{VERSION}/\n\n"
            "To use a different architecture, pass --base-model roberta-base, etc."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data-dir", type=Path,
        default=Path("ml/datasets/processed"),
        help="Directory containing train.csv and val.csv.",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--base-model", default=BERT_BASE_MODEL,
        help=f"HuggingFace model identifier (default: {BERT_BASE_MODEL}).",
    )
    model.add_argument(
        "--version", default=BERT_DEFAULT_VERSION,
        help=f"Model version string (default: {BERT_DEFAULT_VERSION}).",
    )

    hp = parser.add_argument_group("Hyperparameters")
    hp.add_argument("--epochs",     type=int,   default=NUM_EPOCHS,    help=f"Training epochs (default: {NUM_EPOCHS}).")
    hp.add_argument("--batch-size", type=int,   default=BATCH_SIZE,    help=f"Batch size (default: {BATCH_SIZE}). Reduce to 4–8 for BERT on limited VRAM.")
    hp.add_argument("--lr",         type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE}).")

    parser.add_argument("--force",   action="store_true", help="Overwrite existing checkpoint.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

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

    output_dir = TRANSFORMER_MODEL_DIR / f"v{args.version}"

    print("\n" + "=" * 60)
    print("  BERT Fine-Tuning")
    print("=" * 60)
    print(f"  Base model   : {args.base_model}")
    print(f"  Version      : {args.version}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output dir   : {output_dir}")
    print()
    print("  Note: BERT is 40% larger than DistilBERT.")
    print("  Recommended: GPU with >= 8 GB VRAM, batch_size=8–16.")
    print()

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
    print(
        f"\nNext step: python scripts/evaluate_transformer.py "
        f"--version {args.version}"
    )


if __name__ == "__main__":
    main()
