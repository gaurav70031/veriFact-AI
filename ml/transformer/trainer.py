"""
DistilBERT fine-tuning trainer.

Training loop
-------------
1. Load train.csv and val.csv from the processed dataset.
2. Tokenize with DistilBERT tokenizer (fitted only on model vocabulary,
   never on data — transformers use a fixed pre-trained vocabulary).
3. Run N epochs:
     - Forward pass on training batch
     - Compute cross-entropy loss
     - Backward pass + gradient clip + AdamW step + warmup LR schedule
     - After each epoch: evaluate on val set
     - Save checkpoint if val_loss improved (best-only by default)
4. Early stopping if val_loss does not improve for PATIENCE epochs.
5. Save best model + tokenizer to versioned directory.
6. Write training history to JSON.

Model versioning
----------------
Each training run saves to:
    ml/saved_models/distilbert/v{VERSION}/
        config.json         — HuggingFace model config
        pytorch_model.bin   — model weights (or model.safetensors)
        tokenizer/          — tokenizer files
        training_record.json — hyperparams, metrics, version info

If VERSION already exists, training is blocked unless --force is passed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import pandas as pd

from ml.transformer.config import (
    BASE_MODEL,
    VERSIONED_MODEL_DIR,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    EARLY_STOPPING_PATIENCE,
    SAVE_BEST_ONLY,
    RANDOM_SEED,
    NUM_LABELS,
    LABEL2ID,
    ID2LABEL,
    MODEL_VERSION,
    DATALOADER_WORKERS,
    USE_FP16,
    MAX_TOKEN_LENGTH,
)
from ml.transformer.dataset import FakeNewsDataset

logger = logging.getLogger(__name__)


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        dev = torch.device("cpu")
        logger.info("Using CPU (training will be slow)")
    return dev


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {path}\n"
            "Run: python scripts/prepare_dataset.py"
        )
    df = pd.read_csv(path)
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    # Ensure text col is named consistently
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    df["text"]  = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


# ── Training utilities ────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    loss_fn:    nn.Module,
) -> tuple[float, float]:
    """Return (avg_loss, accuracy) over the full DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss     = loss_fn(logits, labels)
            preds    = logits.argmax(dim=-1)

            total_loss += loss.item() * len(labels)
            correct    += (preds == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── Main training function ────────────────────────────────────────────────────

def train(
    train_path:   Path,
    val_path:     Path,
    output_dir:   Path               = VERSIONED_MODEL_DIR,
    base_model:   str                = BASE_MODEL,
    num_epochs:   int                = NUM_EPOCHS,
    batch_size:   int                = BATCH_SIZE,
    lr:           float              = LEARNING_RATE,
    force:        bool               = False,
) -> dict:
    """
    Fine-tune DistilBERT and save the best checkpoint.

    Parameters
    ----------
    train_path : Path to train.csv.
    val_path   : Path to val.csv.
    output_dir : Directory to save model + tokenizer.
    base_model : HuggingFace model identifier.
    num_epochs : Training epochs.
    batch_size : Training batch size.
    lr         : Learning rate.
    force      : Overwrite existing checkpoint.

    Returns
    -------
    training_record dict (also written to output_dir/training_record.json).
    """
    # ── Version guard ─────────────────────────────────────────────────────────
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Model directory already exists: {output_dir}\n"
            "Use force=True (or --force) to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)

    _set_seed(RANDOM_SEED)
    device = get_device()
    use_fp16 = USE_FP16 and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler() if use_fp16 else None

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading training data from %s ...", train_path)
    train_df = _load_split(train_path)
    val_df   = _load_split(val_path)
    logger.info("Train: %d  |  Val: %d", len(train_df), len(val_df))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = FakeNewsDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_TOKEN_LENGTH,
    )
    val_dataset = FakeNewsDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_TOKEN_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=DATALOADER_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=DATALOADER_WORKERS,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading base model: %s", base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_count  = 0
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0
        t_epoch = time.time()

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss    = loss_fn(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()

            scheduler.step()
            epoch_loss  += loss.item()
            epoch_steps += 1

            if step % 50 == 0:
                logger.info(
                    "Epoch %d/%d  step %d/%d  loss=%.4f",
                    epoch, num_epochs, step, len(train_loader),
                    epoch_loss / epoch_steps,
                )

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_acc = _eval_epoch(model, val_loader, device, loss_fn)
        train_loss_avg    = epoch_loss / epoch_steps
        epoch_time        = time.time() - t_epoch

        epoch_record = {
            "epoch":      epoch,
            "train_loss": round(train_loss_avg, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            epoch, num_epochs, train_loss_avg, val_loss, val_acc, epoch_time,
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            if SAVE_BEST_ONLY:
                model.save_pretrained(str(output_dir))
                tokenizer.save_pretrained(str(tokenizer_dir))
                logger.info("  ✓ Checkpoint saved (val_loss improved to %.4f)", val_loss)
        else:
            patience_count += 1
            logger.info(
                "  val_loss did not improve (%d/%d patience)",
                patience_count, EARLY_STOPPING_PATIENCE,
            )
            if patience_count >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    total_time = time.time() - t_start

    # ── Save final checkpoint if not SAVE_BEST_ONLY ───────────────────────────
    if not SAVE_BEST_ONLY:
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(tokenizer_dir))

    # ── Training record ───────────────────────────────────────────────────────
    record = {
        "model_version":   MODEL_VERSION,
        "base_model":      base_model,
        "output_dir":      str(output_dir),
        "train_samples":   len(train_df),
        "val_samples":     len(val_df),
        "num_epochs_run":  len(history),
        "best_val_loss":   round(best_val_loss, 6),
        "best_val_acc":    round(max(h["val_acc"] for h in history), 6),
        "total_time_s":    round(total_time, 2),
        "hyperparameters": {
            "batch_size":    batch_size,
            "learning_rate": lr,
            "num_epochs":    num_epochs,
            "warmup_ratio":  WARMUP_RATIO,
            "weight_decay":  WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "max_length":    MAX_TOKEN_LENGTH,
            "fp16":          use_fp16,
        },
        "history": history,
    }
    record_path = output_dir / "training_record.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("Training record → %s", record_path)
    logger.info(
        "Training complete. Best val_loss=%.4f  total_time=%.1fs",
        best_val_loss, total_time,
    )
    return record
