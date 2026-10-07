"""
DistilBERT inference module.

This is the single interface used by the FastAPI backend.

Long-article chunking
---------------------
DistilBERT accepts at most 512 tokens.  News articles can be 1000–5000
tokens long.  Truncating throws away most of an article's body.

Strategy: overlapping chunk aggregation
  1. Tokenize the full article (no special tokens, no padding).
  2. Slide a window of CHUNK_SIZE tokens with CHUNK_OVERLAP token stride.
     Example (CHUNK_SIZE=510, OVERLAP=50):
       chunk 0: tokens  0–509
       chunk 1: tokens 460–969
       chunk 2: tokens 920–1429
       ...
  3. Add [CLS]/[SEP], pad to 512, run the model → logits per chunk.
  4. Convert logits to probabilities via softmax.
  5. Aggregate (three strategies selectable via CHUNK_AGGREGATION):

     weighted_mean (default)
       weight_i = softmax(max_prob_i across classes)
       final_prob = sum(weight_i * prob_i) / sum(weight_i)
       Rationale: chunks the model is confident about get more vote weight.

     max_confidence
       Use only the chunk with the highest max probability.
       Rationale: a single highly-confident chunk is the strongest signal.

     majority_vote
       Each chunk casts one vote; ties broken by mean probabilities.
       Rationale: democratic — all chunks are treated equally.

  6. final_label = argmax(final_prob)

Public API (called by FastAPI backend)
---------------------------------------
  predict_text(text, model_dir) → PredictionResult dict
  predict_texts_batch(texts, model_dir) → list[PredictionResult]
  load_model_and_tokenizer(model_dir) → (model, tokenizer)   # for cache
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.transformer.config import (
    VERSIONED_MODEL_DIR,
    CHUNK_AGGREGATION,
    EVAL_BATCH_SIZE,
    ID2LABEL,
    DATALOADER_WORKERS,
    MAX_TOKEN_LENGTH,
)
from ml.transformer.dataset import ChunkedArticleDataset

logger = logging.getLogger(__name__)

# ── Module-level model cache ──────────────────────────────────────────────────
# Keyed by str(model_dir) so multiple versions can coexist in memory.
_model_cache:     dict[str, AutoModelForSequenceClassification] = {}
_tokenizer_cache: dict[str, AutoTokenizer]                      = {}


# ── Load / cache helpers ──────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_dir: Path = VERSIONED_MODEL_DIR,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load (and cache) a DistilBERT model + tokenizer from `model_dir`.

    Parameters
    ----------
    model_dir : Path to a versioned saved model directory containing
                config.json, pytorch_model.bin (or model.safetensors),
                and a 'tokenizer/' subdirectory.

    Returns
    -------
    (model, tokenizer)
    """
    key = str(model_dir)
    if key not in _model_cache:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Transformer model not found at: {model_dir}\n"
                "Run: python scripts/train_transformer.py"
            )
        tokenizer_dir = model_dir / "tokenizer"
        if not tokenizer_dir.exists():
            tokenizer_dir = model_dir   # fallback: tokenizer saved in model root

        logger.info("Loading transformer model from %s ...", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        _model_cache[key]     = model
        _tokenizer_cache[key] = tokenizer
        logger.info("Transformer model loaded and cached.")

    return _model_cache[key], _tokenizer_cache[key]


def clear_model_cache() -> None:
    """Evict all cached models (useful for testing / reloading)."""
    _model_cache.clear()
    _tokenizer_cache.clear()


# ── Chunk aggregation strategies ─────────────────────────────────────────────

def _aggregate_weighted_mean(probs: np.ndarray) -> np.ndarray:
    """
    Weight each chunk by softmax of its max probability.

    probs : shape (n_chunks, n_classes)
    returns: shape (n_classes,)
    """
    max_probs = probs.max(axis=1)                        # (n_chunks,)
    weights   = np.exp(max_probs) / np.exp(max_probs).sum()  # softmax
    return (weights[:, None] * probs).sum(axis=0)        # weighted sum → (n_classes,)


def _aggregate_max_confidence(probs: np.ndarray) -> np.ndarray:
    """Return probabilities from the single highest-confidence chunk."""
    best_idx = probs.max(axis=1).argmax()
    return probs[best_idx]


def _aggregate_majority_vote(probs: np.ndarray) -> np.ndarray:
    """
    Each chunk votes for argmax class.
    Ties broken by mean probabilities.
    """
    votes      = probs.argmax(axis=1)                    # (n_chunks,)
    n_classes  = probs.shape[1]
    vote_counts = np.bincount(votes, minlength=n_classes).astype(float)

    # Normalise vote counts; add a tiny fraction of mean prob to break ties
    mean_probs   = probs.mean(axis=0)
    return vote_counts / vote_counts.sum() + mean_probs * 1e-6


_AGGREGATORS = {
    "weighted_mean":   _aggregate_weighted_mean,
    "max_confidence":  _aggregate_max_confidence,
    "majority_vote":   _aggregate_majority_vote,
}


def aggregate_chunk_probs(
    chunk_probs: np.ndarray,
    strategy:    str = CHUNK_AGGREGATION,
) -> np.ndarray:
    """
    Reduce per-chunk probabilities to a single article-level probability vector.

    Parameters
    ----------
    chunk_probs : shape (n_chunks, n_classes) — softmax output per chunk.
    strategy    : One of "weighted_mean", "max_confidence", "majority_vote".

    Returns
    -------
    shape (n_classes,) — article-level class probabilities summing to 1.
    """
    if strategy not in _AGGREGATORS:
        raise ValueError(
            f"Unknown aggregation strategy '{strategy}'. "
            f"Choose from: {list(_AGGREGATORS.keys())}"
        )
    if len(chunk_probs) == 1:
        return chunk_probs[0]

    raw = _AGGREGATORS[strategy](chunk_probs)
    # Re-normalise to ensure probabilities sum to 1.0
    total = raw.sum()
    return raw / total if total > 0 else raw


# ── Single text prediction ────────────────────────────────────────────────────

def predict_text(
    text:       str,
    model_dir:  Path    = VERSIONED_MODEL_DIR,
    strategy:   str     = CHUNK_AGGREGATION,
    batch_size: int     = EVAL_BATCH_SIZE,
    device:     Optional[torch.device] = None,
) -> dict:
    """
    Predict fake/real for a single article text.

    Long articles are automatically split into overlapping chunks.
    Chunk predictions are aggregated per `strategy`.

    Parameters
    ----------
    text      : Raw article text (title + body recommended).
    model_dir : Path to versioned model directory.
    strategy  : Chunk aggregation strategy.
    batch_size: Chunks processed per forward pass.
    device    : torch.device; auto-detected if None.

    Returns
    -------
    {
        label, is_fake, confidence, fake_probability, real_probability,
        num_chunks, aggregation_strategy, inference_time_ms
    }
    """
    if not text or not isinstance(text, str):
        raise ValueError("text must be a non-empty string.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.to(device)
    model.eval()

    # Build chunked dataset for this article
    chunk_dataset = ChunkedArticleDataset(text, tokenizer)
    num_chunks    = chunk_dataset.num_chunks

    loader = DataLoader(
        chunk_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
    )

    all_chunk_probs: list[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_chunk_probs.extend(probs)   # list of (n_classes,) arrays
    inference_ms = (time.perf_counter() - t0) * 1000

    # Stack → (n_chunks, n_classes) and aggregate
    chunk_probs_array = np.stack(all_chunk_probs, axis=0)
    final_probs       = aggregate_chunk_probs(chunk_probs_array, strategy=strategy)

    # Determine label from model's own class ordering
    classes  = list(model.config.id2label.keys())   # [0, 1] by construction
    fake_idx = 0
    real_idx = 1

    fake_prob   = float(final_probs[fake_idx])
    real_prob   = float(final_probs[real_idx])
    is_fake     = fake_prob > real_prob
    label       = "FAKE" if is_fake else "REAL"
    confidence  = max(fake_prob, real_prob)

    return {
        "label":                 label,
        "is_fake":               is_fake,
        "confidence":            round(confidence,  6),
        "fake_probability":      round(fake_prob,   6),
        "real_probability":      round(real_prob,   6),
        "num_chunks":            num_chunks,
        "aggregation_strategy":  strategy,
        "inference_time_ms":     round(inference_ms, 3),
    }


# ── Batch prediction ─────────────────────────────────────────────────────────

def predict_texts_batch(
    texts:      list[str],
    model_dir:  Path = VERSIONED_MODEL_DIR,
    strategy:   str  = CHUNK_AGGREGATION,
) -> list[dict]:
    """
    Predict fake/real for a list of texts.
    Each text is chunked independently.

    Returns a list of prediction dicts in the same order as `texts`.
    """
    if not texts:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [
        predict_text(text, model_dir=model_dir, strategy=strategy, device=device)
        for text in texts
    ]
