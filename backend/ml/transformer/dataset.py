"""
PyTorch Dataset and tokenization for DistilBERT fine-tuning.

Two modes
---------
FakeNewsDataset (training / evaluation)
    Tokenizes each sample to a single fixed-length encoding.
    Used during training — the trainer controls batching.

ChunkedDataset (inference on long articles)
    Splits one article into overlapping token chunks.
    Each chunk becomes one row.  The inference module re-assembles
    per-chunk predictions into an article-level verdict.

Chunking algorithm
------------------
Given tokens [t0, t1, ..., tN]:
  chunk 0 : t[0              : CHUNK_SIZE]
  chunk 1 : t[CHUNK_SIZE - OVERLAP : 2*CHUNK_SIZE - OVERLAP]
  ...
Each chunk is padded to MAX_TOKEN_LENGTH and wrapped with [CLS]/[SEP].
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ml.transformer.config import (
    MAX_TOKEN_LENGTH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_TOKENS,
    LABEL2ID,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Training / evaluation dataset
# =============================================================================

class FakeNewsDataset(Dataset):
    """
    Maps a list of text samples + integer labels to tokenized tensors.

    Parameters
    ----------
    texts     : List of cleaned article texts.
    labels    : List of integer labels (0=FAKE, 1=REAL). Pass None for
                inference (no labels returned).
    tokenizer : HuggingFace tokenizer (already loaded).
    max_length: Maximum token length (default: MAX_TOKEN_LENGTH).
    """

    def __init__(
        self,
        texts:     list[str],
        labels:    Optional[list[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        if labels is not None and len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must be the same length."
            )

        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# =============================================================================
# Chunked inference dataset
# =============================================================================

def tokenize_to_ids(
    text:      str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """
    Tokenize `text` to a flat list of token IDs (no special tokens, no padding).
    Used by the chunker to split at token boundaries.
    """
    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )


def chunk_token_ids(
    token_ids:    list[int],
    chunk_size:   int = CHUNK_SIZE,
    overlap:      int = CHUNK_OVERLAP,
    min_tokens:   int = MIN_CHUNK_TOKENS,
) -> list[list[int]]:
    """
    Split a list of token IDs into overlapping chunks.

    Parameters
    ----------
    token_ids  : Full token ID sequence for an article (no special tokens).
    chunk_size : Max tokens per chunk (before adding [CLS]/[SEP]).
    overlap    : Token overlap between consecutive chunks.
    min_tokens : Drop trailing chunks shorter than this threshold.

    Returns
    -------
    List of token-ID lists.  Each list has at most `chunk_size` elements.
    """
    if not token_ids:
        return []

    if len(token_ids) <= chunk_size:
        return [token_ids]

    chunks = []
    stride = chunk_size - overlap
    start  = 0

    while start < len(token_ids):
        end   = start + chunk_size
        chunk = token_ids[start:end]
        if len(chunk) >= min_tokens:
            chunks.append(chunk)
        start += stride

    # Edge case: if stride > chunk_size we'd skip tokens — guard against that
    if stride <= 0:
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d). Using non-overlapping chunks.",
            overlap, chunk_size,
        )
        return [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)
                if len(token_ids[i:i + chunk_size]) >= min_tokens]

    return chunks


def encode_chunk(
    chunk_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_TOKEN_LENGTH,
) -> dict[str, torch.Tensor]:
    """
    Add [CLS]/[SEP], pad/truncate to max_length, and return tensor dict.

    Constructs the special-token sequence manually to avoid relying on
    `build_inputs_with_special_tokens`, which is not available on all
    tokenizer variants (e.g. fast tokenizers in newer HF versions).
    """
    cls_id = tokenizer.cls_token_id or 101   # 101 = [CLS] in BERT vocab
    sep_id = tokenizer.sep_token_id or 102   # 102 = [SEP]
    pad_id = tokenizer.pad_token_id or 0

    # Truncate content to leave room for [CLS] and [SEP]
    content = chunk_ids[: max_length - 2]
    ids_with_special = [cls_id] + content + [sep_id]

    seq_len   = len(ids_with_special)
    n_padding = max(0, max_length - seq_len)
    input_ids      = ids_with_special + [pad_id] * n_padding
    attention_mask = [1] * seq_len    + [0]     * n_padding

    return {
        "input_ids":      torch.tensor(input_ids[:max_length],     dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask[:max_length], dtype=torch.long),
    }


class ChunkedArticleDataset(Dataset):
    """
    Splits one article into overlapping chunks for inference.

    Parameters
    ----------
    text       : Raw (or lightly cleaned) article text.
    tokenizer  : HuggingFace tokenizer.
    chunk_size : Token budget per chunk (excluding special tokens).
    overlap    : Overlap between consecutive chunks.

    Usage
    -----
    dataset  = ChunkedArticleDataset(text, tokenizer)
    loader   = DataLoader(dataset, batch_size=8)
    for batch in loader:
        outputs = model(**batch)   # batch has input_ids, attention_mask
    """

    def __init__(
        self,
        text:       str,
        tokenizer:  PreTrainedTokenizerBase,
        chunk_size: int = CHUNK_SIZE,
        overlap:    int = CHUNK_OVERLAP,
    ):
        self.tokenizer  = tokenizer
        self.max_length = MAX_TOKEN_LENGTH

        token_ids = tokenize_to_ids(text, tokenizer)

        if not token_ids:
            # Fallback: treat as single empty chunk so inference still runs
            token_ids = [tokenizer.unk_token_id or 0]

        self.chunks = chunk_token_ids(token_ids, chunk_size, overlap)

        if not self.chunks:
            self.chunks = [token_ids[:chunk_size]]

        logger.debug(
            "ChunkedArticleDataset: %d tokens → %d chunks",
            len(token_ids), len(self.chunks),
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return encode_chunk(self.chunks[idx], self.tokenizer, self.max_length)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)
