"""
ml/preprocessing_api.py
========================
Unified preprocessing interface.

Wraps ml.preprocessing.cleaner so that any module can call a single
function without knowing which internal cleaning mode to use.

This satisfies the spec requirement for a top-level preprocessing.py
while keeping the actual implementation in its proper location under
ml/preprocessing/.

Public API
----------
  clean_text(text, mode)         → str
  clean_texts(texts, mode)       → list[str]
  clean_for_inference(text)      → str    (baseline mode, used by predict.py)
  clean_for_training(texts, mode)→ list[str]
  get_cleaning_config()          → dict   (what each mode does)
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from ml.preprocessing.cleaner import clean_for_baseline, clean_for_transformer, clean_series
from ml.preprocessing.config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

Mode = Literal["baseline", "transformer"]

# ── Public API ────────────────────────────────────────────────────────────────

def clean_text(text: str, mode: Mode = "baseline") -> str:
    """
    Clean a single text string.

    Parameters
    ----------
    text : Raw input text.
    mode : "baseline" — heavy normalisation for TF-IDF models.
           "transformer" — light normalisation for BERT-family models.

    Returns
    -------
    Cleaned string.  Empty string if the text is too short after cleaning.
    """
    if mode == "baseline":
        return clean_for_baseline(text)
    elif mode == "transformer":
        return clean_for_transformer(text)
    else:
        raise ValueError(f"mode must be 'baseline' or 'transformer', got '{mode}'")


def clean_texts(texts: list[str], mode: Mode = "baseline") -> list[str]:
    """
    Clean a list of text strings.

    Parameters
    ----------
    texts : List of raw input strings.
    mode  : Cleaning mode (see clean_text).

    Returns
    -------
    List of cleaned strings in the same order.  Empty strings are preserved
    (not dropped) so indices align with the input.
    """
    series = pd.Series(texts, dtype=str)
    cleaned = clean_series(series, mode=mode)
    return cleaned.tolist()


def clean_for_inference(text: str) -> str:
    """
    Clean text for inference with baseline TF-IDF models.

    This is the function called by predict.py before passing text to
    sklearn pipelines.  It uses the same cleaning as training time,
    guaranteeing consistency.

    Parameters
    ----------
    text : Raw input text (article, claim, URL-extracted content).

    Returns
    -------
    Cleaned string.  Empty string if the input is too short.
    """
    return clean_for_baseline(text)


def clean_for_training(
    texts: list[str],
    mode:  Mode = "baseline",
) -> list[str]:
    """
    Clean a batch of texts for model training.

    Identical to clean_texts() but logs the number of empty texts
    produced (useful for training scripts to surface data quality issues).

    Parameters
    ----------
    texts : List of raw training texts.
    mode  : Cleaning mode.

    Returns
    -------
    List of cleaned strings.
    """
    cleaned = clean_texts(texts, mode=mode)
    empty = sum(1 for t in cleaned if not t.strip())
    if empty:
        logger.warning(
            "clean_for_training(mode=%s): %d / %d texts became empty after cleaning.",
            mode, empty, len(texts),
        )
    return cleaned


def get_cleaning_config() -> dict:
    """
    Return a dict documenting what each cleaning mode does.
    Used by predict.py to include in prediction metadata.
    """
    return {
        "baseline": {
            "description": "Heavy normalisation for TF-IDF baseline models",
            "steps": [
                "Decode HTML entities",
                "Strip HTML/XML tags",
                "Remove Reuters dateline",
                "Remove URLs",
                "Remove email addresses",
                "Unicode NFKC normalisation",
                "Lowercase",
                "Expand contractions",
                "Remove non-alphabetic characters",
                "Compress repeated characters",
                "Collapse whitespace",
            ],
            "min_length": MIN_TEXT_LENGTH,
            "max_length": MAX_TEXT_LENGTH,
            "preserves_casing": False,
            "preserves_punctuation": False,
        },
        "transformer": {
            "description": "Light normalisation for BERT-family models",
            "steps": [
                "Decode HTML entities",
                "Strip HTML/XML tags",
                "Remove Reuters dateline",
                "Remove URLs",
                "Remove email addresses",
                "Unicode NFKC normalisation",
                "Collapse whitespace",
            ],
            "min_length": 0,
            "max_length": MAX_TEXT_LENGTH,
            "preserves_casing": True,
            "preserves_punctuation": True,
        },
    }
