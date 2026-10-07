"""
Reproducible stratified train / validation / test split.

Guarantees
----------
* Stratified on 'label' — each split maintains the original class ratio.
* Deterministic — controlled by RANDOM_SEED from config.py.
* No sample appears in more than one split (verified internally).
* Leakage check is run automatically after splitting.

Output
------
Three DataFrames: train_df, val_df, test_df
Each has columns: ['text', 'label']  (plus 'cleaned_text' if present)
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.preprocessing.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE
from ml.preprocessing.deduplication import check_leakage

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    test_size:   float = TEST_SIZE,
    val_size:    float = VAL_SIZE,
    random_seed: int   = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 3-way split.

    Parameters
    ----------
    df          : Cleaned, deduplicated DataFrame with 'text' and 'label'.
    test_size   : Fraction of total data held out for test (default 0.10).
    val_size    : Fraction of total data used for validation (default 0.10).
    random_seed : Controls reproducibility.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    if "label" not in df.columns:
        raise KeyError("DataFrame must have a 'label' column before splitting.")
    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} rows — too small to split meaningfully."
        )

    # ── Step 1: hold out test set ─────────────────────────────────────────────
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"],
        shuffle=True,
    )

    # ── Step 2: split remainder into train + val ──────────────────────────────
    # val_size_adjusted = val fraction relative to the train+val pool
    val_size_adjusted = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=train_val_df["label"],
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # ── Verify no overlap ─────────────────────────────────────────────────────
    _verify_no_overlap(train_df, val_df, test_df)

    # ── Leakage check ─────────────────────────────────────────────────────────
    check_leakage(train_df, val_df, test_df)

    # ── Log split stats ───────────────────────────────────────────────────────
    _log_split_stats(train_df, val_df, test_df)

    return train_df, val_df, test_df


def _verify_no_overlap(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """
    Assert that no index appears in more than one split.
    Because sklearn's train_test_split always produces disjoint sets when
    given the same original DataFrame, this is a sanity check against
    accidental concatenation bugs upstream.
    """
    # Use the actual text as the identity check (not the index, which resets)
    train_texts = set(train_df["text"])
    val_texts   = set(val_df["text"])
    test_texts  = set(test_df["text"])

    tv_overlap = train_texts & val_texts
    tt_overlap = train_texts & test_texts
    vt_overlap = val_texts   & test_texts

    for name, overlap in [
        ("train∩val", tv_overlap),
        ("train∩test", tt_overlap),
        ("val∩test", vt_overlap),
    ]:
        if overlap:
            # This should never happen after deduplication — raise hard
            raise RuntimeError(
                f"Split overlap detected in {name}: {len(overlap)} shared samples.  "
                "Run deduplication before splitting."
            )


def _log_split_stats(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    logger.info("Dataset split complete  (total=%d)", total)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fake = (df["label"] == 0).sum()
        real = (df["label"] == 1).sum()
        pct  = len(df) / total * 100
        logger.info(
            "  %-5s : %6d rows (%4.1f%%)  FAKE=%d  REAL=%d",
            name, len(df), pct, fake, real,
        )
