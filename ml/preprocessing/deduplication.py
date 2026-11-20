"""
Duplicate detection and data-leakage prevention.

Three levels of checking
------------------------
1. Exact duplicates
   Same raw text string (after strip).  Always removed.

2. Near-duplicates (normalised hash)
   Text lowercased + whitespace-collapsed.  Catches trivial variants
   like extra spaces or punctuation differences.

3. Cross-split leakage detection
   After splitting, verifies no normalised text from train appears in
   val or test.  Reports — but does NOT silently drop — leaking rows
   so the caller can decide.

None of these steps use external libraries beyond pandas and hashlib,
so the pipeline has no heavy dedup dependency.
"""

from __future__ import annotations

import hashlib
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_RE_WHITESPACE = re.compile(r"\s+")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_for_hash(text: str) -> str:
    """Lowercase + collapse whitespace — used for near-dup hashing."""
    return _RE_WHITESPACE.sub(" ", str(text).lower()).strip()


def _hash_text(text: str) -> str:
    """MD5 hex digest of a normalised text string (fast, collision-resistant enough)."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows where 'text' is character-for-character identical.

    Returns
    -------
    (deduplicated_df, n_removed)
    """
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("Exact dedup: removed %d rows (%d → %d)", removed, before, len(df))
    return df, removed


def remove_near_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose text is identical after lowercasing + whitespace collapse.
    Applied after exact dedup.

    Returns
    -------
    (deduplicated_df, n_removed)
    """
    before = len(df)
    df = df.copy()
    df["_norm_hash"] = df["text"].map(_normalise_for_hash).map(_hash_text)
    df = df.drop_duplicates(subset=["_norm_hash"], keep="first")
    df = df.drop(columns=["_norm_hash"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info(
            "Near-dup dedup: removed %d rows (%d → %d)", removed, before, len(df)
        )
    return df, removed


def remove_cross_label_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose text appears under more than one label.

    When the same article is filed as both FAKE and REAL (as occurs in the
    ISOT dataset), it is ambiguous and should not be used for training.
    All rows for that text are dropped.

    Returns
    -------
    (clean_df, n_removed)
    """
    before = len(df)
    # Count how many distinct labels each text has
    label_counts = df.groupby("text")["label"].nunique()
    ambiguous_texts = label_counts[label_counts > 1].index
    df = df[~df["text"].isin(ambiguous_texts)].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info(
            "Cross-label dedup: removed %d rows with conflicting labels (%d → %d)",
            removed, before, len(df),
        )
    return df, removed


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Run exact dedup, near-dup removal, and cross-label dedup.

    Returns
    -------
    (clean_df, report_dict)
        report_dict keys: "exact_removed", "near_removed", "cross_label_removed",
                          "total_removed"
    """
    df, exact = remove_exact_duplicates(df)
    df, near  = remove_near_duplicates(df)
    df, cross = remove_cross_label_duplicates(df)
    report = {
        "exact_removed":       exact,
        "near_removed":        near,
        "cross_label_removed": cross,
        "total_removed":       exact + near + cross,
    }
    logger.info(
        "Deduplication summary: exact=%d  near=%d  cross_label=%d  total=%d  remaining=%d",
        exact, near, cross, exact + near + cross, len(df),
    )
    return df, report


def check_leakage(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> dict[str, int]:
    """
    Detect cross-split leakage: samples in val or test whose normalised
    text hash appears in train.

    Does NOT modify any DataFrame — only reports counts.

    Returns
    -------
    dict with keys:
        "train_val_leakage"  — # val rows that appear in train
        "train_test_leakage" — # test rows that appear in train
    """
    def _hashes(df: pd.DataFrame) -> set[str]:
        return set(df["text"].map(_normalise_for_hash).map(_hash_text))

    train_hashes = _hashes(train_df)

    val_leak  = sum(1 for h in _hashes(val_df)  if h in train_hashes)
    test_leak = sum(1 for h in _hashes(test_df) if h in train_hashes)

    if val_leak:
        logger.warning(
            "LEAKAGE: %d val samples have matching text in train set.", val_leak
        )
    if test_leak:
        logger.warning(
            "LEAKAGE: %d test samples have matching text in train set.", test_leak
        )
    if not val_leak and not test_leak:
        logger.info("Leakage check passed: no train→val or train→test overlap.")

    return {
        "train_val_leakage":  val_leak,
        "train_test_leakage": test_leak,
    }
