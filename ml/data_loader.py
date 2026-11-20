"""
ml/data_loader.py
=================
Unified data-loading interface for the ML engine.

This module is the single public entry point for loading datasets.
It wraps ml.preprocessing.loader and ml.preprocessing.pipeline so that
training scripts, evaluation scripts, and the backend can all call a
consistent interface without knowing the internal directory structure.

Public API
----------
  load_raw(paths)              → pd.DataFrame with ['text', 'label']
  load_processed(split)        → pd.DataFrame already cleaned+split
  load_all_splits()            → (train_df, val_df, test_df)
  prepare_if_needed(paths)     → ensure processed splits exist, return them
  describe(path)               → dict with file metadata (no full load)

Data contract
-------------
Every DataFrame returned here has at minimum:
    text          str   — raw or cleaned article text
    label         int   — 0 (FAKE) or 1 (REAL)
    cleaned_text  str   — present only in processed splits (after pipeline)

Labels are always integers (0 / 1).  Raw string labels are normalised
inside the preprocessing pipeline before being written to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from ml.preprocessing.config import TRAIN_CSV, VAL_CSV, TEST_CSV
from ml.preprocessing.loader import load_dataset, describe_file
from ml.preprocessing.pipeline import run_pipeline

logger = logging.getLogger(__name__)

Split = Union[str, None]  # "train" | "val" | "test"

# ── Public API ────────────────────────────────────────────────────────────────

def load_raw(
    paths: Union[str, Path, list[Union[str, Path]]],
) -> pd.DataFrame:
    """
    Load one or more raw dataset files.

    Parameters
    ----------
    paths : Path(s) to raw CSV / JSON / JSONL files.
            Automatically handles ISOT (Fake.csv + True.csv),
            LIAR (.tsv), and generic formats.

    Returns
    -------
    pd.DataFrame with columns ['text', 'label'] where label is a
    raw string — not yet normalised to 0/1.  Call
    ml.preprocessing.labels.normalise_labels() before training.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return load_dataset(paths)


def load_processed(split: str = "train") -> pd.DataFrame:
    """
    Load a preprocessed split from disk.

    Parameters
    ----------
    split : "train" | "val" | "test"

    Returns
    -------
    pd.DataFrame with columns including 'cleaned_text' and integer 'label'.

    Raises
    ------
    FileNotFoundError if the processed CSV does not exist.
    """
    path_map = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}
    if split not in path_map:
        raise ValueError(f"split must be one of {list(path_map.keys())}, got '{split}'")

    path = path_map[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed '{split}' split not found at {path}.\n"
            "Run: python scripts/prepare_dataset.py "
            "--files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv"
        )

    df = pd.read_csv(path)
    _coerce_types(df)
    logger.info("Loaded processed %s split: %d rows from %s", split, len(df), path)
    return df


def load_all_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three processed splits.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    return (
        load_processed("train"),
        load_processed("val"),
        load_processed("test"),
    )


def prepare_if_needed(
    paths:         list[Union[str, Path]],
    cleaning_mode: str   = "baseline",
    force:         bool  = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline if processed splits do not exist,
    then return all three splits.

    Parameters
    ----------
    paths         : Raw dataset file paths.
    cleaning_mode : "baseline" (heavy, TF-IDF) or "transformer" (light, BERT).
    force         : Re-run even if splits already exist.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    if not force and TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists():
        logger.info("Processed splits exist — skipping pipeline. Use force=True to rebuild.")
        return load_all_splits()

    logger.info("Running preprocessing pipeline…")
    train_df, val_df, test_df, _ = run_pipeline(
        paths=[Path(p) for p in paths],
        cleaning_mode=cleaning_mode,
        force=force,
    )
    return train_df, val_df, test_df


def describe(path: Union[str, Path]) -> dict:
    """
    Lightweight file introspection — does NOT fully load the file.

    Returns
    -------
    dict with keys: format, columns, sample_rows, detected_label_col,
                    sample_label_distribution
    """
    return describe_file(str(path))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _coerce_types(df: pd.DataFrame) -> None:
    """Ensure label is int and text columns are str (in-place)."""
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    for col in ("text", "cleaned_text"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
