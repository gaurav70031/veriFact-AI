"""
Preprocessing pipeline orchestrator.

Chains every step in the correct order:

  load → normalise labels → clean text → deduplicate → split → stats → save

Each step is a pure function from its own module.
The pipeline writes three CSV files to datasets/processed/:
    train.csv  |  val.csv  |  test.csv

And a statistics file:
    datasets/processed/dataset_stats.json

Usage
-----
    from ml.preprocessing.pipeline import run_pipeline
    run_pipeline(paths=["ml/datasets/raw/Fake.csv",
                         "ml/datasets/raw/True.csv"])

Or via CLI:
    python scripts/prepare_dataset.py --files ml/datasets/raw/Fake.csv \
                                               ml/datasets/raw/True.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from ml.preprocessing.config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV, STATS_JSON,
    RANDOM_SEED, TEST_SIZE, VAL_SIZE,
)
from ml.preprocessing.loader        import load_dataset
from ml.preprocessing.cleaner       import clean_series
from ml.preprocessing.labels        import normalise_labels
from ml.preprocessing.deduplication import deduplicate, check_leakage
from ml.preprocessing.splitter      import split_dataset
from ml.preprocessing.stats         import compute_stats, save_stats, print_stats

logger = logging.getLogger(__name__)


def run_pipeline(
    paths:         Union[list[str], list[Path]],
    cleaning_mode: str   = "baseline",   # "baseline" | "transformer"
    test_size:     float = TEST_SIZE,
    val_size:      float = VAL_SIZE,
    random_seed:   int   = RANDOM_SEED,
    force:         bool  = False,
    verbose:       bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    paths         : One or more raw dataset file paths.
    cleaning_mode : "baseline" (heavy, for TF-IDF) or "transformer" (light).
    test_size     : Fraction of data for test set.
    val_size      : Fraction of data for validation set.
    random_seed   : Reproducibility seed.
    force         : Re-run even if processed CSVs already exist.
    verbose       : Print statistics to stdout at the end.

    Returns
    -------
    (train_df, val_df, test_df, stats_dict)
    """
    # ── Guard: skip if already processed ─────────────────────────────────────
    if not force and TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists():
        logger.info(
            "Processed splits already exist.  "
            "Pass force=True or --force to rebuild."
        )
        train_df = pd.read_csv(TRAIN_CSV)
        val_df   = pd.read_csv(VAL_CSV)
        test_df  = pd.read_csv(TEST_CSV)
        return train_df, val_df, test_df, {}

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("  Input files  : %s", [str(p) for p in paths])
    logger.info("  Cleaning mode: %s", cleaning_mode)
    logger.info("  Split        : test=%.0f%%  val=%.0f%%", test_size*100, val_size*100)
    logger.info("  Random seed  : %d", random_seed)
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("[1/6] Loading dataset...")
    df = load_dataset(paths)
    logger.info("      Loaded %d rows", len(df))

    # ── Step 2: Normalise labels ──────────────────────────────────────────────
    logger.info("[2/6] Normalising labels...")
    df = normalise_labels(df, drop_unknown=True)
    logger.info("      After label normalisation: %d rows", len(df))

    # ── Step 3: Clean text ────────────────────────────────────────────────────
    logger.info("[3/6] Cleaning text (mode=%s)...", cleaning_mode)
    df["cleaned_text"] = clean_series(df["text"], mode=cleaning_mode)

    # Drop rows where cleaning produced an empty string
    before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("      Dropped %d rows with empty text after cleaning", dropped)
    logger.info("      After cleaning: %d rows", len(df))

    # ── Step 4: Deduplicate ───────────────────────────────────────────────────
    logger.info("[4/6] Deduplicating...")
    # Dedup is performed on the raw 'text' column (before cleaning).
    # This catches duplicate source articles regardless of cleaning mode,
    # and avoids false near-duplicates caused by heavy cleaning normalisation.
    df_dedup = df[["text", "label"]].copy()
    df_dedup, dedup_report = deduplicate(df_dedup)
    # Align both text and cleaned_text to surviving rows
    df = df.loc[df_dedup.index].reset_index(drop=True)
    logger.info(
        "      After dedup: %d rows  (removed %d)",
        len(df), dedup_report["total_removed"],
    )

    # ── Step 5: Split ─────────────────────────────────────────────────────────
    logger.info("[5/6] Splitting dataset...")
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
    )

    # Leakage check (also done inside split_dataset, but we capture the report)
    leakage_report = check_leakage(train_df, val_df, test_df)

    # ── Step 6: Compute statistics ────────────────────────────────────────────
    logger.info("[6/6] Computing statistics...")
    stats = compute_stats(
        train_df, val_df, test_df,
        dedup_report=dedup_report,
        leakage_report=leakage_report,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_splits(train_df, val_df, test_df)
    save_stats(stats, STATS_JSON)

    if verbose:
        print_stats(stats)

    logger.info("Pipeline complete.  Outputs in %s", TRAIN_CSV.parent)
    return train_df, val_df, test_df, stats


def _save_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Save the three splits to CSV."""
    for df, path in [(train_df, TRAIN_CSV), (val_df, VAL_CSV), (test_df, TEST_CSV)]:
        df.to_csv(path, index=False)
        logger.info("Saved %s  (%d rows)", path.name, len(df))
