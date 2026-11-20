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
from ml.preprocessing.audit_log     import PreprocessingAuditLog

logger = logging.getLogger(__name__)


def run_pipeline(
    paths:         Union[list[str], list[Path]],
    cleaning_mode: str   = "baseline",   # "baseline" | "transformer"
    test_size:     float = TEST_SIZE,
    val_size:      float = VAL_SIZE,
    random_seed:   int   = RANDOM_SEED,
    force:         bool  = False,
    verbose:       bool  = True,
    lemmatize:     bool  = False,        # apply lemmatization after cleaning
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
    lemmatize     : Apply lemmatization + stopword removal after cleaning.
                    Requires NLTK data (run scripts/download_nltk_data.py first).

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
    logger.info("  Lemmatize    : %s", lemmatize)
    logger.info("  Split        : test=%.0f%%  val=%.0f%%", test_size*100, val_size*100)
    logger.info("  Random seed  : %d", random_seed)
    logger.info("=" * 60)

    # Initialise audit log
    dataset_name = "+".join(Path(p).name for p in paths)
    audit = PreprocessingAuditLog(
        dataset_name=dataset_name,
        source_files=[str(p) for p in paths],
        cleaning_mode=cleaning_mode,
    )

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("[1/6] Loading dataset...")
    df = load_dataset(paths)
    logger.info("      Loaded %d rows", len(df))
    audit.record("load", rows_before=0, rows_after=len(df),
                 params={"files": [str(p) for p in paths]})

    # ── Step 2: Normalise labels ──────────────────────────────────────────────
    logger.info("[2/6] Normalising labels...")
    rows_before = len(df)
    df = normalise_labels(df, drop_unknown=True)
    logger.info("      After label normalisation: %d rows", len(df))
    audit.record("label_normalisation", rows_before=rows_before, rows_after=len(df),
                 dropped=rows_before - len(df))

    # ── Step 3: Clean text ────────────────────────────────────────────────────
    logger.info("[3/6] Cleaning text (mode=%s)...", cleaning_mode)
    df["cleaned_text"] = clean_series(df["text"], mode=cleaning_mode)

    # Optional lemmatization
    if lemmatize and cleaning_mode == "baseline":
        logger.info("      Applying lemmatization + stopword removal...")
        try:
            from ml.preprocessing.lemmatizer import lemmatize_texts
            df["cleaned_text"] = lemmatize_texts(df["cleaned_text"].tolist())
            logger.info("      Lemmatization complete.")
        except Exception as exc:
            logger.warning("      Lemmatization failed (%s) — skipping.", exc)

    # Drop rows where cleaning produced an empty string
    rows_before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    dropped = rows_before - len(df)
    if dropped:
        logger.info("      Dropped %d rows with empty text after cleaning", dropped)
    logger.info("      After cleaning: %d rows", len(df))
    audit.record("text_cleaning", rows_before=rows_before, rows_after=len(df),
                 dropped=dropped,
                 params={"mode": cleaning_mode, "lemmatize": lemmatize})

    # ── Step 4: Deduplicate ───────────────────────────────────────────────────
    logger.info("[4/6] Deduplicating...")
    rows_before = len(df)
    # Dedup on cleaned_text — catches articles that differ only in raw
    # formatting but are identical after cleaning.
    from ml.preprocessing.deduplication import remove_exact_duplicates, remove_near_duplicates
    # Work on a helper series rather than a copy, to keep index alignment
    df["_dedup_key"] = df["cleaned_text"]
    # Temporarily rename for deduplicate() which expects column "text"
    _tmp = df[["_dedup_key", "label"]].rename(columns={"_dedup_key": "text"})
    _tmp, dedup_report = deduplicate(_tmp)
    df = df.loc[_tmp.index].drop(columns=["_dedup_key"]).reset_index(drop=True)
    logger.info(
        "      After dedup: %d rows  (removed %d)",
        len(df), dedup_report["total_removed"],
    )
    audit.record("deduplication", rows_before=rows_before, rows_after=len(df),
                 dropped=dedup_report["total_removed"],
                 params=dedup_report)

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
    audit.record("split", rows_before=len(df), rows_after=len(df),
                 params={
                     "train": len(train_df), "val": len(val_df), "test": len(test_df),
                     "test_size": test_size, "val_size": val_size,
                     "random_seed": random_seed,
                 })

    # ── Step 6: Compute statistics ────────────────────────────────────────────
    logger.info("[6/6] Computing statistics...")
    stats = compute_stats(
        train_df, val_df, test_df,
        dedup_report=dedup_report,
        leakage_report=leakage_report,
    )
    audit.record("statistics", rows_before=len(df), rows_after=len(df))

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_splits(train_df, val_df, test_df)
    save_stats(stats, STATS_JSON)
    audit.save()

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
