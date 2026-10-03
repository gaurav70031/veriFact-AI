"""
Dataset statistics.

Computes REAL statistics from actual data — no fabricated numbers.

Reports
-------
* Row counts per split and overall
* Class distribution (count + percentage)
* Text length distribution (min, max, mean, median, p5, p95)
* Vocabulary size (unique token count)
* Average tokens per sample
* Empty / very-short text count after cleaning
* Leakage counts (passed in from deduplication module)
* Duplicate counts removed during dedup

All statistics are computed lazily from the DataFrames passed in.
Results are returned as a dict and optionally serialised to JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _text_length_stats(series: pd.Series) -> dict:
    """Character-level length statistics for a text Series."""
    lengths = series.str.len().dropna()
    if len(lengths) == 0:
        return {}
    return {
        "min":    int(lengths.min()),
        "max":    int(lengths.max()),
        "mean":   round(float(lengths.mean()), 1),
        "median": round(float(lengths.median()), 1),
        "p5":     round(float(np.percentile(lengths, 5)), 1),
        "p95":    round(float(np.percentile(lengths, 95)), 1),
    }


def _token_stats(series: pd.Series) -> dict:
    """
    Approximate word-level token statistics.
    Uses whitespace splitting — no heavy tokenizer dependency here.
    """
    token_counts = series.dropna().map(lambda t: len(str(t).split()))
    if len(token_counts) == 0:
        return {}

    # Vocabulary: unique tokens across all texts (memory-efficient approximation)
    all_tokens: set[str] = set()
    sample = series.dropna().sample(
        min(5_000, len(series)), random_state=0
    )
    for text in sample:
        all_tokens.update(str(text).lower().split())

    return {
        "mean_tokens":        round(float(token_counts.mean()), 1),
        "median_tokens":      round(float(token_counts.median()), 1),
        "p5_tokens":          round(float(np.percentile(token_counts, 5)), 1),
        "p95_tokens":         round(float(np.percentile(token_counts, 95)), 1),
        "vocab_size_estimate": len(all_tokens),
    }


def _split_stats(df: pd.DataFrame, split_name: str) -> dict:
    """Compute per-split statistics."""
    total = len(df)
    label_counts = df["label"].value_counts().to_dict()
    fake_count = int(label_counts.get(0, 0))
    real_count = int(label_counts.get(1, 0))

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"

    return {
        "split":         split_name,
        "total_rows":    total,
        "fake_count":    fake_count,
        "real_count":    real_count,
        "fake_pct":      round(fake_count / total * 100, 2) if total else 0.0,
        "real_pct":      round(real_count / total * 100, 2) if total else 0.0,
        "text_length":   _text_length_stats(df[text_col]),
        "token_stats":   _token_stats(df[text_col]),
        "empty_texts":   int((df[text_col].str.strip() == "").sum()),
    }


def compute_stats(
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    test_df:       pd.DataFrame,
    dedup_report:  dict | None = None,
    leakage_report: dict | None = None,
) -> dict:
    """
    Compute full dataset statistics from actual DataFrame contents.

    Parameters
    ----------
    train_df, val_df, test_df  : DataFrames produced by the pipeline.
    dedup_report               : Dict from deduplication.deduplicate().
    leakage_report             : Dict from deduplication.check_leakage().

    Returns
    -------
    A nested dict — serialisable to JSON.
    """
    total_rows = len(train_df) + len(val_df) + len(test_df)

    stats: dict = {
        "total_rows": total_rows,
        "splits": {
            "train": _split_stats(train_df, "train"),
            "val":   _split_stats(val_df,   "val"),
            "test":  _split_stats(test_df,  "test"),
        },
        "deduplication": dedup_report or {},
        "leakage":       leakage_report or {},
    }

    logger.info(
        "Dataset statistics:\n"
        "  Total rows : %d\n"
        "  Train      : %d  (FAKE=%d, REAL=%d)\n"
        "  Val        : %d  (FAKE=%d, REAL=%d)\n"
        "  Test       : %d  (FAKE=%d, REAL=%d)",
        total_rows,
        stats["splits"]["train"]["total_rows"],
        stats["splits"]["train"]["fake_count"],
        stats["splits"]["train"]["real_count"],
        stats["splits"]["val"]["total_rows"],
        stats["splits"]["val"]["fake_count"],
        stats["splits"]["val"]["real_count"],
        stats["splits"]["test"]["total_rows"],
        stats["splits"]["test"]["fake_count"],
        stats["splits"]["test"]["real_count"],
    )
    return stats


def save_stats(stats: dict, path: Path) -> None:
    """Serialise statistics dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Dataset statistics saved to %s", path)


def print_stats(stats: dict) -> None:
    """Pretty-print statistics to stdout."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total rows : {stats['total_rows']:,}")
    print()

    for split_name, s in stats["splits"].items():
        print(f"  {split_name.upper()} ({s['total_rows']:,} rows)")
        print(f"    FAKE : {s['fake_count']:,}  ({s['fake_pct']:.1f}%)")
        print(f"    REAL : {s['real_count']:,}  ({s['real_pct']:.1f}%)")
        tl = s.get("text_length", {})
        if tl:
            print(
                f"    Text length  min={tl['min']}  "
                f"mean={tl['mean']:.0f}  max={tl['max']}"
            )
        tk = s.get("token_stats", {})
        if tk:
            print(
                f"    Tokens/sample  mean={tk['mean_tokens']:.0f}  "
                f"vocab≈{tk['vocab_size_estimate']:,}"
            )
        if s.get("empty_texts"):
            print(f"    Empty texts after cleaning: {s['empty_texts']}")
        print()

    dedup = stats.get("deduplication", {})
    if dedup:
        print(
            f"  Deduplication: exact={dedup.get('exact_removed', 0)}  "
            f"near={dedup.get('near_removed', 0)}  "
            f"total={dedup.get('total_removed', 0)}"
        )

    leak = stats.get("leakage", {})
    if leak:
        tv = leak.get("train_val_leakage", 0)
        tt = leak.get("train_test_leakage", 0)
        status = "CLEAN" if tv == 0 and tt == 0 else "WARNING — leakage detected"
        print(f"  Leakage check : {status}  (train→val={tv}  train→test={tt})")

    print("=" * 60 + "\n")
