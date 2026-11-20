"""
ml/preprocessing/analytics.py
==============================
Dataset analytics generator.

Computes all statistics from the actual DataFrames — nothing hardcoded.
Called by the backend dataset API endpoint and by the CLI.

Public API
----------
  word_frequency(df, label, top_n)        -> list[{"word", "count", "pct"}]
  length_distribution(df)                 -> dict
  quality_report(df, raw_df)              -> dict
  class_distribution(df)                  -> dict
  missing_value_report(df)                -> dict
  full_analytics(df, raw_df, cleaned_df)  -> dict   (complete report)
"""

from __future__ import annotations

import collections
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.preprocessing.lemmatizer import get_stopwords

logger = logging.getLogger(__name__)

_RE_WHITESPACE = re.compile(r"\s+")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize_simple(text: str) -> list[str]:
    """Whitespace-split, lowercase, alphabetic only — no heavy NLP needed."""
    return re.findall(r"[a-z]+", str(text).lower())


def _get_text_col(df: pd.DataFrame) -> str:
    """Pick the best available text column."""
    for col in ("cleaned_text", "text"):
        if col in df.columns:
            return col
    raise KeyError(f"DataFrame has no 'text' or 'cleaned_text' column. Got: {df.columns.tolist()}")


# ── Word frequency ────────────────────────────────────────────────────────────

def word_frequency(
    df:     pd.DataFrame,
    label:  Optional[int] = None,
    top_n:  int = 50,
    remove_stopwords: bool = True,
) -> list[dict]:
    """
    Compute word frequency for a dataset split.

    Parameters
    ----------
    df              : DataFrame with text column and optional label column.
    label           : Filter to only rows with this label (0=FAKE, 1=REAL, None=all).
    top_n           : Number of top words to return.
    remove_stopwords: Strip NLTK English + domain stopwords.

    Returns
    -------
    List of {"word": str, "count": int, "pct": float} dicts, sorted by count desc.
    """
    text_col = _get_text_col(df)

    if label is not None and "label" in df.columns:
        subset = df[df["label"] == label][text_col]
    else:
        subset = df[text_col]

    stopwords = get_stopwords() if remove_stopwords else frozenset()
    counter: collections.Counter = collections.Counter()

    for text in subset.dropna():
        tokens = _tokenize_simple(str(text))
        filtered = [t for t in tokens if len(t) > 2 and t not in stopwords]
        counter.update(filtered)

    total = sum(counter.values()) or 1
    return [
        {"word": word, "count": count, "pct": round(count / total * 100, 4)}
        for word, count in counter.most_common(top_n)
    ]


# ── Length distribution ───────────────────────────────────────────────────────

def length_distribution(df: pd.DataFrame) -> dict:
    """
    Character-level and word-level article length statistics, broken down by label.

    Returns
    -------
    {
        "overall": {"char": {...}, "word": {...}},
        "by_label": {
            "FAKE": {"char": {...}, "word": {...}},
            "REAL": {"char": {...}, "word": {...}},
        },
        "histogram": [{"bucket": "0-500", "count": int}, ...]
    }
    """
    text_col = _get_text_col(df)
    texts = df[text_col].fillna("").astype(str)

    char_lens = texts.str.len()
    word_lens = texts.map(lambda t: len(t.split()))

    def _stats(arr: pd.Series) -> dict:
        if arr.empty:
            return {}
        return {
            "min":    int(arr.min()),
            "max":    int(arr.max()),
            "mean":   round(float(arr.mean()), 1),
            "median": round(float(arr.median()), 1),
            "p5":     round(float(np.percentile(arr, 5)), 1),
            "p25":    round(float(np.percentile(arr, 25)), 1),
            "p75":    round(float(np.percentile(arr, 75)), 1),
            "p95":    round(float(np.percentile(arr, 95)), 1),
        }

    result: dict = {
        "overall": {
            "char": _stats(char_lens),
            "word": _stats(word_lens),
        },
        "by_label": {},
    }

    if "label" in df.columns:
        for lbl_id, lbl_name in {0: "FAKE", 1: "REAL"}.items():
            mask = df["label"] == lbl_id
            if mask.any():
                result["by_label"][lbl_name] = {
                    "char": _stats(char_lens[mask]),
                    "word": _stats(word_lens[mask]),
                }

    # Histogram (500-char buckets, capped at 10000+)
    buckets = [
        (0,    500,   "0–500"),
        (500,  1000,  "500–1k"),
        (1000, 2000,  "1k–2k"),
        (2000, 3000,  "2k–3k"),
        (3000, 5000,  "3k–5k"),
        (5000, 8000,  "5k–8k"),
        (8000, 10001, "8k+"),
    ]
    histogram = []
    for lo, hi, label in buckets:
        count = int(((char_lens >= lo) & (char_lens < hi)).sum())
        histogram.append({"bucket": label, "count": count})
    result["histogram"] = histogram

    return result


# ── Class distribution ────────────────────────────────────────────────────────

def class_distribution(df: pd.DataFrame) -> dict:
    """
    Label distribution with counts and percentages.

    Returns
    -------
    {
        "total": int,
        "fake":  {"count": int, "pct": float},
        "real":  {"count": int, "pct": float},
        "balance_ratio": float,   # real/fake — 1.0 = perfectly balanced
    }
    """
    if "label" not in df.columns:
        return {"total": len(df), "fake": {}, "real": {}, "balance_ratio": None}

    total = len(df)
    fake_count = int((df["label"] == 0).sum())
    real_count = int((df["label"] == 1).sum())
    total_nz   = max(total, 1)

    return {
        "total":  total,
        "fake":   {"count": fake_count, "pct": round(fake_count / total_nz * 100, 2)},
        "real":   {"count": real_count, "pct": round(real_count / total_nz * 100, 2)},
        "balance_ratio": round(real_count / fake_count, 4) if fake_count > 0 else None,
    }


# ── Missing-value report ──────────────────────────────────────────────────────

def missing_value_report(df: pd.DataFrame) -> dict:
    """
    Per-column missing value counts and percentages.

    Returns
    -------
    {
        "total_rows": int,
        "columns": [{"name": str, "missing": int, "pct": float}, ...]
    }
    """
    total = len(df)
    total_nz = max(total, 1)
    cols = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        # Also count empty strings as missing for text columns
        if df[col].dtype == object:
            missing += int((df[col] == "").sum())
        cols.append({
            "name":    col,
            "missing": missing,
            "pct":     round(missing / total_nz * 100, 2),
        })
    return {
        "total_rows": total,
        "columns":    sorted(cols, key=lambda c: c["missing"], reverse=True),
    }


# ── Duplicate report ──────────────────────────────────────────────────────────

def duplicate_report(df: pd.DataFrame) -> dict:
    """
    Count exact and near-duplicate records.

    Returns
    -------
    {
        "exact_duplicates":    int,
        "near_duplicates":     int,
        "total_duplicates":    int,
        "pct_duplicate":       float,
    }
    """
    text_col = _get_text_col(df)
    texts = df[text_col].fillna("").astype(str)
    total = len(df)

    # Exact duplicates
    exact = int(texts.duplicated().sum())

    # Near-duplicates (lowercase + collapsed whitespace)
    normed = texts.map(lambda t: _RE_WHITESPACE.sub(" ", t.lower()).strip())
    near = int(normed.duplicated().sum()) - exact

    total_dup = exact + max(near, 0)
    return {
        "exact_duplicates":  exact,
        "near_duplicates":   max(near, 0),
        "total_duplicates":  total_dup,
        "pct_duplicate":     round(total_dup / max(total, 1) * 100, 2),
    }


# ── Quality report ────────────────────────────────────────────────────────────

def quality_report(
    df:     pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Comprehensive dataset quality report.

    Parameters
    ----------
    df      : Processed DataFrame (with cleaned_text if available).
    raw_df  : Original raw DataFrame before preprocessing (optional).

    Returns
    -------
    {
        "rows_processed":        int,
        "rows_raw":              int | None,
        "rows_dropped":          int | None,
        "pct_dropped":           float | None,
        "missing_values":        dict,
        "duplicates":            dict,
        "class_distribution":    dict,
        "text_quality": {
            "empty_after_clean": int,
            "very_short":        int,   # < 50 chars
            "very_long":         int,   # > 5000 chars
        },
        "issues":        list[str],     # human-readable warnings
        "quality_score": float,         # 0.0–1.0 (higher = better)
    }
    """
    mv_report  = missing_value_report(df)
    dup_report = duplicate_report(df)
    cls_report = class_distribution(df)
    text_col   = _get_text_col(df)
    texts      = df[text_col].fillna("").astype(str)

    total = len(df)
    empty_after_clean = int((texts.str.strip() == "").sum())
    very_short = int((texts.str.len() < 50).sum())
    very_long  = int((texts.str.len() > 5_000).sum())

    rows_raw     = len(raw_df) if raw_df is not None else None
    rows_dropped = (rows_raw - total) if rows_raw is not None else None
    pct_dropped  = (
        round(rows_dropped / max(rows_raw, 1) * 100, 2)
        if rows_dropped is not None else None
    )

    # Collect issues
    issues: list[str] = []
    if empty_after_clean > 0:
        issues.append(f"{empty_after_clean} rows have empty text after cleaning.")
    if dup_report["exact_duplicates"] > 0:
        issues.append(f"{dup_report['exact_duplicates']} exact duplicate records found.")
    if dup_report["near_duplicates"] > 0:
        issues.append(f"{dup_report['near_duplicates']} near-duplicate records found.")
    if cls_report.get("balance_ratio") is not None:
        ratio = cls_report["balance_ratio"]
        if ratio < 0.7 or ratio > 1.43:
            issues.append(
                f"Class imbalance detected: "
                f"REAL/FAKE ratio = {ratio:.2f} "
                f"(ideal: 0.9–1.1)."
            )
    total_missing = sum(c["missing"] for c in mv_report["columns"])
    if total_missing > 0:
        issues.append(f"{total_missing} total missing values across all columns.")
    if pct_dropped is not None and pct_dropped > 5:
        issues.append(f"{pct_dropped:.1f}% of raw rows dropped during preprocessing.")

    # Quality score (simple heuristic: start at 1.0 and penalise issues)
    score = 1.0
    score -= min(dup_report["pct_duplicate"] / 100 * 0.3, 0.3)
    score -= min(empty_after_clean / max(total, 1) * 0.2, 0.2)
    if cls_report.get("balance_ratio") is not None:
        ratio = cls_report["balance_ratio"]
        imbalance = abs(1.0 - ratio)
        score -= min(imbalance * 0.15, 0.15)
    if pct_dropped is not None:
        score -= min(pct_dropped / 100 * 0.1, 0.1)
    score = round(max(score, 0.0), 4)

    return {
        "rows_processed":     total,
        "rows_raw":           rows_raw,
        "rows_dropped":       rows_dropped,
        "pct_dropped":        pct_dropped,
        "missing_values":     mv_report,
        "duplicates":         dup_report,
        "class_distribution": cls_report,
        "text_quality": {
            "empty_after_clean": empty_after_clean,
            "very_short":        very_short,
            "very_long":         very_long,
        },
        "issues":        issues,
        "quality_score": score,
    }


# ── Full analytics ────────────────────────────────────────────────────────────

def full_analytics(
    df:         pd.DataFrame,
    raw_df:     Optional[pd.DataFrame] = None,
    top_words:  int = 30,
) -> dict:
    """
    Generate the complete analytics report for a dataset.

    Parameters
    ----------
    df         : Processed split (from load_processed() or run_pipeline()).
    raw_df     : Original raw DataFrame (before preprocessing, optional).
    top_words  : Top-N words to include in frequency lists.

    Returns
    -------
    Complete analytics dict ready to serialise as JSON.
    """
    return {
        "class_distribution":    class_distribution(df),
        "length_distribution":   length_distribution(df),
        "missing_values":        missing_value_report(df),
        "duplicates":            duplicate_report(df),
        "quality_report":        quality_report(df, raw_df=raw_df),
        "word_frequency": {
            "all":  word_frequency(df, label=None, top_n=top_words),
            "fake": word_frequency(df, label=0,    top_n=top_words),
            "real": word_frequency(df, label=1,    top_n=top_words),
        },
    }


# ── Sample rows ───────────────────────────────────────────────────────────────

def sample_rows(
    df:        pd.DataFrame,
    n:         int = 10,
    label:     Optional[int] = None,
    random_state: int = 42,
) -> list[dict]:
    """
    Return a random sample of rows as a list of dicts.

    Parameters
    ----------
    df           : DataFrame to sample from.
    n            : Number of rows to return.
    label        : Filter to this label before sampling (None = all labels).
    random_state : Reproducibility seed.

    Returns
    -------
    List of row dicts — safe to JSON-serialise.
    """
    subset = df if label is None else df[df["label"] == label]
    sample = subset.sample(min(n, len(subset)), random_state=random_state)
    rows = []
    for _, row in sample.iterrows():
        d: dict = {}
        for col in row.index:
            val = row[col]
            if isinstance(val, float) and (val != val):  # NaN
                d[col] = None
            elif hasattr(val, "item"):  # numpy scalar
                d[col] = val.item()
            else:
                d[col] = str(val) if not isinstance(val, (int, float, bool, type(None))) else val
        rows.append(d)
    return rows
