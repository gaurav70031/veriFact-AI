"""
Dataset loader.

Responsibilities
----------------
1. Accept one or more file paths (CSV or JSON/JSONL).
2. Auto-detect which columns carry the text, title, and label.
3. Merge title + body into a single 'text' column.
4. Validate that mandatory columns are present.
5. Return a normalised DataFrame with exactly two columns:
       text  (str)
       label (str, raw — not yet integer-encoded)

Supported formats
-----------------
CSV   — standard comma-separated; also tab-separated (.tsv)
JSON  — list of objects: [{"text": "...", "label": "..."}, ...]
JSONL — one JSON object per line (common in HuggingFace datasets)

Supported datasets (auto-detected by filename heuristics)
----------------------------------------------------------
* ISOT Fake News Dataset   Fake.csv + True.csv
* LIAR dataset             train.tsv / valid.tsv / test.tsv
* FakeNewsNet              news content.json files
* Generic                  any CSV/JSON with detectable text+label columns
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from ml.preprocessing.config import (
    TEXT_FIELD_CANDIDATES,
    TITLE_FIELD_CANDIDATES,
    LABEL_FIELD_CANDIDATES,
)

logger = logging.getLogger(__name__)

# ── LIAR dataset column names (positional, no header) ────────────────────────
# https://arxiv.org/abs/1705.00648
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count", "context",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip all column names in-place."""
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that matches a candidate (substring)."""
    cols = df.columns.tolist()
    for candidate in candidates:
        for col in cols:
            if candidate in col:
                return col
    return None


def _merge_title_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    If both a title and a text column exist, prepend the title to the text.
    Result is stored in 'text'.  Original columns are dropped.
    """
    title_col = _find_col(df, TITLE_FIELD_CANDIDATES)
    text_col  = _find_col(df, TEXT_FIELD_CANDIDATES)

    if text_col is None:
        raise ValueError(
            f"Could not find a text column.  "
            f"Searched for: {TEXT_FIELD_CANDIDATES}.  "
            f"Available columns: {df.columns.tolist()}"
        )

    if title_col and title_col != text_col:
        df["text"] = (
            df[title_col].fillna("").astype(str).str.strip()
            + " "
            + df[text_col].fillna("").astype(str).str.strip()
        ).str.strip()
        logger.debug("Merged columns '%s' + '%s' → 'text'", title_col, text_col)
    else:
        df["text"] = df[text_col].fillna("").astype(str).str.strip()
        logger.debug("Using column '%s' as 'text'", text_col)

    return df


def _extract_label_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the label column and rename it to 'label'.
    Raises ValueError if none found.
    """
    label_col = _find_col(df, LABEL_FIELD_CANDIDATES)
    if label_col is None:
        raise ValueError(
            f"Could not find a label column.  "
            f"Searched for: {LABEL_FIELD_CANDIDATES}.  "
            f"Available columns: {df.columns.tolist()}"
        )
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
        logger.debug("Renamed column '%s' → 'label'", label_col)
    return df


# ── Format-specific readers ───────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV or TSV, automatically detecting delimiter."""
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise IOError(f"Failed to read CSV file {path}: {exc}") from exc
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def _read_json(path: Path) -> pd.DataFrame:
    """Read JSON (array) or JSONL (one object per line)."""
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()

    if stripped.startswith("["):
        # Standard JSON array
        try:
            records = json.loads(stripped)
            df = pd.DataFrame(records).astype(str)
        except json.JSONDecodeError as exc:
            raise IOError(f"Invalid JSON in {path}: {exc}") from exc
    else:
        # JSONL — one object per line
        records = []
        for lineno, line in enumerate(stripped.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IOError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
        df = pd.DataFrame(records).astype(str)

    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


# ── ISOT-specific loader ──────────────────────────────────────────────────────

def _load_isot(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """
    Special handler for the ISOT dataset which comes as two separate files
    (Fake.csv and True.csv) with no label column.
    """
    fake_df = _read_csv(fake_path)
    fake_df = _lower_cols(fake_df)
    fake_df["label"] = "FAKE"

    true_df = _read_csv(true_path)
    true_df = _lower_cols(true_df)
    true_df["label"] = "REAL"

    df = pd.concat([fake_df, true_df], ignore_index=True)
    logger.info(
        "ISOT: %d fake + %d real = %d total",
        len(fake_df), len(true_df), len(df),
    )
    return df


# ── LIAR-specific loader ──────────────────────────────────────────────────────

def _load_liar(path: Path) -> pd.DataFrame:
    """
    LIAR dataset has no header row — assign column names from the paper.
    Maps 6-class labels to binary: pants-fire/false/barely-true → FAKE,
    half-true/mostly-true/true → REAL.
    Caller should pass the mapping through labels.py for full normalisation.
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if len(df.columns) == len(LIAR_COLUMNS):
        df.columns = LIAR_COLUMNS
    else:
        # Graceful fallback: use positional names
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        logger.warning(
            "LIAR file %s has %d columns (expected %d). "
            "Using positional names.",
            path.name, len(df.columns), len(LIAR_COLUMNS),
        )
    logger.info("Loaded LIAR file %s: %d rows", path.name, len(df))
    return df


# ── Generic single-file loader ────────────────────────────────────────────────

def _load_single(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".tab"):
        df = _read_csv(path)
    elif suffix in (".json", ".jsonl"):
        df = _read_json(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}' for {path}. "
            "Supported: .csv, .tsv, .json, .jsonl"
        )
    return _lower_cols(df)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(
    paths: Union[Path, list[Path], str, list[str]],
) -> pd.DataFrame:
    """
    Load one or more dataset files and return a normalised DataFrame
    with columns ['text', 'label'].

    Parameters
    ----------
    paths:
        A single path or list of paths.
        Special cases handled automatically:
          - [Fake.csv, True.csv]  → ISOT format
          - *.tsv with 14 columns → LIAR format
          - Any CSV/JSON          → generic auto-detect

    Returns
    -------
    pd.DataFrame with columns:
        text  (str) — raw, uncleaned article text
        label (str) — raw label string (not yet normalised to 0/1)
    """
    # Normalise to list of Path objects
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(p) for p in paths]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                "Place your dataset file(s) in ml/datasets/raw/ "
                "and pass the path(s) to load_dataset()."
            )

    # ── ISOT detection: exactly two files named Fake.csv + True.csv ──────────
    names = {p.name.lower() for p in paths}
    if names == {"fake.csv", "true.csv"}:
        fake_path = next(p for p in paths if p.name.lower() == "fake.csv")
        true_path = next(p for p in paths if p.name.lower() == "true.csv")
        df = _load_isot(fake_path, true_path)
    elif len(paths) == 1:
        df = _load_single(paths[0])
        # LIAR detection: TSV with expected column count
        if paths[0].suffix.lower() in (".tsv", ".tab"):
            if len(df.columns) == len(LIAR_COLUMNS):
                if len(df.columns) == len(LIAR_COLUMNS):
                    df.columns = LIAR_COLUMNS
    else:
        # Multiple generic files: load each and concatenate
        frames = [_load_single(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        logger.info("Concatenated %d files → %d total rows", len(paths), len(df))

    # ── Normalise to [text, label] ────────────────────────────────────────────
    df = _merge_title_text(df)
    df = _extract_label_col(df)

    # Keep only the two canonical columns
    df = df[["text", "label"]].copy()

    # Strip leading/trailing whitespace
    df["text"]  = df["text"].str.strip()
    df["label"] = df["label"].str.strip()

    logger.info(
        "Dataset loaded: %d rows, label distribution:\n%s",
        len(df),
        df["label"].value_counts().to_string(),
    )
    return df


def describe_file(path: Union[str, Path]) -> dict:
    """
    Lightweight introspection of a raw file.
    Returns a dict with: format, row_count, columns, sample_labels.
    Does NOT fully load the file — reads only the first 200 rows.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".tab"):
        sep = "\t" if suffix in (".tsv", ".tab") else ","
        sample = pd.read_csv(path, sep=sep, nrows=200, dtype=str, keep_default_na=False)
    elif suffix in (".json", ".jsonl"):
        sample = _read_json(path).head(200)
    else:
        return {"error": f"Unsupported format: {suffix}"}

    sample = _lower_cols(sample)
    label_col = _find_col(sample, LABEL_FIELD_CANDIDATES)
    sample_labels = (
        sample[label_col].value_counts().to_dict() if label_col else {}
    )

    return {
        "path": str(path),
        "format": suffix.lstrip("."),
        "columns": sample.columns.tolist(),
        "sample_rows": len(sample),
        "detected_label_col": label_col,
        "sample_label_distribution": sample_labels,
    }
