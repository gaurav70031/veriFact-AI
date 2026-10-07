"""
Label normalisation and validation.

Responsibilities
----------------
1. Map raw heterogeneous label strings to canonical integers:
       0  =  FAKE
       1  =  REAL
2. Handle multi-class → binary mapping (LIAR 6-class → binary).
3. Report and optionally drop rows with unmappable labels.
4. Validate class balance — warn if heavily imbalanced.

Label alias mapping is driven by config.py so it can be extended
without touching this module.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml.preprocessing.config import (
    LABEL2ID,
    FAKE_LABEL_ALIASES,
    REAL_LABEL_ALIASES,
)

logger = logging.getLogger(__name__)

# ── LIAR 6-class → binary mapping ────────────────────────────────────────────
# Reference: https://arxiv.org/abs/1705.00648
_LIAR_TO_BINARY: dict[str, int] = {
    "true":         LABEL2ID["REAL"],
    "mostly-true":  LABEL2ID["REAL"],
    "half-true":    LABEL2ID["REAL"],   # borderline — treated as REAL
    "barely-true":  LABEL2ID["FAKE"],   # borderline — treated as FAKE
    "false":        LABEL2ID["FAKE"],
    "pants-fire":   LABEL2ID["FAKE"],
    "pants on fire":LABEL2ID["FAKE"],
}

# ── Imbalance threshold ───────────────────────────────────────────────────────
# Warn if minority class is below this fraction of total samples
_IMBALANCE_WARN_THRESHOLD = 0.30


def _map_single_label(raw: str) -> int | None:
    """
    Map one raw label string to 0 (FAKE) or 1 (REAL).
    Returns None if the label cannot be mapped.
    """
    normalised = str(raw).strip().lower()

    # Direct integer strings
    if normalised == "0":
        return LABEL2ID["FAKE"]
    if normalised == "1":
        return LABEL2ID["REAL"]

    # LIAR 6-class
    if normalised in _LIAR_TO_BINARY:
        return _LIAR_TO_BINARY[normalised]

    # FAKE aliases
    if normalised in FAKE_LABEL_ALIASES:
        return LABEL2ID["FAKE"]

    # REAL aliases
    if normalised in REAL_LABEL_ALIASES:
        return LABEL2ID["REAL"]

    return None   # unmappable


def normalise_labels(
    df: pd.DataFrame,
    drop_unknown: bool = True,
) -> pd.DataFrame:
    """
    Normalise the 'label' column of a DataFrame to integer 0/1.

    Parameters
    ----------
    df            : DataFrame with a 'label' column (string).
    drop_unknown  : If True, rows whose label cannot be mapped are dropped
                    with a warning.  If False, a ValueError is raised instead.

    Returns
    -------
    DataFrame with 'label' column replaced by integer 0 or 1.
    """
    if "label" not in df.columns:
        raise KeyError("DataFrame has no 'label' column.")

    original_count = len(df)
    df = df.copy()

    df["label"] = df["label"].map(_map_single_label)

    unknown_mask = df["label"].isna()
    unknown_count = unknown_mask.sum()

    if unknown_count > 0:
        unknown_values = (
            df.loc[unknown_mask, "label"]
            .value_counts()
            .to_dict()
        )
        # Re-read originals for reporting (they were overwritten with NaN)
        # We can't recover them here — just report the count
        message = (
            f"{unknown_count} rows ({unknown_count/original_count:.1%}) "
            f"have unmappable labels."
        )
        if drop_unknown:
            logger.warning("%s  Dropping these rows.", message)
            df = df[~unknown_mask]
        else:
            raise ValueError(
                f"{message}  Set drop_unknown=True to remove them, "
                "or extend FAKE_LABEL_ALIASES / REAL_LABEL_ALIASES in config.py."
            )

    # Convert to int (NaN rows already dropped or errored)
    df["label"] = df["label"].astype(int)

    _check_balance(df)

    logger.info(
        "Label normalisation complete: %d rows.  "
        "FAKE=%d  REAL=%d",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    )
    return df


def _check_balance(df: pd.DataFrame) -> None:
    """Warn if the dataset is heavily imbalanced."""
    counts = df["label"].value_counts(normalize=True)
    for label_id, fraction in counts.items():
        if fraction < _IMBALANCE_WARN_THRESHOLD:
            label_name = "FAKE" if label_id == 0 else "REAL"
            logger.warning(
                "Class imbalance detected: %s represents only %.1f%% of data.  "
                "Consider using class_weight='balanced' during training.",
                label_name, fraction * 100,
            )


def get_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """
    Return a dict of {label_name: count} from a normalised DataFrame.
    Requires integer 'label' column (0/1).
    """
    counts = df["label"].value_counts().to_dict()
    return {
        "FAKE": int(counts.get(LABEL2ID["FAKE"], 0)),
        "REAL": int(counts.get(LABEL2ID["REAL"], 0)),
    }
