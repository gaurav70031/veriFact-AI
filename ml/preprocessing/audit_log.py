"""
ml/preprocessing/audit_log.py
==============================
Preprocessing audit log.

Records every preprocessing operation performed on a dataset with:
  - Timestamp
  - Operation name
  - Parameters used
  - Row count before and after
  - Any warnings raised

The log is written to ml/datasets/processed/preprocessing_audit.json
alongside the processed splits.

Usage
-----
    log = PreprocessingAuditLog(dataset_name="ISOT", source_files=[...])
    log.record("load",          rows_before=0,      rows_after=44898)
    log.record("label_norm",    rows_before=44898,  rows_after=44895, dropped=3)
    log.record("clean_text",    rows_before=44895,  rows_after=44890, dropped=5,
               params={"mode": "baseline"})
    log.record("deduplication", rows_before=44890,  rows_after=44800, dropped=90,
               params={"exact": 60, "near": 30})
    log.record("split",         rows_before=44800,  rows_after=44800,
               params={"train": 35840, "val": 4480, "test": 4480})
    log.save()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default output location
_DEFAULT_LOG_PATH = Path("ml/datasets/processed/preprocessing_audit.json")


class PreprocessingAuditLog:
    """
    Immutable-append audit log for a single preprocessing run.

    Each entry records one preprocessing operation and what it changed.
    Callers never delete or modify existing entries.
    """

    def __init__(
        self,
        dataset_name:  str,
        source_files:  list[str],
        cleaning_mode: str = "baseline",
    ):
        self.dataset_name  = dataset_name
        self.source_files  = [str(p) for p in source_files]
        self.cleaning_mode = cleaning_mode
        self.started_at    = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.entries:      list[dict] = []

    def record(
        self,
        operation:    str,
        rows_before:  int,
        rows_after:   int,
        dropped:      int = 0,
        params:       Optional[dict] = None,
        warnings:     Optional[list[str]] = None,
        notes:        Optional[str] = None,
    ) -> None:
        """
        Record one preprocessing step.

        Parameters
        ----------
        operation   : Human-readable step name (e.g. "deduplication").
        rows_before : Row count entering this step.
        rows_after  : Row count leaving this step.
        dropped     : Rows removed in this step (should equal rows_before - rows_after).
        params      : Step-specific parameters (e.g. {"mode": "baseline"}).
        warnings    : Any warnings raised during this step.
        notes       : Optional free-text notes.
        """
        entry = {
            "step":        len(self.entries) + 1,
            "operation":   operation,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "rows_before": rows_before,
            "rows_after":  rows_after,
            "dropped":     dropped,
            "retained":    rows_after,
        }
        if params:
            entry["params"] = params
        if warnings:
            entry["warnings"] = warnings
        if notes:
            entry["notes"] = notes

        self.entries.append(entry)

        # Validate consistency
        expected_drop = rows_before - rows_after
        if dropped != expected_drop and dropped > 0:
            logger.warning(
                "Audit log: reported dropped=%d but rows_before - rows_after = %d "
                "for step '%s'.",
                dropped, expected_drop, operation,
            )

        logger.info(
            "Preprocessing step [%d] %-20s  %d → %d rows  (dropped: %d)",
            entry["step"], operation, rows_before, rows_after, dropped,
        )

    def complete(self) -> None:
        """Mark the pipeline run as completed."""
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def total_dropped(self) -> int:
        """Total rows dropped across all preprocessing steps."""
        return sum(e.get("dropped", 0) for e in self.entries)

    def to_dict(self) -> dict:
        """Serialise the full audit log to a dict."""
        return {
            "dataset_name":  self.dataset_name,
            "source_files":  self.source_files,
            "cleaning_mode": self.cleaning_mode,
            "started_at":    self.started_at,
            "completed_at":  self.completed_at,
            "total_steps":   len(self.entries),
            "total_dropped": self.total_dropped(),
            "steps":         self.entries,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Write the audit log to a JSON file.

        Parameters
        ----------
        path : Output path.  Defaults to ml/datasets/processed/preprocessing_audit.json.

        Returns
        -------
        Path to the written file.
        """
        self.complete()
        out = Path(path) if path else _DEFAULT_LOG_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Preprocessing audit log saved → %s", out)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional[dict]:
        """
        Load a previously saved audit log.

        Returns
        -------
        dict if the file exists, None otherwise.
        """
        p = Path(path) if path else _DEFAULT_LOG_PATH
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)
