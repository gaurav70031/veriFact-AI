"""
Pydantic schemas for the dataset management API.

GET  /api/v1/datasets                   -> list[DatasetCatalogEntry]
GET  /api/v1/datasets/{name}/stats      -> DatasetStats
GET  /api/v1/datasets/{name}/analytics  -> DatasetAnalytics
GET  /api/v1/datasets/{name}/samples    -> DatasetSamples
GET  /api/v1/datasets/{name}/audit      -> PreprocessingAudit
POST /api/v1/datasets/upload            -> DatasetUploadResponse
GET  /api/v1/datasets/{name}/export     -> CSV download
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Dataset catalog entry ─────────────────────────────────────────────────────

class DatasetLicense(BaseModel):
    type:          str             # "research_only" | "cc_by" | "unknown"
    summary:       str
    commercial_ok: bool
    redistribution_ok: bool
    source_url:    str


class DatasetCatalogEntry(BaseModel):
    """Metadata about a supported / available dataset."""
    name:              str
    display_name:      str
    description:       str
    source_url:        str
    paper_url:         Optional[str]
    citation:          str
    license:           DatasetLicense
    domain:            str             # e.g. "Political news (2015–2018)"
    language:          str
    total_articles:    Optional[int]   # None if not downloaded
    fake_articles:     Optional[int]
    real_articles:     Optional[int]
    expected_files:    list[str]       # filenames to place in ml/datasets/raw/
    is_downloaded:     bool            # True if files exist on disk
    is_processed:      bool            # True if processed splits exist
    download_command:  str             # CLI command to download


class DatasetCatalogResponse(BaseModel):
    datasets: list[DatasetCatalogEntry]
    total:    int


# ── Dataset statistics ────────────────────────────────────────────────────────

class SplitStats(BaseModel):
    split:       str
    total_rows:  int
    fake_count:  int
    real_count:  int
    fake_pct:    float
    real_pct:    float
    text_length: dict   # {min, max, mean, median, p5, p95}
    token_stats: dict   # {mean_tokens, median_tokens, vocab_size_estimate}
    empty_texts: int


class DatasetStats(BaseModel):
    """Statistics from dataset_stats.json — computed by preprocessing pipeline."""
    dataset_name:    str
    total_rows:      int
    splits:          dict[str, SplitStats]
    deduplication:   dict   # {exact_removed, near_removed, total_removed}
    leakage:         dict   # {train_val_leakage, train_test_leakage}
    last_processed:  Optional[str]   # ISO timestamp from audit log


# ── Analytics ─────────────────────────────────────────────────────────────────

class WordFreqEntry(BaseModel):
    word:  str
    count: int
    pct:   float


class LengthHistogramEntry(BaseModel):
    bucket: str
    count:  int


class ClassDistribution(BaseModel):
    total:          int
    fake:           dict   # {count, pct}
    real:           dict   # {count, pct}
    balance_ratio:  Optional[float]


class MissingValueColumn(BaseModel):
    name:    str
    missing: int
    pct:     float


class DuplicateReport(BaseModel):
    exact_duplicates: int
    near_duplicates:  int
    total_duplicates: int
    pct_duplicate:    float


class TextQuality(BaseModel):
    empty_after_clean: int
    very_short:        int
    very_long:         int


class QualityReport(BaseModel):
    rows_processed:     int
    rows_raw:           Optional[int]
    rows_dropped:       Optional[int]
    pct_dropped:        Optional[float]
    missing_values:     dict
    duplicates:         DuplicateReport
    class_distribution: ClassDistribution
    text_quality:       TextQuality
    issues:             list[str]
    quality_score:      float          # 0.0–1.0


class DatasetAnalytics(BaseModel):
    """Full analytics computed from actual dataset contents."""
    dataset_name:        str
    split:               str           # "train" | "val" | "test" | "all"
    computed_at:         str           # ISO timestamp
    class_distribution:  ClassDistribution
    length_distribution: dict
    missing_values:      dict
    duplicates:          DuplicateReport
    quality_report:      QualityReport
    word_frequency: dict[str, list[WordFreqEntry]]   # {all, fake, real}


# ── Sample rows ───────────────────────────────────────────────────────────────

class DatasetSamples(BaseModel):
    dataset_name: str
    split:        str
    label_filter: Optional[int]   # 0=FAKE, 1=REAL, None=all
    total_in_split: int
    samples:      list[dict]      # each row as a dict


# ── Audit log ─────────────────────────────────────────────────────────────────

class AuditStep(BaseModel):
    step:        int
    operation:   str
    timestamp:   str
    rows_before: int
    rows_after:  int
    dropped:     int
    params:      Optional[dict]
    warnings:    Optional[list[str]]
    notes:       Optional[str]


class PreprocessingAudit(BaseModel):
    dataset_name:  str
    source_files:  list[str]
    cleaning_mode: str
    started_at:    str
    completed_at:  Optional[str]
    total_steps:   int
    total_dropped: int
    steps:         list[AuditStep]


# ── Upload ────────────────────────────────────────────────────────────────────

class DatasetUploadResponse(BaseModel):
    filename:      str
    size_bytes:    int
    rows_detected: Optional[int]
    columns:       list[str]
    detected_text_col:  Optional[str]
    detected_label_col: Optional[str]
    sample_labels:      dict[str, int]
    message:            str
    next_step:          str


# ── Preprocessing trigger ─────────────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    files:          list[str] = Field(..., description="Filenames in ml/datasets/raw/")
    cleaning_mode:  str = Field("baseline", pattern="^(baseline|transformer)$")
    lemmatize:      bool = False
    test_size:      float = Field(0.10, gt=0, lt=0.5)
    val_size:       float = Field(0.10, gt=0, lt=0.5)
    force:          bool = False


class PreprocessResponse(BaseModel):
    status:        str    # "started" | "completed" | "error"
    message:       str
    train_rows:    Optional[int]
    val_rows:      Optional[int]
    test_rows:     Optional[int]
    audit_path:    Optional[str]
