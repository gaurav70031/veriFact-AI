"""
Preprocessing configuration.

Single source of truth for every path, constant, and tunable parameter
used across the preprocessing pipeline.  Import from here — never
hard-code paths or magic numbers in individual modules.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ML_ROOT       = Path(__file__).resolve().parents[1]   # ml/
DATASETS_DIR  = ML_ROOT / "datasets"
RAW_DIR       = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"

# Ensure output directories exist when config is imported
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Processed output filenames ────────────────────────────────────────────────
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV   = PROCESSED_DIR / "val.csv"
TEST_CSV  = PROCESSED_DIR / "test.csv"
STATS_JSON = PROCESSED_DIR / "dataset_stats.json"

# ── Label mapping ─────────────────────────────────────────────────────────────
# All models use integer labels: 0 = FAKE, 1 = REAL
LABEL2ID: dict[str, int] = {"FAKE": 0, "REAL": 1}
ID2LABEL: dict[int, str] = {0: "FAKE", 1: "REAL"}

# Accepted raw label strings that map to FAKE (case-insensitive)
FAKE_LABEL_ALIASES: set[str] = {
    "fake", "0", "false", "misinformation", "satire",
    "conspiracy", "unreliable", "bs", "hate", "junksci",
}
# Accepted raw label strings that map to REAL (case-insensitive)
REAL_LABEL_ALIASES: set[str] = {
    "real", "1", "true", "reliable", "credible",
    "mostly-true", "half-true",
}

# ── Text field candidates ─────────────────────────────────────────────────────
# Loader will search column names for these patterns (case-insensitive)
TEXT_FIELD_CANDIDATES:  list[str] = ["text", "body", "content", "article", "statement"]
TITLE_FIELD_CANDIDATES: list[str] = ["title", "headline", "subject", "head"]
LABEL_FIELD_CANDIDATES: list[str] = ["label", "class", "target", "fake", "real", "category"]

# ── Text cleaning ─────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH: int   = 20       # discard texts shorter than this (characters)
MAX_TEXT_LENGTH: int   = 10_000   # truncate texts longer than this
MAX_REPEATED_CHARS: int = 3       # compress runs: "loooool" → "lool"

# ── Deduplication ─────────────────────────────────────────────────────────────
# Exact deduplication is always applied.
# Near-duplicate detection uses hash of normalised text (lowercased, whitespace-collapsed).
# Leakage detection: a train sample whose normalised text appears in test/val is flagged.
DEDUP_NORMALISE: bool = True     # normalise before exact dedup hash

# ── Dataset splitting ─────────────────────────────────────────────────────────
RANDOM_SEED: int  = 42
TEST_SIZE:  float = 0.10    # 10 % held-out test set
VAL_SIZE:   float = 0.10    # 10 % validation (from remaining 90 %)
