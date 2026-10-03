"""
Unit tests for the ml.preprocessing package.

Run from the project root (fake-news-detection/):
    pytest tests/ml/test_preprocessing.py -v

No GPU, no internet connection, and no raw dataset files are required.
All tests use in-memory synthetic data.

Coverage
--------
* cleaner.py       — baseline and transformer cleaning modes
* labels.py        — label normalisation, alias mapping, unknown handling
* deduplication.py — exact dedup, near-dedup, leakage detection
* splitter.py      — split ratios, stratification, no-overlap guarantee
* stats.py         — statistics computed from known synthetic data
* loader.py        — CSV loading, JSON loading, ISOT detection, field detection
* pipeline.py      — full pipeline integration on synthetic data
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ── Allow importing from project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.cleaner       import clean_for_baseline, clean_for_transformer, clean_series
from ml.preprocessing.labels        import normalise_labels, get_label_distribution
from ml.preprocessing.deduplication import (
    remove_exact_duplicates,
    remove_near_duplicates,
    deduplicate,
    check_leakage,
)
from ml.preprocessing.splitter      import split_dataset
from ml.preprocessing.stats         import compute_stats


# =============================================================================
# Fixtures — synthetic data helpers
# =============================================================================

def _make_df(
    texts:  list[str],
    labels: list[int | str],
) -> pd.DataFrame:
    """Build a minimal DataFrame matching pipeline expectations."""
    return pd.DataFrame({"text": texts, "label": labels})


def _make_clean_df(n_fake: int = 100, n_real: int = 100) -> pd.DataFrame:
    """
    Return a balanced DataFrame with unique, long-enough text samples.
    Used for split and stats tests that need realistic sizes.
    """
    rows = []
    for i in range(n_fake):
        rows.append({
            "text":  f"This is a fake news article number {i} about " + "word " * 30,
            "label": 0,
        })
    for i in range(n_real):
        rows.append({
            "text":  f"This is a real news article number {i} about " + "fact " * 30,
            "label": 1,
        })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# =============================================================================
# cleaner.py
# =============================================================================

class TestCleanForBaseline:
    def test_html_entities_decoded(self):
        result = clean_for_baseline("Breaking &amp; important news today here")
        assert "&amp;" not in result
        assert "breaking" in result

    def test_html_tags_removed(self):
        result = clean_for_baseline("<b>Breaking</b> <i>news</i> story today right now")
        assert "<b>" not in result
        assert "<i>" not in result
        assert "breaking" in result

    def test_urls_removed(self):
        result = clean_for_baseline(
            "Read more at https://example.com/article and visit www.news.org now"
        )
        assert "http" not in result
        assert "www" not in result

    def test_emails_removed(self):
        result = clean_for_baseline(
            "Contact reporter@example.com for details about this story today"
        )
        assert "@" not in result

    def test_lowercased(self):
        result = clean_for_baseline("BREAKING NEWS: Scientists Discover New Planet Today")
        assert result == result.lower()

    def test_contractions_expanded(self):
        result = clean_for_baseline(
            "Scientists won't confirm this story but they can't deny it now today"
        )
        assert "will not" in result
        assert "cannot" in result

    def test_non_alpha_removed(self):
        result = clean_for_baseline(
            "article 123 cost $500 today is very important for everyone"
        )
        assert not any(c.isdigit() for c in result)
        assert "$" not in result

    def test_repeated_chars_compressed(self):
        result = clean_for_baseline(
            "This is sooooo incredibly amazinggggg news today here"
        )
        # "sooooo" → max 3 repeated chars = "sooo" (3 o's)
        assert "ooooo" not in result

    def test_whitespace_collapsed(self):
        result = clean_for_baseline("Breaking   news    story    today")
        assert "  " not in result

    def test_below_min_length_returns_empty(self):
        result = clean_for_baseline("Hi")
        assert result == ""

    def test_none_input_returns_empty(self):
        result = clean_for_baseline(None)  # type: ignore[arg-type]
        assert result == ""

    def test_empty_string_returns_empty(self):
        result = clean_for_baseline("")
        assert result == ""

    def test_long_text_truncated(self):
        from ml.preprocessing.config import MAX_TEXT_LENGTH
        long_text = "word " * 5000    # well over MAX_TEXT_LENGTH characters
        result = clean_for_baseline(long_text)
        assert len(result) <= MAX_TEXT_LENGTH

    def test_reuters_dateline_stripped(self):
        result = clean_for_baseline(
            "(REUTERS) - Oil prices rose on Thursday as traders watched the market today"
        )
        assert "reuters" not in result

    def test_smart_quotes_normalised(self):
        result = clean_for_baseline(
            "\u201cBreaking news\u201d is very important story for everyone today"
        )
        assert "\u201c" not in result
        assert "\u201d" not in result


class TestCleanForTransformer:
    def test_html_stripped(self):
        result = clean_for_transformer("<p>Breaking news today in the world</p>")
        assert "<p>" not in result
        assert "Breaking news today" in result

    def test_preserves_casing(self):
        original = "Breaking NEWS: Scientists Discover Important Planet"
        result = clean_for_transformer(original)
        # Some uppercase letters should survive
        assert any(c.isupper() for c in result)

    def test_preserves_punctuation(self):
        result = clean_for_transformer(
            "Scientists said: 'This is real.' But is it true?"
        )
        assert "." in result or "?" in result

    def test_urls_removed(self):
        result = clean_for_transformer("Visit https://example.com for full story today")
        assert "https" not in result


class TestCleanSeries:
    def test_returns_series(self):
        s = pd.Series(["Breaking news today here", "Another story is happening now"])
        result = clean_series(s, mode="baseline")
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_empty_becomes_empty_string(self):
        s = pd.Series(["ok", ""])
        result = clean_series(s, mode="baseline")
        assert result.iloc[1] == ""

    def test_transformer_mode(self):
        s = pd.Series(["<b>Breaking News Today Here</b>"])
        result = clean_series(s, mode="transformer")
        assert "<b>" not in result.iloc[0]
        assert "Breaking" in result.iloc[0]


# =============================================================================
# labels.py
# =============================================================================

class TestNormaliseLabels:
    def test_fake_string(self):
        df = _make_df(["text one two three words here now"] * 3, ["fake", "FAKE", "0"])
        result = normalise_labels(df)
        assert (result["label"] == 0).all()

    def test_real_string(self):
        df = _make_df(["text one two three words here now"] * 3, ["real", "REAL", "1"])
        result = normalise_labels(df)
        assert (result["label"] == 1).all()

    def test_integer_strings(self):
        df = _make_df(["text one here now"] * 2, ["0", "1"])
        result = normalise_labels(df)
        assert list(result["label"]) == [0, 1]

    def test_liar_labels_mapped(self):
        df = _make_df(
            ["text one two three words"] * 6,
            ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"],
        )
        result = normalise_labels(df)
        expected = [1, 1, 1, 0, 0, 0]
        assert list(result["label"]) == expected

    def test_unknown_labels_dropped(self):
        df = _make_df(
            ["text one" * 5, "text two" * 5, "text three" * 5],
            ["fake", "unknown_label_xyz", "real"],
        )
        result = normalise_labels(df, drop_unknown=True)
        assert len(result) == 2

    def test_unknown_labels_raise_when_not_dropping(self):
        df = _make_df(["text one here" * 3], ["totally_unknown"])
        with pytest.raises(ValueError, match="unmappable"):
            normalise_labels(df, drop_unknown=False)

    def test_missing_label_column_raises(self):
        df = pd.DataFrame({"text": ["something here"]})
        with pytest.raises(KeyError, match="label"):
            normalise_labels(df)

    def test_output_dtype_is_int(self):
        df = _make_df(["text one two three four"] * 2, ["fake", "real"])
        result = normalise_labels(df)
        assert result["label"].dtype in (int, "int64", "int32")

    def test_mixed_case_aliases(self):
        df = _make_df(
            ["text one two three"] * 4,
            ["FAKE", "fake", "Fake", "fAkE"],
        )
        result = normalise_labels(df)
        assert (result["label"] == 0).all()


class TestGetLabelDistribution:
    def test_balanced(self):
        df = _make_df(
            ["text one two three four"] * 4,
            [0, 0, 1, 1],
        )
        dist = get_label_distribution(df)
        assert dist["FAKE"] == 2
        assert dist["REAL"] == 2

    def test_all_fake(self):
        df = _make_df(["text one two three four"] * 3, [0, 0, 0])
        dist = get_label_distribution(df)
        assert dist["FAKE"] == 3
        assert dist["REAL"] == 0


# =============================================================================
# deduplication.py
# =============================================================================

class TestExactDedup:
    def test_removes_identical_rows(self):
        df = _make_df(
            ["unique text here one", "duplicate text here two", "duplicate text here two"],
            [0, 1, 1],
        )
        result, removed = remove_exact_duplicates(df)
        assert removed == 1
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = _make_df(["text alpha here", "text beta here", "text gamma here"], [0, 1, 0])
        result, removed = remove_exact_duplicates(df)
        assert removed == 0
        assert len(result) == 3

    def test_keeps_first_occurrence(self):
        df = _make_df(["same text here now", "same text here now"], [0, 1])
        result, _ = remove_exact_duplicates(df)
        assert len(result) == 1
        assert result.iloc[0]["label"] == 0   # first occurrence kept


class TestNearDedup:
    def test_removes_case_variants(self):
        df = _make_df(
            ["Breaking News Today Here", "breaking news today here"],
            [0, 1],
        )
        result, removed = remove_near_duplicates(df)
        assert removed == 1
        assert len(result) == 1

    def test_removes_whitespace_variants(self):
        df = _make_df(
            ["text   with   extra    spaces", "text with extra spaces"],
            [0, 0],
        )
        result, removed = remove_near_duplicates(df)
        assert removed == 1

    def test_distinct_texts_kept(self):
        df = _make_df(
            ["completely different article", "totally unrelated story here today"],
            [0, 1],
        )
        result, removed = remove_near_duplicates(df)
        assert removed == 0
        assert len(result) == 2


class TestDeduplicateFull:
    def test_combined_report(self):
        df = _make_df(
            [
                "unique story number one here",
                "duplicate story number two",
                "duplicate story number two",    # exact dup
                "DUPLICATE STORY NUMBER TWO",    # near dup of above
            ],
            [0, 1, 1, 1],
        )
        result, report = deduplicate(df)
        assert report["exact_removed"] >= 1
        assert report["total_removed"] >= 2
        assert "near_removed" in report


class TestLeakageDetection:
    def test_no_leakage(self):
        train = _make_df(["train text alpha here", "train text beta here"], [0, 1])
        val   = _make_df(["val text gamma here",   "val text delta here"],  [0, 1])
        test  = _make_df(["test text epsilon here", "test text zeta here"], [0, 1])
        report = check_leakage(train, val, test)
        assert report["train_val_leakage"]  == 0
        assert report["train_test_leakage"] == 0

    def test_detects_val_leakage(self):
        shared = "this text is in both train and val today"
        train = _make_df([shared, "other train text here now today"], [0, 1])
        val   = _make_df([shared, "unique val text here now today"],  [0, 1])
        test  = _make_df(["unique test text here now today"],         [1])
        report = check_leakage(train, val, test)
        assert report["train_val_leakage"] >= 1

    def test_detects_test_leakage(self):
        shared = "leaked text appears in both train and test sets now"
        train = _make_df([shared, "other text here today now words"], [0, 1])
        val   = _make_df(["unique validation text only here today"],  [0])
        test  = _make_df([shared, "different test text here today now"], [0, 1])
        report = check_leakage(train, val, test)
        assert report["train_test_leakage"] >= 1


# =============================================================================
# splitter.py
# =============================================================================

class TestSplitDataset:
    def setup_method(self):
        self.df = _make_clean_df(n_fake=100, n_real=100)

    def test_three_splits_returned(self):
        train, val, test = split_dataset(self.df)
        assert len(train) > 0
        assert len(val)   > 0
        assert len(test)  > 0

    def test_total_rows_preserved(self):
        train, val, test = split_dataset(self.df)
        assert len(train) + len(val) + len(test) == len(self.df)

    def test_default_ratios_approximate(self):
        train, val, test = split_dataset(self.df)
        total = len(self.df)
        # Test  ≈ 10%  → between 8–12%
        assert 0.08 <= len(test)  / total <= 0.12
        # Val   ≈ 10%  → between 8–12%
        assert 0.08 <= len(val)   / total <= 0.12
        # Train ≈ 80%  → at least 75%
        assert len(train) / total >= 0.75

    def test_no_overlap(self):
        train, val, test = split_dataset(self.df)
        train_texts = set(train["text"])
        val_texts   = set(val["text"])
        test_texts  = set(test["text"])
        assert len(train_texts & val_texts)  == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts   & test_texts) == 0

    def test_stratified_labels(self):
        train, val, test = split_dataset(self.df)
        for split_df in (train, val, test):
            fake_pct = (split_df["label"] == 0).mean()
            real_pct = (split_df["label"] == 1).mean()
            # Each split should be close to 50/50 (original balance)
            assert 0.40 <= fake_pct <= 0.60
            assert 0.40 <= real_pct <= 0.60

    def test_reproducible_with_same_seed(self):
        train1, val1, test1 = split_dataset(self.df, random_seed=99)
        train2, val2, test2 = split_dataset(self.df, random_seed=99)
        assert list(train1["text"]) == list(train2["text"])

    def test_different_seeds_differ(self):
        train1, _, _ = split_dataset(self.df, random_seed=1)
        train2, _, _ = split_dataset(self.df, random_seed=2)
        assert list(train1["text"]) != list(train2["text"])

    def test_custom_split_sizes(self):
        train, val, test = split_dataset(self.df, test_size=0.20, val_size=0.20)
        total = len(self.df)
        assert 0.17 <= len(test) / total <= 0.23
        assert 0.17 <= len(val)  / total <= 0.23

    def test_too_small_raises(self):
        tiny = _make_df(["text"] * 5, [0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="too small"):
            split_dataset(tiny)


# =============================================================================
# stats.py
# =============================================================================

class TestComputeStats:
    def setup_method(self):
        df = _make_clean_df(n_fake=80, n_real=80)
        self.train, self.val, self.test = split_dataset(df)

    def test_returns_dict(self):
        stats = compute_stats(self.train, self.val, self.test)
        assert isinstance(stats, dict)

    def test_total_rows_correct(self):
        stats = compute_stats(self.train, self.val, self.test)
        total = len(self.train) + len(self.val) + len(self.test)
        assert stats["total_rows"] == total

    def test_split_keys_present(self):
        stats = compute_stats(self.train, self.val, self.test)
        assert "train" in stats["splits"]
        assert "val"   in stats["splits"]
        assert "test"  in stats["splits"]

    def test_fake_real_counts_sum_to_total(self):
        stats = compute_stats(self.train, self.val, self.test)
        for split_name, split_df in [
            ("train", self.train), ("val", self.val), ("test", self.test)
        ]:
            s = stats["splits"][split_name]
            assert s["fake_count"] + s["real_count"] == len(split_df)

    def test_dedup_report_included(self):
        dedup_report = {"exact_removed": 3, "near_removed": 1, "total_removed": 4}
        stats = compute_stats(
            self.train, self.val, self.test, dedup_report=dedup_report
        )
        assert stats["deduplication"]["total_removed"] == 4

    def test_leakage_report_included(self):
        leakage_report = {"train_val_leakage": 0, "train_test_leakage": 0}
        stats = compute_stats(
            self.train, self.val, self.test, leakage_report=leakage_report
        )
        assert stats["leakage"]["train_val_leakage"] == 0

    def test_text_length_stats_present(self):
        stats = compute_stats(self.train, self.val, self.test)
        tl = stats["splits"]["train"]["text_length"]
        assert "min" in tl and "max" in tl and "mean" in tl

    def test_no_fabricated_values(self):
        """
        Verify stats actually reflect the data — not hard-coded.
        Change the input and stats must change.
        """
        df_small = _make_clean_df(n_fake=10, n_real=10)
        t1, v1, te1 = split_dataset(df_small)
        stats_small = compute_stats(t1, v1, te1)

        df_large = _make_clean_df(n_fake=80, n_real=80)
        t2, v2, te2 = split_dataset(df_large)
        stats_large = compute_stats(t2, v2, te2)

        assert stats_small["total_rows"] < stats_large["total_rows"]


# =============================================================================
# loader.py — in-memory CSV and JSON loading
# =============================================================================

class TestLoader:
    def _write_temp_csv(self, content: str, suffix: str = ".csv") -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.flush()
        return Path(tmp.name)

    def test_load_generic_csv(self):
        from ml.preprocessing.loader import load_dataset
        content = "text,label\n"
        for i in range(10):
            label = "fake" if i % 2 == 0 else "real"
            content += f"This is article number {i} about the world,{label}\n"
        path = self._write_temp_csv(content)
        df = load_dataset(path)
        assert "text"  in df.columns
        assert "label" in df.columns
        assert len(df) == 10

    def test_load_csv_with_title_column(self):
        from ml.preprocessing.loader import load_dataset
        content = "title,text,label\n"
        for i in range(6):
            label = "fake" if i % 2 == 0 else "real"
            content += f"Headline {i},Body text of article number {i},{ label}\n"
        path = self._write_temp_csv(content)
        df = load_dataset(path)
        # Title should be prepended to text
        assert df["text"].iloc[0].startswith("Headline")

    def test_load_json_array(self):
        from ml.preprocessing.loader import load_dataset
        records = [
            {"text": f"Article text content number {i} for testing", "label": "fake" if i % 2 == 0 else "real"}
            for i in range(8)
        ]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(records, tmp)
        tmp.flush()
        path = Path(tmp.name)
        df = load_dataset(path)
        assert len(df) == 8
        assert "text" in df.columns

    def test_load_jsonl(self):
        from ml.preprocessing.loader import load_dataset
        lines = "\n".join(
            json.dumps({"text": f"News article {i} text content here", "label": "real"})
            for i in range(5)
        )
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        tmp.write(lines)
        tmp.flush()
        df = load_dataset(Path(tmp.name))
        assert len(df) == 5

    def test_missing_file_raises(self):
        from ml.preprocessing.loader import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/path/file.csv"))

    def test_missing_text_column_raises(self):
        from ml.preprocessing.loader import load_dataset
        content = "headline,label\nSome headline text here,fake\n"
        path = self._write_temp_csv(content)
        with pytest.raises(ValueError, match="text column"):
            load_dataset(path)

    def test_missing_label_column_raises(self):
        from ml.preprocessing.loader import load_dataset
        # Use a column name that has no match in LABEL_FIELD_CANDIDATES
        content = "text,source_outlet\nSome long article text here,newspaper\n"
        path = self._write_temp_csv(content)
        with pytest.raises(ValueError, match="label column"):
            load_dataset(path)

    def test_unsupported_format_raises(self):
        from ml.preprocessing.loader import load_dataset
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        )
        tmp.write("<data/>")
        tmp.flush()
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(Path(tmp.name))

    def test_tsv_loading(self):
        from ml.preprocessing.loader import load_dataset
        content = "text\tlabel\n"
        for i in range(5):
            content += f"Article text number {i}\t{'fake' if i%2==0 else 'real'}\n"
        path = self._write_temp_csv(content, suffix=".tsv")
        df = load_dataset(path)
        assert len(df) == 5


# =============================================================================
# pipeline.py — integration test on synthetic data
# =============================================================================

class TestPipelineIntegration:
    def test_full_pipeline_on_synthetic_csv(self, tmp_path, monkeypatch):
        """
        End-to-end pipeline test using a synthetic CSV file.
        Writes a temporary CSV, runs run_pipeline(), and verifies outputs.
        No real dataset required.

        Uses monkeypatch to redirect all output path references — both in
        config.py and in pipeline.py (which imports them at module load time).
        """
        import ml.preprocessing.config   as cfg
        import ml.preprocessing.pipeline as pipeline_mod
        from ml.preprocessing.pipeline import run_pipeline

        train_path = tmp_path / "train.csv"
        val_path   = tmp_path / "val.csv"
        test_path  = tmp_path / "test.csv"
        stats_path = tmp_path / "stats.json"

        # Patch config module attributes
        monkeypatch.setattr(cfg, "TRAIN_CSV",  train_path)
        monkeypatch.setattr(cfg, "VAL_CSV",    val_path)
        monkeypatch.setattr(cfg, "TEST_CSV",   test_path)
        monkeypatch.setattr(cfg, "STATS_JSON", stats_path)

        # Patch pipeline module's own imported names
        monkeypatch.setattr(pipeline_mod, "TRAIN_CSV",  train_path)
        monkeypatch.setattr(pipeline_mod, "VAL_CSV",    val_path)
        monkeypatch.setattr(pipeline_mod, "TEST_CSV",   test_path)
        monkeypatch.setattr(pipeline_mod, "STATS_JSON", stats_path)

        # Build a synthetic CSV with 200 unique samples
        rows = []
        for i in range(100):
            rows.append({
                "text": f"fake article number {i} contains fabricated story about " + f"topic{i} " * 10,
                "label": "fake",
            })
        for i in range(100):
            rows.append({
                "text": f"real article number {i} reports verified facts about " + f"event{i} " * 10,
                "label": "real",
            })

        csv_path = tmp_path / "synthetic.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        train_df, val_df, test_df, stats = run_pipeline(
            paths=[csv_path],
            force=True,
            verbose=False,
        )

        # All three splits returned with rows
        assert len(train_df) > 0
        assert len(val_df)   > 0
        assert len(test_df)  > 0

        # Total rows preserved (may be slightly less due to dedup/empty-text drop)
        assert len(train_df) + len(val_df) + len(test_df) <= 200

        # Output CSV files written to tmp_path
        assert train_path.exists(), "train.csv not written"
        assert val_path.exists(),   "val.csv not written"
        assert test_path.exists(),  "test.csv not written"
        assert stats_path.exists(), "dataset_stats.json not written"

        # Stats reflect actual data (not hard-coded)
        assert stats["total_rows"] == len(train_df) + len(val_df) + len(test_df)

        # No cross-split overlap
        train_texts = set(train_df["text"])
        assert len(train_texts & set(val_df["text"]))  == 0
        assert len(train_texts & set(test_df["text"])) == 0
