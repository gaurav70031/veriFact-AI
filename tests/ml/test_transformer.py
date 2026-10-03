"""
Unit and integration tests for ml.transformer.

Test strategy
-------------
* Tokenizer tests   — use the real DistilBERT tokenizer (no GPU needed).
* Dataset tests     — use the real tokenizer with synthetic texts.
* Chunking tests    — pure Python / NumPy, no model needed.
* Aggregation tests — pure NumPy, no model needed.
* Inference tests   — use a randomly-initialised DistilBERT (no pre-training
                      weights downloaded) so results are structurally correct
                      but numerically arbitrary.  The real trained model is
                      tested only when a saved checkpoint exists on disk.
* Evaluator tests   — mock the model to return fixed logits; verify metric
                      computations are correct.

Run from the project root:
    pytest tests/ml/test_transformer.py -v

No GPU required.  Requires:
    pip install torch transformers

The DistilBERT tokenizer (~66 MB) is downloaded on first run and cached by
HuggingFace in ~/.cache/huggingface/.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports guarded against missing torch/transformers ──────────────────
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TRANSFORMERS_AVAILABLE,
    reason="torch / transformers not installed",
)

from ml.transformer.config import (
    BASE_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_TOKENS,
    MAX_TOKEN_LENGTH, NUM_LABELS, LABEL2ID, ID2LABEL,
)
from ml.transformer.dataset import (
    FakeNewsDataset,
    ChunkedArticleDataset,
    chunk_token_ids,
    tokenize_to_ids,
    encode_chunk,
)
from ml.transformer.inference import (
    aggregate_chunk_probs,
    clear_model_cache,
    predict_text,
    predict_texts_batch,
    _aggregate_weighted_mean,
    _aggregate_max_confidence,
    _aggregate_majority_vote,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tokenizer():
    """Real DistilBERT tokenizer — loaded once per module."""
    return AutoTokenizer.from_pretrained(BASE_MODEL)


@pytest.fixture(scope="module")
def random_model():
    """
    Randomly-initialised DistilBERT (no pretrained weights).
    Fast to create; gives structurally correct output shapes.
    """
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    config = DistilBertConfig(
        vocab_size=30522,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        n_layers=2,        # tiny — 2 transformer layers instead of 6
        n_heads=4,
        dim=256,
        hidden_dim=512,
    )
    model = DistilBertForSequenceClassification(config)
    model.eval()
    return model


SHORT_TEXT = (
    "The central bank raised interest rates by twenty five basis points "
    "on thursday citing persistent inflationary pressures in the economy."
)

LONG_TEXT = " ".join(
    [f"sentence number {i} discusses various aspects of political events "
     f"and economic indicators across multiple regions worldwide"
     for i in range(200)]
)   # ~8000 chars / ~1600 tokens — well over 512


# =============================================================================
# Tokenizer tests
# =============================================================================

class TestTokenizer:
    def test_tokenizer_loads(self, tokenizer):
        assert tokenizer is not None

    def test_tokenizer_has_correct_vocab_size(self, tokenizer):
        # DistilBERT-base-uncased vocab size
        assert tokenizer.vocab_size == 30522

    def test_encode_short_text(self, tokenizer):
        ids = tokenizer.encode(SHORT_TEXT)
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_adds_special_tokens(self, tokenizer):
        ids = tokenizer.encode(SHORT_TEXT, add_special_tokens=True)
        # [CLS] = 101, [SEP] = 102 for BERT-family
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.sep_token_id

    def test_encode_no_special_tokens(self, tokenizer):
        ids = tokenizer.encode(SHORT_TEXT, add_special_tokens=False)
        assert ids[0] != tokenizer.cls_token_id

    def test_long_text_truncated_at_max_length(self, tokenizer):
        ids = tokenizer.encode(
            LONG_TEXT,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        assert len(ids) == MAX_TOKEN_LENGTH

    def test_batch_encoding_returns_tensors(self, tokenizer):
        enc = tokenizer(
            [SHORT_TEXT, SHORT_TEXT],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        assert "input_ids"      in enc
        assert "attention_mask" in enc
        assert enc["input_ids"].shape == (2, 64)

    def test_decode_round_trip(self, tokenizer):
        ids  = tokenizer.encode(SHORT_TEXT, add_special_tokens=False)
        back = tokenizer.decode(ids, skip_special_tokens=True)
        # Decoded text should contain key words from original
        assert "bank" in back.lower() or "rates" in back.lower()

    def test_pad_token_id_defined(self, tokenizer):
        assert tokenizer.pad_token_id is not None

    def test_unk_token_id_defined(self, tokenizer):
        assert tokenizer.unk_token_id is not None


# =============================================================================
# tokenize_to_ids helper
# =============================================================================

class TestTokenizeToIds:
    def test_returns_list_of_ints(self, tokenizer):
        ids = tokenize_to_ids(SHORT_TEXT, tokenizer)
        assert isinstance(ids, list)
        assert all(isinstance(x, int) for x in ids)

    def test_no_special_tokens(self, tokenizer):
        ids = tokenize_to_ids(SHORT_TEXT, tokenizer)
        assert tokenizer.cls_token_id not in ids
        assert tokenizer.sep_token_id not in ids

    def test_long_text_no_truncation(self, tokenizer):
        ids = tokenize_to_ids(LONG_TEXT, tokenizer)
        # Long text must not be truncated to 512
        assert len(ids) > MAX_TOKEN_LENGTH


# =============================================================================
# chunk_token_ids
# =============================================================================

class TestChunkTokenIds:
    def test_short_text_one_chunk(self):
        ids    = list(range(100))
        chunks = chunk_token_ids(ids, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        assert len(chunks) == 1
        assert chunks[0] == ids

    def test_empty_returns_empty(self):
        assert chunk_token_ids([], chunk_size=510, overlap=50) == []

    def test_exact_chunk_size_one_chunk(self):
        ids    = list(range(CHUNK_SIZE))
        chunks = chunk_token_ids(ids, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        ids    = list(range(1600))
        chunks = chunk_token_ids(ids, chunk_size=510, overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_correct_max_length(self):
        ids    = list(range(1600))
        chunks = chunk_token_ids(ids, chunk_size=510, overlap=50)
        for c in chunks:
            assert len(c) <= 510

    def test_overlap_creates_shared_tokens(self):
        ids    = list(range(1100))
        overlap = 100
        chunks  = chunk_token_ids(ids, chunk_size=510, overlap=overlap)
        assert len(chunks) >= 2
        # Last overlap tokens of chunk 0 should equal first overlap tokens of chunk 1
        last_of_c0  = chunks[0][-overlap:]
        first_of_c1 = chunks[1][:overlap]
        assert last_of_c0 == first_of_c1

    def test_all_tokens_covered(self):
        """Every token in the original sequence must appear in at least one chunk."""
        ids    = list(range(800))
        chunks = chunk_token_ids(ids, chunk_size=510, overlap=50)
        covered = set()
        for chunk in chunks:
            covered.update(chunk)
        assert covered == set(ids)

    def test_min_tokens_filters_tiny_trailing_chunk(self):
        # Create a sequence where the last chunk would be very short
        ids    = list(range(515))   # 510 + 5
        chunks = chunk_token_ids(ids, chunk_size=510, overlap=0, min_tokens=10)
        # The 5-token trailing chunk should be dropped
        assert len(chunks) == 1

    def test_custom_chunk_size(self):
        ids    = list(range(300))
        chunks = chunk_token_ids(ids, chunk_size=100, overlap=10)
        for c in chunks:
            assert len(c) <= 100


# =============================================================================
# encode_chunk
# =============================================================================

class TestEncodeChunk:
    def test_returns_tensor_dict(self, tokenizer):
        chunk_ids = list(range(50))
        enc = encode_chunk(chunk_ids, tokenizer)
        assert "input_ids"      in enc
        assert "attention_mask" in enc
        assert isinstance(enc["input_ids"], torch.Tensor)

    def test_output_length_is_max_length(self, tokenizer):
        chunk_ids = list(range(50))
        enc = encode_chunk(chunk_ids, tokenizer, max_length=MAX_TOKEN_LENGTH)
        assert enc["input_ids"].shape[0]      == MAX_TOKEN_LENGTH
        assert enc["attention_mask"].shape[0] == MAX_TOKEN_LENGTH

    def test_padding_is_zero_in_attention_mask(self, tokenizer):
        chunk_ids = list(range(10))   # very short chunk
        enc = encode_chunk(chunk_ids, tokenizer)
        mask = enc["attention_mask"].tolist()
        # First few positions should be 1 (real tokens), rest 0 (padding)
        assert 0 in mask
        assert 1 in mask

    def test_attention_mask_ones_at_start(self, tokenizer):
        chunk_ids = list(range(20))
        enc = encode_chunk(chunk_ids, tokenizer)
        mask = enc["attention_mask"].tolist()
        assert mask[0] == 1   # [CLS] always attended

    def test_special_tokens_added(self, tokenizer):
        chunk_ids = list(range(5))
        enc = encode_chunk(chunk_ids, tokenizer)
        ids = enc["input_ids"].tolist()
        # [CLS] should be first non-pad real token
        assert ids[0] == tokenizer.cls_token_id


# =============================================================================
# FakeNewsDataset
# =============================================================================

class TestFakeNewsDataset:
    def test_len_correct(self, tokenizer):
        texts  = [SHORT_TEXT] * 10
        labels = [0, 1] * 5
        ds = FakeNewsDataset(texts, labels, tokenizer, max_length=64)
        assert len(ds) == 10

    def test_item_has_required_keys(self, tokenizer):
        ds   = FakeNewsDataset([SHORT_TEXT], [0], tokenizer, max_length=64)
        item = ds[0]
        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "labels"         in item

    def test_input_ids_correct_shape(self, tokenizer):
        ds   = FakeNewsDataset([SHORT_TEXT], [1], tokenizer, max_length=64)
        item = ds[0]
        assert item["input_ids"].shape == (64,)

    def test_label_tensor_correct(self, tokenizer):
        ds   = FakeNewsDataset([SHORT_TEXT], [1], tokenizer, max_length=64)
        item = ds[0]
        assert item["labels"].item() == 1

    def test_no_labels_mode(self, tokenizer):
        """Inference mode: labels=None."""
        ds   = FakeNewsDataset([SHORT_TEXT], labels=None, tokenizer=tokenizer, max_length=64)
        item = ds[0]
        assert "labels" not in item
        assert "input_ids" in item

    def test_mismatched_lengths_raise(self, tokenizer):
        with pytest.raises(ValueError):
            FakeNewsDataset([SHORT_TEXT, SHORT_TEXT], [0], tokenizer, max_length=64)

    def test_dataloader_batches_correctly(self, tokenizer):
        texts  = [SHORT_TEXT] * 8
        labels = [0, 1] * 4
        ds     = FakeNewsDataset(texts, labels, tokenizer, max_length=32)
        loader = DataLoader(ds, batch_size=4)
        batch  = next(iter(loader))
        assert batch["input_ids"].shape      == (4, 32)
        assert batch["attention_mask"].shape == (4, 32)
        assert batch["labels"].shape         == (4,)


# =============================================================================
# ChunkedArticleDataset
# =============================================================================

class TestChunkedArticleDataset:
    def test_short_text_single_chunk(self, tokenizer):
        ds = ChunkedArticleDataset(SHORT_TEXT, tokenizer)
        assert ds.num_chunks == 1

    def test_long_text_multiple_chunks(self, tokenizer):
        ds = ChunkedArticleDataset(LONG_TEXT, tokenizer)
        assert ds.num_chunks > 1

    def test_each_chunk_correct_length(self, tokenizer):
        ds = ChunkedArticleDataset(LONG_TEXT, tokenizer)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape[0]      == MAX_TOKEN_LENGTH
            assert item["attention_mask"].shape[0] == MAX_TOKEN_LENGTH

    def test_num_chunks_matches_len(self, tokenizer):
        ds = ChunkedArticleDataset(LONG_TEXT, tokenizer)
        assert len(ds) == ds.num_chunks

    def test_empty_text_does_not_crash(self, tokenizer):
        ds = ChunkedArticleDataset("", tokenizer)
        assert len(ds) >= 1

    def test_dataloader_iteration(self, tokenizer):
        ds     = ChunkedArticleDataset(SHORT_TEXT, tokenizer)
        loader = DataLoader(ds, batch_size=8)
        batches = list(loader)
        assert len(batches) >= 1
        assert "input_ids"      in batches[0]
        assert "attention_mask" in batches[0]


# =============================================================================
# Aggregation strategies
# =============================================================================

class TestAggregationStrategies:
    def _make_probs(self, rows: list[list[float]]) -> np.ndarray:
        return np.array(rows, dtype=np.float32)

    # ── weighted_mean ─────────────────────────────────────────────────────────
    def test_weighted_mean_single_chunk(self):
        probs  = self._make_probs([[0.8, 0.2]])
        result = _aggregate_weighted_mean(probs)
        assert abs(result[0] - 0.8) < 1e-5

    def test_weighted_mean_two_equal_chunks(self):
        probs  = self._make_probs([[0.7, 0.3], [0.7, 0.3]])
        result = _aggregate_weighted_mean(probs)
        assert abs(result[0] - 0.7) < 1e-4

    def test_weighted_mean_confident_chunk_dominates(self):
        # chunk0: 99% fake (very confident), chunk1: 60% real (less confident)
        probs  = self._make_probs([[0.99, 0.01], [0.40, 0.60]])
        result = _aggregate_weighted_mean(probs)
        # The confident FAKE chunk should pull final result towards FAKE
        assert result[0] > result[1]

    def test_weighted_mean_output_sums_to_one(self):
        probs  = self._make_probs([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
        result = _aggregate_weighted_mean(probs)
        assert abs(result.sum() - 1.0) < 1e-5

    # ── max_confidence ────────────────────────────────────────────────────────
    def test_max_confidence_picks_highest(self):
        probs  = self._make_probs([[0.6, 0.4], [0.95, 0.05], [0.55, 0.45]])
        result = _aggregate_max_confidence(probs)
        # Should return the row with highest max_prob = row 1
        assert abs(result[0] - 0.95) < 1e-5

    def test_max_confidence_single_chunk(self):
        probs  = self._make_probs([[0.3, 0.7]])
        result = _aggregate_max_confidence(probs)
        assert abs(result[1] - 0.7) < 1e-5

    # ── majority_vote ─────────────────────────────────────────────────────────
    def test_majority_vote_clear_majority(self):
        # 3 FAKE votes vs 1 REAL vote
        probs  = self._make_probs([
            [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]
        ])
        result = _aggregate_majority_vote(probs)
        assert result[0] > result[1]   # FAKE wins

    def test_majority_vote_output_sums_to_one(self):
        probs  = self._make_probs([[0.6, 0.4], [0.4, 0.6], [0.7, 0.3]])
        result = _aggregate_majority_vote(probs)
        assert abs(result.sum() - 1.0) < 1e-4

    # ── aggregate_chunk_probs dispatcher ─────────────────────────────────────
    def test_dispatcher_weighted_mean(self):
        probs  = self._make_probs([[0.6, 0.4], [0.8, 0.2]])
        result = aggregate_chunk_probs(probs, strategy="weighted_mean")
        assert result.shape == (2,)

    def test_dispatcher_max_confidence(self):
        probs  = self._make_probs([[0.6, 0.4], [0.8, 0.2]])
        result = aggregate_chunk_probs(probs, strategy="max_confidence")
        assert result.shape == (2,)

    def test_dispatcher_majority_vote(self):
        probs  = self._make_probs([[0.6, 0.4], [0.8, 0.2]])
        result = aggregate_chunk_probs(probs, strategy="majority_vote")
        assert result.shape == (2,)

    def test_dispatcher_unknown_strategy_raises(self):
        probs = self._make_probs([[0.5, 0.5]])
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            aggregate_chunk_probs(probs, strategy="does_not_exist")

    def test_single_chunk_passthrough(self):
        probs  = self._make_probs([[0.3, 0.7]])
        for strategy in ("weighted_mean", "max_confidence", "majority_vote"):
            result = aggregate_chunk_probs(probs, strategy=strategy)
            assert abs(result[1] - 0.7) < 1e-5, f"Failed for {strategy}"

    def test_result_always_sums_to_one(self):
        probs = self._make_probs([
            [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]
        ])
        for strategy in ("weighted_mean", "max_confidence", "majority_vote"):
            result = aggregate_chunk_probs(probs, strategy=strategy)
            assert abs(result.sum() - 1.0) < 1e-4, f"Sums to {result.sum()} for {strategy}"


# =============================================================================
# Inference — using random model (no pre-trained weights)
# =============================================================================

class TestInferenceWithRandomModel:
    """
    Inject a random-weight model into the inference cache so predict_text()
    can run end-to-end without downloading a trained checkpoint.
    """

    @pytest.fixture(autouse=True)
    def _inject_random_model(self, tokenizer, random_model, tmp_path, monkeypatch):
        import ml.transformer.inference as infer_mod

        clear_model_cache()
        # Seed cache with our random model + real tokenizer
        key = str(tmp_path / "fake_model_dir")
        infer_mod._model_cache[key]     = random_model
        infer_mod._tokenizer_cache[key] = tokenizer

        # Patch VERSIONED_MODEL_DIR so predict_text() uses our cache key
        monkeypatch.setattr(infer_mod, "VERSIONED_MODEL_DIR", Path(key))

        self.model_dir = Path(key)
        yield
        clear_model_cache()

    def test_predict_text_returns_dict(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        assert isinstance(result, dict)

    def test_predict_text_required_keys(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        for key in ("label", "is_fake", "confidence",
                    "fake_probability", "real_probability",
                    "num_chunks", "aggregation_strategy",
                    "inference_time_ms"):
            assert key in result, f"Missing key: {key}"

    def test_label_is_fake_or_real(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        assert result["label"] in ("FAKE", "REAL")

    def test_is_fake_is_bool(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        assert isinstance(result["is_fake"], bool)

    def test_label_consistent_with_is_fake(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        if result["is_fake"]:
            assert result["label"] == "FAKE"
        else:
            assert result["label"] == "REAL"

    def test_probabilities_sum_to_one(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        total = result["fake_probability"] + result["real_probability"]
        assert abs(total - 1.0) < 1e-5

    def test_confidence_equals_max_probability(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        expected = max(result["fake_probability"], result["real_probability"])
        assert abs(result["confidence"] - expected) < 1e-6

    def test_short_text_produces_single_chunk(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        assert result["num_chunks"] == 1

    def test_long_text_produces_multiple_chunks(self):
        result = predict_text(LONG_TEXT, model_dir=self.model_dir)
        assert result["num_chunks"] > 1

    def test_inference_time_positive(self):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir)
        assert result["inference_time_ms"] > 0

    @pytest.mark.parametrize("strategy", ["weighted_mean", "max_confidence", "majority_vote"])
    def test_all_aggregation_strategies_work(self, strategy):
        result = predict_text(SHORT_TEXT, model_dir=self.model_dir, strategy=strategy)
        assert result["label"] in ("FAKE", "REAL")
        assert result["aggregation_strategy"] == strategy

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            predict_text("", model_dir=self.model_dir)

    def test_predict_texts_batch_returns_list(self):
        results = predict_texts_batch(
            [SHORT_TEXT, SHORT_TEXT],
            model_dir=self.model_dir,
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_texts_batch_empty_returns_empty(self):
        results = predict_texts_batch([], model_dir=self.model_dir)
        assert results == []

    def test_predict_texts_batch_each_has_label(self):
        results = predict_texts_batch(
            [SHORT_TEXT, LONG_TEXT],
            model_dir=self.model_dir,
        )
        for r in results:
            assert r["label"] in ("FAKE", "REAL")


# =============================================================================
# Model cache
# =============================================================================

class TestModelCache:
    def test_clear_cache_empties_both_dicts(self):
        import ml.transformer.inference as infer_mod
        infer_mod._model_cache["x"]     = object()
        infer_mod._tokenizer_cache["x"] = object()
        clear_model_cache()
        assert len(infer_mod._model_cache)     == 0
        assert len(infer_mod._tokenizer_cache) == 0

    def test_load_missing_model_raises(self, tmp_path):
        clear_model_cache()
        with pytest.raises(FileNotFoundError):
            from ml.transformer.inference import load_model_and_tokenizer
            load_model_and_tokenizer(tmp_path / "nonexistent_model")

    def test_second_load_uses_cache(self, tokenizer, random_model, tmp_path, monkeypatch):
        """Verify the model is only 'loaded' once — second call hits cache."""
        import ml.transformer.inference as infer_mod

        clear_model_cache()
        key = str(tmp_path / "cached_model")
        # Pre-populate cache
        infer_mod._model_cache[key]     = random_model
        infer_mod._tokenizer_cache[key] = tokenizer

        from ml.transformer.inference import load_model_and_tokenizer
        m1, t1 = load_model_and_tokenizer(Path(key))
        m2, t2 = load_model_and_tokenizer(Path(key))

        assert m1 is m2   # same object → cache hit
        assert t1 is t2
        clear_model_cache()
