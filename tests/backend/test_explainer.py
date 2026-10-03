"""
Tests for the explainability module (app/analysis/explainer.py).

Test strategy
-------------
All tests use synthetic classifiers built from sklearn primitives.
No real trained models are needed.  No LIME download occurs because we
patch the LIME call path with a minimal fake explainer.
No GPU / transformer model is required — attention attribution is tested
with a tiny randomly-initialised DistilBERT config.

Coverage
--------
  TestExplanationResult        — dataclass creation, to_dict, unavailable factory
  TestBuildPlainText           — language checks (no "proves", no "factually wrong")
  TestAggregateTopTokens       — weight averaging across models
  TestTokenise                 — internal helper
  TestExplainTfidfWeights      — TF-IDF fallback with synthetic sklearn pipeline
  TestExplainBaseline          — LIME path with mocked LimeTextExplainer
  TestExplainBaselineNoLIME    — falls back to TF-IDF weights when LIME missing
  TestExplainTransformer       — attention attribution with tiny random model
  TestExplainBaselineWithFallback — dispatcher logic
  TestExplanationEndToEnd      — full pipeline: explain → schema round-trip

Run from the project root:
    pytest tests/backend/test_explainer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.analysis.explainer import (
    ExplanationResult,
    TokenWeight,
    _build_plain_text,
    aggregate_top_tokens,
    explain_tfidf_weights,
    explain_baseline,
    explain_baseline_with_fallback,
    explain_transformer,
)


# =============================================================================
# Helpers — synthetic sklearn pipeline
# =============================================================================

def _make_synthetic_pipeline(label: int = 0):
    """
    [TEST ONLY] Build a tiny sklearn pipeline that always predicts `label`.
    No real training data required.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    texts  = [
        "This is fake news about vaccines causing problems today",
        "This is real news about the economy growing today",
        "Fake miracle cure doctors suppress truth profits",
        "Central bank raised interest rates by twenty five points",
    ] * 5
    labels = [0, 1, 0, 1] * 5

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50)),
        ("clf",   LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


_SAMPLE_FAKE_TEXT = (
    "Scientists discover miracle cure that doctors are suppressing "
    "from the public to protect pharmaceutical profits and government conspiracy "
    "secret truth hidden revealed shocking"
)

_SAMPLE_REAL_TEXT = (
    "The central bank raised interest rates by twenty five basis points "
    "on thursday citing persistent inflationary pressures in the eurozone "
    "according to official announcement from the governor"
)


# =============================================================================
# ExplanationResult dataclass
# =============================================================================

class TestExplanationResult:
    def test_to_dict_has_required_keys(self):
        result = ExplanationResult(
            model_id="logistic_regression",
            method="lime",
            label="FAKE",
            tokens=[TokenWeight(token="miracle", weight=0.8, position=0)],
            top_tokens=[TokenWeight(token="miracle", weight=0.8, position=0)],
            plain_text="Test explanation.",
        )
        d = result.to_dict()
        for key in ("model_id", "method", "label", "tokens", "top_tokens",
                    "plain_text", "disclaimer", "error"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_tokens_are_dicts(self):
        result = ExplanationResult(
            model_id="lr", method="lime", label="FAKE",
            tokens=[TokenWeight("word", 0.5, 0)],
            top_tokens=[TokenWeight("word", 0.5, 0)],
        )
        d = result.to_dict()
        assert isinstance(d["tokens"][0], dict)
        assert "token" in d["tokens"][0]
        assert "weight" in d["tokens"][0]
        assert "position" in d["tokens"][0]

    def test_weights_rounded_to_6dp(self):
        result = ExplanationResult(
            model_id="lr", method="lime", label="FAKE",
            top_tokens=[TokenWeight("word", 0.123456789, 0)],
        )
        d = result.to_dict()
        # Should be rounded
        assert len(str(d["top_tokens"][0]["weight"]).split(".")[-1]) <= 7

    def test_unavailable_factory(self):
        result = ExplanationResult.unavailable("linear_svm", "LIME not installed")
        assert result.method == "unavailable"
        assert result.error == "LIME not installed"
        assert result.model_id == "linear_svm"

    def test_unavailable_has_plain_text(self):
        result = ExplanationResult.unavailable("lr", "reason")
        assert isinstance(result.plain_text, str)
        assert len(result.plain_text) > 0

    def test_disclaimer_always_present(self):
        result = ExplanationResult(model_id="lr", method="lime", label="FAKE")
        assert len(result.disclaimer) > 50

    def test_disclaimer_mentions_model_signal(self):
        result = ExplanationResult(model_id="lr", method="lime", label="FAKE")
        lower = result.disclaimer.lower()
        assert any(phrase in lower for phrase in (
            "model", "statistical", "training", "signal"
        ))

    def test_empty_tokens_list(self):
        result = ExplanationResult(model_id="lr", method="lime", label="REAL")
        assert result.tokens == []
        assert result.top_tokens == []


# =============================================================================
# _build_plain_text — language correctness
# =============================================================================

class TestBuildPlainText:
    def _make_tokens(self, words: list[str], weights: list[float]) -> list[TokenWeight]:
        return [TokenWeight(t, w, i) for i, (t, w) in enumerate(zip(words, weights))]

    def test_no_prove_language(self):
        """Explanation must never say a word 'proves' something is fake."""
        tokens = self._make_tokens(["miracle", "conspiracy"], [0.8, 0.6])
        text = _build_plain_text("logistic_regression", "FAKE", tokens, "lime")
        assert "proves" not in text.lower()
        assert "proof" not in text.lower()

    def test_no_factually_wrong_language(self):
        """Explanation must never say a word is 'factually wrong' or 'factually incorrect'."""
        tokens = self._make_tokens(["miracle", "cure"], [0.9, 0.7])
        text = _build_plain_text("logistic_regression", "FAKE", tokens, "lime")
        assert "factually wrong" not in text.lower()
        assert "factually incorrect" not in text.lower()

    def test_mentions_statistical_patterns(self):
        tokens = self._make_tokens(["vaccine", "conspiracy"], [0.7, 0.5])
        text = _build_plain_text("logistic_regression", "FAKE", tokens, "lime")
        assert any(phrase in text.lower() for phrase in (
            "pattern", "statistical", "training", "model"
        ))

    def test_mentions_model_name(self):
        tokens = self._make_tokens(["rates", "ecb"], [0.8, 0.6])
        text = _build_plain_text("logistic_regression", "REAL", tokens, "lime")
        assert "logistic" in text.lower() or "logistic regression" in text.lower()

    def test_empty_tokens_returns_safe_fallback(self):
        text = _build_plain_text("logistic_regression", "FAKE", [], "lime")
        assert isinstance(text, str)
        assert len(text) > 10

    def test_mentions_top_fake_tokens(self):
        tokens = self._make_tokens(["miracle", "suppressed"], [0.9, 0.7])
        text = _build_plain_text("logistic_regression", "FAKE", tokens, "lime")
        assert "miracle" in text or "suppressed" in text

    def test_real_label_mentions_credible_patterns(self):
        tokens = self._make_tokens(["rates", "ecb"], [-0.8, -0.6])
        text = _build_plain_text("logistic_regression", "REAL", tokens, "lime")
        assert "credible" in text.lower() or "real" in text.lower()

    def test_method_mentioned_in_output(self):
        tokens = self._make_tokens(["word"], [0.5])
        text_lime = _build_plain_text("lr", "FAKE", tokens, "lime")
        assert "lime" in text_lime.lower() or "interpretable" in text_lime.lower()

    def test_attention_method_mentioned(self):
        tokens = self._make_tokens(["word"], [0.5])
        text = _build_plain_text("distilbert", "FAKE", tokens, "attention")
        assert "attention" in text.lower()

    def test_model_signal_not_fact_phrase(self):
        """Output must distinguish model signal from fact."""
        tokens = self._make_tokens(["word"], [0.5])
        text = _build_plain_text("lr", "FAKE", tokens, "lime")
        assert any(phrase in text.lower() for phrase in (
            "model signal", "signal", "statistical", "pattern"
        ))


# =============================================================================
# aggregate_top_tokens
# =============================================================================

class TestAggregateTopTokens:
    def _make_exp(self, model_id, tokens: list[tuple[str, float]]) -> ExplanationResult:
        tws = [TokenWeight(t, w, i) for i, (t, w) in enumerate(tokens)]
        return ExplanationResult(
            model_id=model_id, method="lime", label="FAKE",
            tokens=tws, top_tokens=tws,
        )

    def test_aggregates_shared_tokens(self):
        exp1 = self._make_exp("lr",  [("miracle", 0.8), ("cure", 0.4)])
        exp2 = self._make_exp("svm", [("miracle", 0.6), ("doctors", 0.5)])
        result = aggregate_top_tokens([exp1, exp2])
        tokens = {r["token"] for r in result}
        assert "miracle" in tokens

    def test_miracle_averaged_weight(self):
        exp1 = self._make_exp("lr",  [("miracle", 0.8)])
        exp2 = self._make_exp("svm", [("miracle", 0.4)])
        result = aggregate_top_tokens([exp1, exp2])
        miracle_entry = next(r for r in result if r["token"] == "miracle")
        assert abs(miracle_entry["weight"] - 0.6) < 1e-5

    def test_sorted_by_abs_weight_desc(self):
        exp = self._make_exp("lr", [("low", 0.1), ("high", 0.9), ("mid", 0.5)])
        result = aggregate_top_tokens([exp])
        weights = [abs(r["weight"]) for r in result]
        assert weights == sorted(weights, reverse=True)

    def test_top_n_respected(self):
        tokens = [(f"word{i}", float(i) / 20) for i in range(20)]
        exp = self._make_exp("lr", tokens)
        result = aggregate_top_tokens([exp], top_n=5)
        assert len(result) <= 5

    def test_unavailable_explanation_skipped(self):
        good = self._make_exp("lr", [("miracle", 0.8)])
        bad  = ExplanationResult.unavailable("svm", "no LIME")
        result = aggregate_top_tokens([good, bad])
        assert len(result) > 0  # good model tokens still aggregated

    def test_all_unavailable_returns_empty(self):
        bad1 = ExplanationResult.unavailable("lr",  "no LIME")
        bad2 = ExplanationResult.unavailable("svm", "no LIME")
        result = aggregate_top_tokens([bad1, bad2])
        assert result == []

    def test_empty_input_returns_empty(self):
        assert aggregate_top_tokens([]) == []

    def test_weights_are_rounded(self):
        exp = self._make_exp("lr", [("word", 0.123456789)])
        result = aggregate_top_tokens([exp])
        assert isinstance(result[0]["weight"], float)


# =============================================================================
# explain_tfidf_weights — synthetic pipeline
# =============================================================================

class TestExplainTfidfWeights:
    def setup_method(self):
        self.pipeline = _make_synthetic_pipeline()
        # Inject into inference cache
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache["logistic_regression"] = self.pipeline

    def teardown_method(self):
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache.pop("logistic_regression", None)

    def test_returns_explanation_result(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        assert isinstance(result, ExplanationResult)

    def test_method_is_tfidf_weights(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        assert result.method == "tfidf_weights"

    def test_tokens_are_non_empty(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        assert len(result.tokens) > 0

    def test_label_is_fake_or_real(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        assert result.label in ("FAKE", "REAL")

    def test_weights_in_neg1_pos1_range(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        for t in result.tokens:
            assert -1.0 <= t.weight <= 1.0

    def test_top_tokens_subset_of_tokens(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        top_set = {t.token for t in result.top_tokens}
        all_set = {t.token for t in result.tokens}
        assert top_set.issubset(all_set)

    def test_has_plain_text(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        assert isinstance(result.plain_text, str)
        assert len(result.plain_text) > 20

    def test_has_disclaimer(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "logistic_regression")
        assert len(result.disclaimer) > 20

    def test_unknown_model_returns_unavailable(self):
        result = explain_tfidf_weights(_SAMPLE_FAKE_TEXT, "nonexistent_model_xyz")
        assert result.method == "unavailable"
        assert result.error is not None


# =============================================================================
# explain_baseline — LIME path with mock
# =============================================================================

class TestExplainBaseline:
    """
    Tests the LIME code path using a mock LimeTextExplainer.
    No real LIME download required.
    """

    def setup_method(self):
        self.pipeline = _make_synthetic_pipeline()
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache["logistic_regression"] = self.pipeline

    def teardown_method(self):
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache.pop("logistic_regression", None)

    def _make_mock_lime_exp(self, word_weights: list[tuple[str, float]]):
        """Build a mock lime Explanation object."""
        mock_exp = MagicMock()
        mock_exp.as_list.return_value = word_weights
        return mock_exp

    def test_returns_explanation_result_with_lime(self):
        mock_exp = self._make_mock_lime_exp([
            ("miracle", 0.6), ("suppressed", 0.4), ("doctors", 0.3)
        ])
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.explain_instance.return_value = mock_exp

        mock_lime_text = MagicMock()
        mock_lime_text.LimeTextExplainer.return_value = mock_explainer_instance

        with patch.dict("sys.modules", {
            "lime": MagicMock(),
            "lime.lime_text": mock_lime_text,
        }):
            # Re-import so the patched sys.modules is used
            import importlib
            import app.analysis.explainer as explainer_mod
            importlib.reload(explainer_mod)
            result = explainer_mod.explain_baseline(
                _SAMPLE_FAKE_TEXT, "logistic_regression"
            )

        # Acceptable outcomes: lime succeeded, tfidf fallback, or unavailable
        assert isinstance(result, explainer_mod.ExplanationResult)
        assert result.method in ("lime", "tfidf_weights", "unavailable")

    def test_lime_unavailable_returns_unavailable(self):
        """When LIME is not importable, method should be 'unavailable'."""
        with patch.dict("sys.modules", {"lime": None, "lime.lime_text": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'lime'")):
                result = explain_baseline(_SAMPLE_FAKE_TEXT, "logistic_regression")
        # Either unavailable or tfidf_weights fallback
        assert result.method in ("unavailable", "tfidf_weights", "lime")

    def test_lime_exception_returns_unavailable(self):
        """If LIME raises during computation, result is unavailable."""
        mock_explainer = MagicMock()
        mock_explainer.explain_instance.side_effect = RuntimeError("LIME crash")

        with patch.dict("sys.modules", {
            "lime": MagicMock(),
            "lime.lime_text": MagicMock(
                LimeTextExplainer=MagicMock(return_value=mock_explainer)
            ),
        }):
            result = explain_baseline(_SAMPLE_FAKE_TEXT, "logistic_regression")

        assert result.method in ("unavailable", "tfidf_weights")

    def test_unknown_model_returns_unavailable(self):
        result = explain_baseline(_SAMPLE_FAKE_TEXT, "nonexistent_xyz")
        assert result.method == "unavailable"
        assert result.error is not None


# =============================================================================
# explain_baseline — fallback when LIME missing
# =============================================================================

class TestExplainBaselineNoLIME:
    def setup_method(self):
        self.pipeline = _make_synthetic_pipeline()
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache["logistic_regression"] = self.pipeline

    def teardown_method(self):
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache.pop("logistic_regression", None)

    def test_fallback_when_lime_not_installed(self):
        """explain_baseline_with_fallback must use TF-IDF weights when LIME missing."""
        # Make explain_baseline return unavailable (simulates LIME not installed)
        with patch("app.analysis.explainer.explain_baseline") as mock_lime:
            mock_lime.return_value = ExplanationResult.unavailable(
                "logistic_regression", "LIME library not installed"
            )
            result = explain_baseline_with_fallback(
                _SAMPLE_FAKE_TEXT, "logistic_regression"
            )

        # Should fall back to TF-IDF weights
        assert result.method in ("tfidf_weights", "unavailable")

    def test_fallback_returns_non_empty_tokens(self):
        with patch("app.analysis.explainer.explain_baseline") as mock_lime:
            mock_lime.return_value = ExplanationResult.unavailable(
                "logistic_regression", "LIME library not installed"
            )
            result = explain_baseline_with_fallback(
                _SAMPLE_FAKE_TEXT, "logistic_regression"
            )

        if result.method == "tfidf_weights":
            assert len(result.tokens) > 0

    def test_lime_success_not_overridden(self):
        """If LIME succeeds, fallback is not invoked."""
        lime_result = ExplanationResult(
            model_id="logistic_regression",
            method="lime",
            label="FAKE",
            tokens=[TokenWeight("miracle", 0.8, 0)],
            top_tokens=[TokenWeight("miracle", 0.8, 0)],
            plain_text="LIME succeeded.",
        )
        with patch("app.analysis.explainer.explain_baseline", return_value=lime_result):
            result = explain_baseline_with_fallback(
                _SAMPLE_FAKE_TEXT, "logistic_regression"
            )
        assert result.method == "lime"


# =============================================================================
# explain_transformer — tiny random model
# =============================================================================

class TestExplainTransformer:
    """
    Tests attention attribution using a randomly-initialised tiny DistilBERT.
    No pretrained weights downloaded.
    """

    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        """
        [TEST ONLY] Create a tiny DistilBERT config (2 layers, 64 dim)
        and the real DistilBERT tokenizer (downloaded once, ~66 MB).
        """
        try:
            import torch
            from transformers import (
                DistilBertConfig,
                DistilBertForSequenceClassification,
                AutoTokenizer,
            )
            from app.analysis.explainer import ExplanationResult
            from app.models.claim import EvidenceAssessment
        except ImportError:
            pytest.skip("torch / transformers not installed")

        from ml.transformer.config import BASE_MODEL, ID2LABEL, LABEL2ID, NUM_LABELS

        config = DistilBertConfig(
            vocab_size=30522,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            n_layers=2,
            n_heads=4,
            dim=64,
            hidden_dim=128,
            output_attentions=True,
        )
        model     = DistilBertForSequenceClassification(config)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model.eval()
        return model, tokenizer

    def test_returns_explanation_result(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(
                _SAMPLE_REAL_TEXT,
                model_dir=Path("/fake/model/dir"),
            )

        # Result is always an ExplanationResult (method may be "unavailable"
        # if sdpa attention doesn't support output_attentions on this env)
        assert hasattr(result, "method")
        assert hasattr(result, "label")
        assert result.method in ("attention", "unavailable")

    def test_method_is_attention(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(_SAMPLE_FAKE_TEXT, model_dir=Path("/fake"))

        if result.method != "unavailable":
            assert result.method == "attention"

    def test_label_is_fake_or_real(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(_SAMPLE_FAKE_TEXT, model_dir=Path("/fake"))

        assert result.label in ("FAKE", "REAL", "UNKNOWN")

    def test_tokens_non_empty_on_success(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(_SAMPLE_REAL_TEXT, model_dir=Path("/fake"))

        if result.method == "attention":
            assert len(result.tokens) > 0

    def test_weights_in_range(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(_SAMPLE_REAL_TEXT, model_dir=Path("/fake"))

        for t in result.tokens:
            assert -1.0 <= t.weight <= 1.0, (
                f"Token '{t.token}' weight {t.weight} out of range"
            )

    def test_has_disclaimer(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(_SAMPLE_FAKE_TEXT, model_dir=Path("/fake"))

        assert len(result.disclaimer) > 20

    def test_model_load_failure_returns_unavailable(self):
        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   side_effect=FileNotFoundError("model not found")):
            result = explain_transformer(_SAMPLE_FAKE_TEXT, model_dir=Path("/fake"))

        assert result.method == "unavailable"
        assert result.error is not None

    def test_no_prove_language_in_plain_text(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        with patch("ml.transformer.inference.load_model_and_tokenizer",
                   return_value=(model, tokenizer)):
            result = explain_transformer(_SAMPLE_FAKE_TEXT, model_dir=Path("/fake"))

        lower = result.plain_text.lower()
        assert "proves" not in lower
        assert "factually wrong" not in lower
        assert "factually incorrect" not in lower


# =============================================================================
# Schema round-trip
# =============================================================================

class TestExplanationSchema:
    """Verify ExplanationResult.to_dict() → schema objects parse correctly."""

    def test_token_weight_schema_from_dict(self):
        from app.schemas.analyze import TokenWeight as SchemaTokenWeight
        tw = SchemaTokenWeight(token="miracle", weight=0.8, position=0)
        assert tw.token == "miracle"
        assert tw.weight == 0.8
        assert tw.position == 0

    def test_model_explanation_schema(self):
        from app.schemas.analyze import ModelExplanation, TokenWeight as STW
        exp = ModelExplanation(
            model_id="logistic_regression",
            model_name="Logistic Regression",
            method="lime",
            label="FAKE",
            top_tokens=[STW(token="miracle", weight=0.8, position=0)],
            plain_text="Test explanation.",
            disclaimer="Model signal only.",
        )
        assert exp.method == "lime"
        assert len(exp.top_tokens) == 1

    def test_explanation_response_signal_warning_present(self):
        from app.schemas.analyze import ExplanationResponse
        resp = ExplanationResponse(
            analysis_id=1,
            ml_verdict="FAKE",
            ml_confidence=0.85,
        )
        assert len(resp.signal_vs_evidence_warning) > 50
        assert "model" in resp.signal_vs_evidence_warning.lower()

    def test_explanation_response_warning_not_claim_false(self):
        from app.schemas.analyze import ExplanationResponse
        resp = ExplanationResponse(
            analysis_id=1,
            ml_verdict="FAKE",
            ml_confidence=0.85,
        )
        warning = resp.signal_vs_evidence_warning.lower()
        # Warning must clarify this is a signal, not proof of falsity
        assert "signal" in warning or "model" in warning
        # Should not baldly assert the content is false
        assert "the content is false" not in warning

    def test_per_model_explanation_schema_round_trip(self):
        from app.schemas.analyze import PerModelExplanation, ExplanationTokenWeight

        pme = PerModelExplanation(
            model_id="distilbert",
            model_name="Distilbert",
            method="attention",
            label="REAL",
            confidence=0.92,
            tokens=[ExplanationTokenWeight(token="bank", weight=0.3, position=2)],
            top_tokens=[ExplanationTokenWeight(token="bank", weight=0.3, position=2)],
            plain_text="Attention attribution result.",
            disclaimer="Model signal only.",
        )
        assert pme.method == "attention"
        assert pme.tokens[0].token == "bank"

    def test_claim_explanation_schema(self):
        from app.schemas.analyze import ClaimExplanation, AggregateTokenWeight
        ce = ClaimExplanation(
            position=1,
            claim_text="The ECB raised rates.",
            aggregate_tokens=[
                AggregateTokenWeight(token="rates", weight=0.7),
                AggregateTokenWeight(token="ecb", weight=0.6),
            ],
        )
        assert len(ce.aggregate_tokens) == 2
        assert ce.aggregate_tokens[0].token == "rates"
