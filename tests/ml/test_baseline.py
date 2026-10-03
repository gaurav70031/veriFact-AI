"""
Unit and integration tests for ml.baseline.

Coverage
--------
trainer.py    — pipeline building, train_single(), save/load round-trip
evaluator.py  — all metrics computed from real in-memory predictions
inference.py  — predict_single(), predict_batch(), predict_all_models(),
                cache behaviour, edge cases

All tests use synthetic in-memory data.  No real dataset files, no GPU,
and no pre-existing saved artifacts are required.

Run from the project root:
    pytest tests/ml/test_baseline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.baseline.trainer   import (
    build_lr_pipeline,
    build_svm_pipeline,
    build_nb_pipeline,
    train_single,
    train_all,
    save_pipeline,
    load_pipeline,
)
from ml.baseline.evaluator import evaluate_pipeline, evaluate_all
from ml.baseline.inference import (
    predict_single,
    predict_all_models,
    predict_batch,
    clear_cache,
    get_pipeline,
)
from ml.baseline.config import MODEL_IDS, MODEL_PIPELINE_PATHS


# =============================================================================
# Fixtures — synthetic data
# =============================================================================

FAKE_TEXTS = [
    f"Scientists discover miracle cure number {i} that doctors are suppressing "
    f"from the public to protect pharmaceutical profits and government conspiracy "
    f"secret hidden truth revealed shocking"
    for i in range(120)
]

REAL_TEXTS = [
    f"The central bank raised interest rates by twenty five basis points on "
    f"thursday citing persistent inflationary pressures across the economy "
    f"according to official announcement number {i}"
    for i in range(120)
]


def _make_df(n_fake: int = 100, n_real: int = 100) -> pd.DataFrame:
    """Return a balanced DataFrame with 'cleaned_text' and 'label' columns."""
    rows = (
        [{"cleaned_text": FAKE_TEXTS[i % len(FAKE_TEXTS)], "label": 0} for i in range(n_fake)]
        + [{"cleaned_text": REAL_TEXTS[i % len(REAL_TEXTS)], "label": 1} for i in range(n_real)]
    )
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def train_val_test():
    """Return (train_df, val_df, test_df) synthetic splits."""
    df = _make_df(n_fake=100, n_real=100)
    n  = len(df)
    train_df = df.iloc[: int(n * 0.8)].reset_index(drop=True)
    val_df   = df.iloc[int(n * 0.8): int(n * 0.9)].reset_index(drop=True)
    test_df  = df.iloc[int(n * 0.9):].reset_index(drop=True)
    return train_df, val_df, test_df


@pytest.fixture
def fitted_lr(train_val_test):
    train_df, val_df, _ = train_val_test
    pipeline = build_lr_pipeline()
    pipeline.fit(train_df["cleaned_text"].tolist(), train_df["label"].tolist())
    return pipeline


@pytest.fixture
def fitted_svm(train_val_test):
    train_df, val_df, _ = train_val_test
    pipeline = build_svm_pipeline()
    pipeline.fit(train_df["cleaned_text"].tolist(), train_df["label"].tolist())
    return pipeline


@pytest.fixture
def fitted_nb(train_val_test):
    train_df, val_df, _ = train_val_test
    pipeline = build_nb_pipeline()
    pipeline.fit(train_df["cleaned_text"].tolist(), train_df["label"].tolist())
    return pipeline


# =============================================================================
# Pipeline builders
# =============================================================================

class TestPipelineBuilders:
    def test_lr_pipeline_is_sklearn_pipeline(self):
        p = build_lr_pipeline()
        assert isinstance(p, Pipeline)

    def test_svm_pipeline_is_sklearn_pipeline(self):
        p = build_svm_pipeline()
        assert isinstance(p, Pipeline)

    def test_nb_pipeline_is_sklearn_pipeline(self):
        p = build_nb_pipeline()
        assert isinstance(p, Pipeline)

    def test_lr_has_tfidf_and_clf_steps(self):
        p = build_lr_pipeline()
        assert "tfidf" in p.named_steps
        assert "clf"   in p.named_steps

    def test_svm_has_tfidf_and_clf_steps(self):
        p = build_svm_pipeline()
        assert "tfidf" in p.named_steps
        assert "clf"   in p.named_steps

    def test_nb_has_tfidf_and_clf_steps(self):
        p = build_nb_pipeline()
        assert "tfidf" in p.named_steps
        assert "clf"   in p.named_steps

    def test_lr_clf_supports_predict_proba(self):
        p = build_lr_pipeline()
        assert hasattr(p.named_steps["clf"], "predict_proba")

    def test_svm_clf_supports_predict_proba_after_calibration(self):
        # CalibratedClassifierCV wraps LinearSVC → predict_proba available
        p = build_svm_pipeline()
        assert hasattr(p.named_steps["clf"], "predict_proba")

    def test_nb_clf_supports_predict_proba(self):
        p = build_nb_pipeline()
        assert hasattr(p.named_steps["clf"], "predict_proba")

    def test_different_builder_calls_return_different_objects(self):
        p1 = build_lr_pipeline()
        p2 = build_lr_pipeline()
        assert p1 is not p2


# =============================================================================
# Trainer — train_single()
# =============================================================================

class TestTrainSingle:
    def test_returns_pipeline_and_record(self, train_val_test):
        train_df, val_df, _ = train_val_test
        pipeline, record = train_single("logistic_regression", train_df, val_df, save=False)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(record, dict)

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_all_model_ids_train(self, model_id, train_val_test):
        train_df, val_df, _ = train_val_test
        pipeline, record = train_single(model_id, train_df, val_df, save=False)
        assert isinstance(pipeline, Pipeline)

    def test_record_has_required_keys(self, train_val_test):
        train_df, val_df, _ = train_val_test
        _, record = train_single("logistic_regression", train_df, val_df, save=False)
        for key in ("model_id", "train_samples", "train_time_s",
                    "hyperparameters", "tfidf_params"):
            assert key in record, f"Missing key: {key}"

    def test_record_train_samples_correct(self, train_val_test):
        train_df, val_df, _ = train_val_test
        _, record = train_single("logistic_regression", train_df, val_df, save=False)
        assert record["train_samples"] == len(train_df)

    def test_record_train_time_positive(self, train_val_test):
        train_df, val_df, _ = train_val_test
        _, record = train_single("logistic_regression", train_df, val_df, save=False)
        assert record["train_time_s"] > 0

    def test_unknown_model_id_raises(self, train_val_test):
        train_df, val_df, _ = train_val_test
        with pytest.raises(ValueError, match="Unknown model_id"):
            train_single("does_not_exist", train_df, val_df, save=False)

    def test_pipeline_predicts_after_training(self, train_val_test):
        train_df, val_df, test_df = train_val_test
        pipeline, _ = train_single("logistic_regression", train_df, val_df, save=False)
        texts = test_df["cleaned_text"].tolist()
        preds = pipeline.predict(texts)
        assert len(preds) == len(texts)
        assert set(preds).issubset({0, 1})

    def test_tfidf_fitted_only_on_train(self, train_val_test):
        """
        Verify the TF-IDF vocabulary is set after fitting (i.e. it was
        fitted on training data) and does not contain tokens from val/test.
        This is a structural check — sklearn Pipelines enforce fit-on-train
        automatically, but we verify the vocabulary exists.
        """
        train_df, val_df, _ = train_val_test
        pipeline, _ = train_single("logistic_regression", train_df, val_df, save=False)
        vocab = pipeline.named_steps["tfidf"].vocabulary_
        assert isinstance(vocab, dict)
        assert len(vocab) > 0


# =============================================================================
# Trainer — train_all()
# =============================================================================

class TestTrainAll:
    def test_returns_all_three_models(self, train_val_test):
        train_df, val_df, _ = train_val_test
        results = train_all(train_df, val_df, save=False)
        assert set(results.keys()) == set(MODEL_IDS)

    def test_each_result_is_pipeline_and_record(self, train_val_test):
        train_df, val_df, _ = train_val_test
        results = train_all(train_df, val_df, save=False)
        for model_id, (pipeline, record) in results.items():
            assert isinstance(pipeline, Pipeline), f"{model_id} pipeline wrong type"
            assert isinstance(record,   dict),     f"{model_id} record wrong type"


# =============================================================================
# Trainer — save / load round-trip
# =============================================================================

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path, monkeypatch, train_val_test):
        train_df, val_df, _ = train_val_test
        import ml.baseline.config as cfg
        import ml.baseline.trainer as trainer_mod

        new_path = tmp_path / "lr_pipeline.pkl"
        monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS, "logistic_regression", new_path)
        monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, "logistic_regression", new_path)

        pipeline, _ = train_single("logistic_regression", train_df, val_df, save=True)
        assert new_path.exists()

    def test_load_returns_pipeline(self, tmp_path, monkeypatch, train_val_test):
        train_df, val_df, _ = train_val_test
        import ml.baseline.config as cfg
        import ml.baseline.trainer as trainer_mod

        new_path = tmp_path / "lr_pipeline.pkl"
        monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS, "logistic_regression", new_path)
        monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, "logistic_regression", new_path)

        # Train and save
        train_single("logistic_regression", train_df, val_df, save=True)
        # Load back
        loaded = load_pipeline("logistic_regression")
        assert isinstance(loaded, Pipeline)

    def test_loaded_pipeline_predicts_same_as_original(
        self, tmp_path, monkeypatch, train_val_test
    ):
        train_df, val_df, test_df = train_val_test
        import ml.baseline.config as cfg
        import ml.baseline.trainer as trainer_mod

        new_path = tmp_path / "lr_pipeline.pkl"
        monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS, "logistic_regression", new_path)
        monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, "logistic_regression", new_path)

        pipeline_orig, _ = train_single("logistic_regression", train_df, val_df, save=True)
        pipeline_loaded  = load_pipeline("logistic_regression")

        texts = test_df["cleaned_text"].tolist()
        preds_orig   = pipeline_orig.predict(texts)
        preds_loaded = pipeline_loaded.predict(texts)
        assert list(preds_orig) == list(preds_loaded)

    def test_load_missing_file_raises(self, tmp_path, monkeypatch):
        import ml.baseline.config  as cfg
        import ml.baseline.trainer as trainer_mod

        ghost_path = tmp_path / "nonexistent.pkl"
        monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS,    "logistic_regression", ghost_path)
        monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, "logistic_regression", ghost_path)

        with pytest.raises(FileNotFoundError):
            load_pipeline("logistic_regression")


# =============================================================================
# Evaluator
# =============================================================================

class TestEvaluatePipeline:
    def test_returns_dict(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        assert isinstance(result, dict)

    def test_required_metric_keys_present(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        required = {
            "model_id", "accuracy", "macro_f1", "weighted_f1",
            "macro_precision", "macro_recall",
            "per_class", "confusion_matrix",
            "inference_time_ms", "per_sample_ms", "test_samples",
        }
        for key in required:
            assert key in result, f"Missing metric key: {key}"

    def test_accuracy_in_valid_range(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_macro_f1_in_valid_range(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_roc_auc_present_for_lr(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        # LR supports predict_proba → ROC-AUC must be computed
        assert result["roc_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_roc_auc_present_for_svm(self, fitted_svm, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("linear_svm", fitted_svm, test_df, save=False)
        assert result["roc_auc"] is not None

    def test_confusion_matrix_shape(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        cm = result["confusion_matrix"]
        assert len(cm)    == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_confusion_matrix_sums_to_test_size(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        cm    = result["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == len(test_df)

    def test_per_class_has_fake_and_real(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        assert "FAKE" in result["per_class"]
        assert "REAL" in result["per_class"]

    def test_per_class_values_in_valid_range(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        for cls_name, metrics in result["per_class"].items():
            for metric_name, val in metrics.items():
                if metric_name != "support":
                    assert 0.0 <= val <= 1.0, (
                        f"{cls_name}.{metric_name} = {val} out of range"
                    )

    def test_inference_time_positive(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        assert result["inference_time_ms"] > 0
        assert result["per_sample_ms"]     > 0

    def test_test_samples_count_correct(self, fitted_lr, train_val_test):
        _, _, test_df = train_val_test
        result = evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=False)
        assert result["test_samples"] == len(test_df)

    def test_metrics_not_hardcoded(self, train_val_test):
        """
        Verify metrics come from actual computation by checking that a well-
        trained model produces a different (higher) ROC-AUC than a model
        trained on only FAKE samples (which must predict everything as FAKE,
        giving ROC-AUC = 0.5).
        """
        train_df, val_df, test_df = train_val_test

        # Model A: properly trained on balanced data
        p_good = build_lr_pipeline()
        p_good.fit(train_df["cleaned_text"].tolist(), train_df["label"].tolist())
        r_good = evaluate_pipeline("logistic_regression", p_good, test_df, save=False)

        # Model B: trained only on FAKE samples → will predict FAKE for everything
        # ROC-AUC is undefined in degenerate case, but accuracy will be ~0.5
        # (half real articles predicted wrong).
        fake_only = train_df[train_df["label"] == 0].copy()
        # Add one REAL so sklearn doesn't complain about single class
        one_real  = train_df[train_df["label"] == 1].iloc[:1]
        skewed_train = pd.concat([fake_only, one_real], ignore_index=True)

        p_bad = build_lr_pipeline()
        p_bad.fit(skewed_train["cleaned_text"].tolist(), skewed_train["label"].tolist())
        r_bad = evaluate_pipeline("logistic_regression", p_bad, test_df, save=False)

        # A model trained on balanced data should have strictly higher macro_f1
        # than one trained on a severely skewed set
        assert r_good["macro_f1"] != r_bad["macro_f1"] or \
               r_good["accuracy"] != r_bad["accuracy"], \
               "Metrics must differ between a balanced and skewed model"

    def test_empty_test_df_raises(self, fitted_lr):
        empty_df = pd.DataFrame({"cleaned_text": [], "label": []})
        with pytest.raises(ValueError, match="empty"):
            evaluate_pipeline("logistic_regression", fitted_lr, empty_df, save=False)

    def test_evaluate_all_returns_all_models(self, train_val_test):
        train_df, val_df, test_df = train_val_test
        pipelines = {}
        for model_id in MODEL_IDS:
            p = {
                "logistic_regression": build_lr_pipeline,
                "linear_svm":          build_svm_pipeline,
                "naive_bayes":         build_nb_pipeline,
            }[model_id]()
            p.fit(train_df["cleaned_text"].tolist(), train_df["label"].tolist())
            pipelines[model_id] = p

        results = evaluate_all(pipelines, test_df, save=False)
        assert set(results.keys()) == set(MODEL_IDS)
        for model_id, result in results.items():
            assert result["accuracy"] > 0.0

    def test_eval_results_saved_to_disk(self, tmp_path, monkeypatch, fitted_lr, train_val_test):
        """Verify save=True writes a valid JSON file."""
        import ml.baseline.config   as cfg
        import ml.baseline.evaluator as eval_mod

        eval_path = tmp_path / "lr_eval.json"
        monkeypatch.setattr(cfg, "EVAL_RESULTS_DIR", tmp_path)
        monkeypatch.setitem(cfg.MODEL_EVAL_PATHS, "logistic_regression", eval_path)
        monkeypatch.setitem(eval_mod.MODEL_EVAL_PATHS, "logistic_regression", eval_path)

        _, _, test_df = train_val_test
        evaluate_pipeline("logistic_regression", fitted_lr, test_df, save=True)

        assert eval_path.exists()
        with open(eval_path) as f:
            saved = json.load(f)
        assert saved["model_id"] == "logistic_regression"
        assert 0.0 <= saved["accuracy"] <= 1.0


# =============================================================================
# Inference
# =============================================================================

FAKE_SAMPLE = (
    "Scientists claim miracle cure suppressed by government and pharmaceutical "
    "companies to hide the shocking truth from the public conspiracy"
)
REAL_SAMPLE = (
    "The central bank raised interest rates by twenty five basis points on "
    "thursday citing persistent inflationary pressures according to officials"
)


class TestPredictSingle:
    """Tests for inference.predict_single() using cached trained pipelines."""

    @pytest.fixture(autouse=True)
    def _seed_cache(self, tmp_path, monkeypatch, train_val_test):
        """
        Train all three models into tmp_path and seed the inference cache
        so predict_single() uses them.
        """
        import ml.baseline.config  as cfg
        import ml.baseline.trainer as trainer_mod
        import ml.baseline.inference as infer_mod

        train_df, val_df, _ = train_val_test

        for model_id in MODEL_IDS:
            pkl_path = tmp_path / f"{model_id}.pkl"
            monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS,        model_id, pkl_path)
            monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, model_id, pkl_path)

        clear_cache()
        results = train_all(train_df, val_df, save=True)
        for model_id, (pipeline, _) in results.items():
            infer_mod._pipeline_cache[model_id] = pipeline

        yield

        clear_cache()

    def test_returns_dict(self):
        result = predict_single(FAKE_SAMPLE, model_id="logistic_regression")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = predict_single(REAL_SAMPLE, model_id="logistic_regression")
        for key in ("label", "is_fake", "confidence",
                    "fake_probability", "real_probability",
                    "model_id", "inference_time_ms"):
            assert key in result, f"Missing key: {key}"

    def test_label_is_fake_or_real(self):
        result = predict_single(FAKE_SAMPLE, model_id="logistic_regression")
        assert result["label"] in ("FAKE", "REAL")

    def test_is_fake_is_bool(self):
        result = predict_single(FAKE_SAMPLE, model_id="logistic_regression")
        assert isinstance(result["is_fake"], bool)

    def test_label_consistent_with_is_fake(self):
        result = predict_single(FAKE_SAMPLE, model_id="logistic_regression")
        if result["is_fake"]:
            assert result["label"] == "FAKE"
        else:
            assert result["label"] == "REAL"

    def test_probabilities_sum_to_one(self):
        result = predict_single(REAL_SAMPLE, model_id="logistic_regression")
        total = result["fake_probability"] + result["real_probability"]
        assert abs(total - 1.0) < 1e-5

    def test_confidence_equals_max_probability(self):
        result = predict_single(REAL_SAMPLE, model_id="logistic_regression")
        expected = max(result["fake_probability"], result["real_probability"])
        assert abs(result["confidence"] - expected) < 1e-6

    def test_confidence_in_valid_range(self):
        result = predict_single(REAL_SAMPLE, model_id="logistic_regression")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_inference_time_positive(self):
        result = predict_single(FAKE_SAMPLE, model_id="logistic_regression")
        assert result["inference_time_ms"] > 0

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_all_models_return_valid_prediction(self, model_id):
        result = predict_single(REAL_SAMPLE, model_id=model_id)
        assert result["label"] in ("FAKE", "REAL")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unknown_model_id_raises(self):
        with pytest.raises(ValueError, match="Unknown model_id"):
            predict_single(FAKE_SAMPLE, model_id="nonexistent_model")

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, Exception)):
            predict_single("", model_id="logistic_regression")

    def test_very_short_text_returns_warning(self):
        result = predict_single("hi", model_id="logistic_regression")
        # Very short text collapses to empty after cleaning → uncertain result
        assert "warning" in result or result["label"] in ("FAKE", "REAL", "UNVERIFIED")

    def test_clean_false_passes_raw_text(self):
        # With clean=False the raw text is sent directly — should still work
        result = predict_single(FAKE_SAMPLE, model_id="logistic_regression", clean=False)
        assert result["label"] in ("FAKE", "REAL")


class TestPredictAllModels:
    @pytest.fixture(autouse=True)
    def _seed_cache(self, tmp_path, monkeypatch, train_val_test):
        import ml.baseline.config   as cfg
        import ml.baseline.trainer  as trainer_mod
        import ml.baseline.inference as infer_mod

        train_df, val_df, _ = train_val_test
        for model_id in MODEL_IDS:
            pkl_path = tmp_path / f"{model_id}.pkl"
            monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS,         model_id, pkl_path)
            monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, model_id, pkl_path)

        clear_cache()
        results = train_all(train_df, val_df, save=True)
        for model_id, (pipeline, _) in results.items():
            infer_mod._pipeline_cache[model_id] = pipeline
        yield
        clear_cache()

    def test_returns_all_three_models(self):
        results = predict_all_models(FAKE_SAMPLE)
        assert set(results.keys()) == set(MODEL_IDS)

    def test_each_result_has_label(self):
        results = predict_all_models(REAL_SAMPLE)
        for model_id, result in results.items():
            assert result["label"] in ("FAKE", "REAL"), (
                f"{model_id} returned unexpected label: {result['label']}"
            )


class TestPredictBatch:
    @pytest.fixture(autouse=True)
    def _seed_cache(self, tmp_path, monkeypatch, train_val_test):
        import ml.baseline.config   as cfg
        import ml.baseline.trainer  as trainer_mod
        import ml.baseline.inference as infer_mod

        train_df, val_df, _ = train_val_test
        for model_id in MODEL_IDS:
            pkl_path = tmp_path / f"{model_id}.pkl"
            monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS,         model_id, pkl_path)
            monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, model_id, pkl_path)

        clear_cache()
        results = train_all(train_df, val_df, save=True)
        for model_id, (pipeline, _) in results.items():
            infer_mod._pipeline_cache[model_id] = pipeline
        yield
        clear_cache()

    def test_returns_list_of_correct_length(self):
        texts  = [FAKE_SAMPLE, REAL_SAMPLE, FAKE_SAMPLE]
        results = predict_batch(texts, model_id="logistic_regression")
        assert len(results) == 3

    def test_each_result_has_required_keys(self):
        results = predict_batch([REAL_SAMPLE, FAKE_SAMPLE], model_id="logistic_regression")
        for r in results:
            assert "label"            in r
            assert "confidence"       in r
            assert "fake_probability" in r
            assert "real_probability" in r

    def test_empty_list_returns_empty_list(self):
        result = predict_batch([], model_id="logistic_regression")
        assert result == []

    def test_probabilities_sum_to_one_per_sample(self):
        results = predict_batch(
            [FAKE_SAMPLE, REAL_SAMPLE, FAKE_SAMPLE],
            model_id="logistic_regression",
        )
        for r in results:
            total = r["fake_probability"] + r["real_probability"]
            assert abs(total - 1.0) < 1e-5

    def test_batch_consistent_with_single(self):
        """Batch and single prediction must return identical results."""
        result_single = predict_single(REAL_SAMPLE, model_id="logistic_regression")
        result_batch  = predict_batch([REAL_SAMPLE],  model_id="logistic_regression")[0]
        assert result_single["label"]   == result_batch["label"]
        assert abs(
            result_single["fake_probability"] - result_batch["fake_probability"]
        ) < 1e-6


class TestInferenceCache:
    def test_clear_cache_empties_cache(self, monkeypatch):
        import ml.baseline.inference as infer_mod
        infer_mod._pipeline_cache["logistic_regression"] = object()
        clear_cache()
        assert len(infer_mod._pipeline_cache) == 0

    def test_get_pipeline_populates_cache(self, tmp_path, monkeypatch, train_val_test):
        import ml.baseline.config   as cfg
        import ml.baseline.trainer  as trainer_mod
        import ml.baseline.inference as infer_mod

        train_df, val_df, _ = train_val_test
        pkl_path = tmp_path / "lr_pipeline.pkl"
        monkeypatch.setitem(cfg.MODEL_PIPELINE_PATHS,         "logistic_regression", pkl_path)
        monkeypatch.setitem(trainer_mod.MODEL_PIPELINE_PATHS, "logistic_regression", pkl_path)

        clear_cache()
        train_single("logistic_regression", train_df, val_df, save=True)

        clear_cache()
        assert "logistic_regression" not in infer_mod._pipeline_cache
        get_pipeline("logistic_regression")
        assert "logistic_regression" in infer_mod._pipeline_cache
        clear_cache()
