"""
ml.baseline
===========
Traditional TF-IDF + sklearn baseline models.

Models
------
  logistic_regression : TF-IDF (50k features, 1-2 ngrams) + LR (C=1, balanced)
  linear_svm          : TF-IDF + LinearSVC wrapped in CalibratedClassifierCV
  naive_bayes         : TF-IDF + MultinomialNB (alpha=0.1)

Public re-exports
-----------------
  build_lr_pipeline()     → sklearn.pipeline.Pipeline
  build_svm_pipeline()    → sklearn.pipeline.Pipeline
  build_nb_pipeline()     → sklearn.pipeline.Pipeline
  train_single(...)       → (Pipeline, training_record_dict)
  train_all(...)          → {model_id: (Pipeline, record)}
  load_pipeline(model_id) → Pipeline
  predict_single(...)     → dict
  predict_all_models(...) → dict[model_id, dict]
  predict_batch(...)      → list[dict]
  evaluate_pipeline(...)  → dict
  evaluate_all(...)       → dict[model_id, dict]
  TF_IDF_CONFIG           → dict  (hyperparameters reference)
"""

from ml.baseline.trainer import (
    build_lr_pipeline,
    build_svm_pipeline,
    build_nb_pipeline,
    train_single,
    train_all,
    load_pipeline,
    save_pipeline,
)

from ml.baseline.inference import (
    predict_single,
    predict_all_models,
    predict_batch,
    get_pipeline,
    load_all_pipelines,
)

from ml.baseline.evaluator import (
    evaluate_pipeline,
    evaluate_all,
    print_evaluation_summary,
)

from ml.baseline.config import (
    MODEL_IDS,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    LR_C, SVM_C, NB_ALPHA,
)

TF_IDF_CONFIG = {
    "max_features":  TFIDF_MAX_FEATURES,
    "ngram_range":   TFIDF_NGRAM_RANGE,
    "sublinear_tf":  TFIDF_SUBLINEAR_TF,
}

__all__ = [
    "build_lr_pipeline", "build_svm_pipeline", "build_nb_pipeline",
    "train_single", "train_all", "load_pipeline", "save_pipeline",
    "predict_single", "predict_all_models", "predict_batch",
    "get_pipeline", "load_all_pipelines",
    "evaluate_pipeline", "evaluate_all", "print_evaluation_summary",
    "MODEL_IDS", "TF_IDF_CONFIG",
]
