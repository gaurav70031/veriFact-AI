"""
Baseline model configuration.

All hyperparameters, artifact paths, and model identifiers live here.
Training scripts read from this file; never hard-code values elsewhere.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ML_ROOT      = Path(__file__).resolve().parents[1]   # ml/
SAVED_MODELS = ML_ROOT / "saved_models"
SAVED_MODELS.mkdir(parents=True, exist_ok=True)

# ── Artifact paths ────────────────────────────────────────────────────────────
TFIDF_VECTORIZER_PATH = SAVED_MODELS / "tfidf_vectorizer.pkl"
LR_PIPELINE_PATH      = SAVED_MODELS / "lr_pipeline.pkl"
SVM_PIPELINE_PATH     = SAVED_MODELS / "svm_pipeline.pkl"
NB_PIPELINE_PATH      = SAVED_MODELS / "nb_pipeline.pkl"

# Evaluation results (machine-readable JSON)
EVAL_RESULTS_DIR  = SAVED_MODELS / "eval"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LR_EVAL_PATH  = EVAL_RESULTS_DIR / "lr_eval.json"
SVM_EVAL_PATH = EVAL_RESULTS_DIR / "svm_eval.json"
NB_EVAL_PATH  = EVAL_RESULTS_DIR / "nb_eval.json"
ALL_EVAL_PATH = EVAL_RESULTS_DIR / "all_models_eval.json"

# ── Model identifiers (used as dict keys and artifact names) ──────────────────
MODEL_IDS = ["logistic_regression", "linear_svm", "naive_bayes"]

MODEL_EVAL_PATHS = {
    "logistic_regression": LR_EVAL_PATH,
    "linear_svm":          SVM_EVAL_PATH,
    "naive_bayes":         NB_EVAL_PATH,
}

MODEL_PIPELINE_PATHS = {
    "logistic_regression": LR_PIPELINE_PATH,
    "linear_svm":          SVM_PIPELINE_PATH,
    "naive_bayes":         NB_PIPELINE_PATH,
}

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── TF-IDF hyperparameters ────────────────────────────────────────────────────
# Fit ONLY on training data — never on val or test.
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)      # unigrams + bigrams
TFIDF_SUBLINEAR_TF = True        # apply 1 + log(tf)
TFIDF_MIN_DF       = 2           # ignore tokens that appear in < 2 docs
TFIDF_MAX_DF       = 0.95        # ignore tokens in > 95 % of docs
TFIDF_ANALYZER     = "word"

# ── Logistic Regression ───────────────────────────────────────────────────────
LR_C           = 1.0
LR_MAX_ITER    = 1000
LR_SOLVER      = "lbfgs"
LR_CLASS_WEIGHT = "balanced"     # handle class imbalance automatically

# ── Linear SVM ────────────────────────────────────────────────────────────────
# LinearSVC does not natively output probabilities.
# Wrapped in CalibratedClassifierCV (cv=5) so predict_proba() is available.
SVM_C            = 1.0
SVM_MAX_ITER     = 2000
SVM_CLASS_WEIGHT = "balanced"
SVM_CALIBRATION_CV = 5

# ── Naive Bayes ───────────────────────────────────────────────────────────────
# MultinomialNB requires non-negative features → use TF-IDF (always >= 0).
NB_ALPHA = 0.1    # Laplace / Lidstone smoothing

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL2ID = {"FAKE": 0, "REAL": 1}
ID2LABEL = {0: "FAKE", 1: "REAL"}
