"""
ml/feature_extraction.py
========================
Feature extraction module.

For TF-IDF baseline models, feature extraction is embedded inside the
sklearn Pipeline (TfidfVectorizer is step 1 of every pipeline).
This module provides:

  1. A standalone TfidfVectorizer factory and fit/transform API — useful
     for analysis, feature inspection, and research outside of training.
  2. A vocabulary analyser that shows top features per class.
  3. A feature-importance extractor for Logistic Regression coefficients.
  4. Document-frequency statistics for inspecting what the vectorizer learned.

For transformer models, "feature extraction" means tokenization, which
is handled inside ml.transformer.dataset.  This module provides a
convenience wrapper for getting token sequences without the full dataset.

None of these functions train a model or modify saved artifacts.

Public API
----------
  TfidfFeatureExtractor           class  — standalone TF-IDF with fit/transform
  get_top_features(pipeline, n)   → dict  — top-N discriminating tokens per class
  get_feature_weights(pipeline)   → pd.DataFrame
  tokenize_for_transformer(text)  → dict  — input_ids, attention_mask as Python lists
  explain_tfidf_features(text, pipeline) → list[dict]  — per-token TF-IDF scores
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.baseline.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_ANALYZER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Standalone TF-IDF feature extractor
# =============================================================================

class TfidfFeatureExtractor:
    """
    Standalone TF-IDF vectorizer with the same hyperparameters as the
    baseline training pipelines.

    Useful for:
    - Inspecting what the vectorizer has learned
    - Generating feature matrices for external tools
    - Analysis and visualisation outside of model training

    This is SEPARATE from the sklearn Pipelines used in training.
    It does NOT affect saved model artifacts.

    Usage
    -----
    >>> extractor = TfidfFeatureExtractor()
    >>> extractor.fit(train_texts)
    >>> X = extractor.transform(texts)   # sparse matrix (n_docs, n_features)
    >>> extractor.get_feature_names()    # list of token strings
    """

    def __init__(
        self,
        max_features: int    = TFIDF_MAX_FEATURES,
        ngram_range:  tuple  = TFIDF_NGRAM_RANGE,
        sublinear_tf: bool   = TFIDF_SUBLINEAR_TF,
        min_df:       int    = TFIDF_MIN_DF,
        max_df:       float  = TFIDF_MAX_DF,
        analyzer:     str    = TFIDF_ANALYZER,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            analyzer=analyzer,
            strip_accents="unicode",
            lowercase=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfFeatureExtractor":
        """
        Fit the vectorizer on training texts.

        IMPORTANT: Only call this on TRAINING data.  Never pass val or test
        data here — that would cause feature leakage.
        """
        logger.info("Fitting TF-IDF vectorizer on %d texts…", len(texts))
        self.vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted: %d features in vocabulary", vocab_size)
        return self

    def transform(self, texts: list[str]):
        """
        Transform texts to a sparse TF-IDF feature matrix.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_docs, n_features)
        """
        self._check_fitted()
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        """Fit then transform (for training data only)."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return the list of token strings in vocabulary order."""
        self._check_fitted()
        return self.vectorizer.get_feature_names_out().tolist()

    def get_vocab_size(self) -> int:
        """Number of features (vocabulary size) after fitting."""
        self._check_fitted()
        return len(self.vectorizer.vocabulary_)

    def document_frequency(self) -> pd.DataFrame:
        """
        Return a DataFrame of (token, document_frequency) sorted by frequency.
        document_frequency = number of training documents containing the token.
        """
        self._check_fitted()
        # idf_ = log((1 + n_docs) / (1 + df)) + 1  (sklearn default)
        # Rearrange: df = (1 + n_docs) / exp(idf_ - 1) - 1
        n_docs = self.vectorizer.n_samples_seen_
        idf    = self.vectorizer.idf_
        df_arr = np.round((1 + n_docs) / np.exp(idf - 1) - 1).astype(int)
        names  = self.get_feature_names()
        result = pd.DataFrame({"token": names, "doc_freq": df_arr})
        return result.sort_values("doc_freq", ascending=False).reset_index(drop=True)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TfidfFeatureExtractor is not fitted. Call .fit(train_texts) first."
            )


# =============================================================================
# Feature importance from trained pipelines
# =============================================================================

def get_top_features(
    pipeline: Pipeline,
    n:        int = 20,
) -> dict[str, list[dict]]:
    """
    Extract the top-N most discriminating tokens per class from a fitted
    TF-IDF + Logistic Regression pipeline.

    Only works for Logistic Regression (has interpretable linear coefficients).
    For SVM and NB, use explain_tfidf_features() on individual documents.

    Parameters
    ----------
    pipeline : Fitted sklearn Pipeline with 'tfidf' and 'clf' steps.
               The classifier must be LogisticRegression or similar with
               a .coef_ attribute.
    n        : Number of top features per class.

    Returns
    -------
    {
        "FAKE": [{"token": str, "coefficient": float}, ...],
        "REAL": [{"token": str, "coefficient": float}, ...],
    }
    """
    tfidf = pipeline.named_steps.get("tfidf")
    clf   = pipeline.named_steps.get("clf")

    if tfidf is None or clf is None:
        raise ValueError("Pipeline must have 'tfidf' and 'clf' steps.")

    # Unwrap CalibratedClassifierCV if present
    base_clf = getattr(clf, "estimator", clf)

    if not hasattr(base_clf, "coef_"):
        raise TypeError(
            f"Classifier {type(base_clf).__name__} does not have .coef_ attribute. "
            "get_top_features() only works with linear classifiers (LogisticRegression, LinearSVC)."
        )

    feature_names = tfidf.get_feature_names_out()
    coefs         = base_clf.coef_[0]   # shape (n_features,)

    # Positive coef → FAKE, negative coef → REAL
    fake_indices = np.argsort(coefs)[::-1][:n]      # most positive
    real_indices = np.argsort(coefs)[:n]              # most negative

    return {
        "FAKE": [
            {"token": feature_names[i], "coefficient": round(float(coefs[i]), 6)}
            for i in fake_indices
        ],
        "REAL": [
            {"token": feature_names[i], "coefficient": round(float(coefs[i]), 6)}
            for i in real_indices
        ],
    }


def get_feature_weights(pipeline: Pipeline) -> pd.DataFrame:
    """
    Return a DataFrame of all features with their LR coefficients.

    Columns: token, coefficient, label (FAKE if coef > 0, REAL otherwise)

    Only works for Logistic Regression pipelines.
    """
    top = get_top_features(pipeline, n=len(pipeline.named_steps["tfidf"].vocabulary_))
    rows = []
    for label, features in top.items():
        for f in features:
            rows.append({"token": f["token"], "coefficient": f["coefficient"], "label": label})
    return pd.DataFrame(rows).sort_values("coefficient", ascending=False)


def explain_tfidf_features(
    text:     str,
    pipeline: Pipeline,
    top_n:    int = 20,
) -> list[dict]:
    """
    Show which tokens the TF-IDF vectorizer assigned the highest weight
    for a specific document.

    This is a document-level explanation: it shows TF-IDF *feature scores*
    (how term frequency × inverse document frequency scores each token),
    NOT the model's discriminative signal.

    For discriminative signal, use ml.analysis.explainer.explain_baseline_with_fallback().

    Parameters
    ----------
    text     : A single input text (cleaned or raw — the pipeline tfidf handles it).
    pipeline : Fitted sklearn Pipeline with a 'tfidf' step.
    top_n    : Number of top-scored tokens to return.

    Returns
    -------
    List of {"token": str, "tfidf_score": float} dicts, sorted by score desc.
    """
    tfidf = pipeline.named_steps.get("tfidf")
    if tfidf is None:
        raise ValueError("Pipeline must have a 'tfidf' step.")

    vec          = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    cx           = vec.tocsr().getrow(0)
    nz_indices   = cx.nonzero()[1]

    if len(nz_indices) == 0:
        return []

    scored = [(feature_names[i], float(cx[0, i])) for i in nz_indices]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        {"token": token, "tfidf_score": round(score, 6)}
        for token, score in scored[:top_n]
    ]


# =============================================================================
# Transformer tokenization utility
# =============================================================================

def tokenize_for_transformer(
    text:       str,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 512,
    return_tensors: Optional[str] = None,
) -> dict:
    """
    Tokenize a single text for a transformer model.

    Returns
    -------
    dict with keys:
        input_ids      : list[int]  — token IDs including [CLS] and [SEP]
        attention_mask : list[int]  — 1 for real tokens, 0 for padding
        tokens         : list[str]  — decoded token strings (for inspection)
        n_tokens       : int        — number of non-padding tokens
        was_truncated  : bool       — True if the text exceeded max_length
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers is required. Install with: pip install transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=return_tensors,
    )

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Convert to lists if tensors were returned
    if return_tensors is not None:
        input_ids      = input_ids[0].tolist()
        attention_mask = attention_mask[0].tolist()

    tokens         = tokenizer.convert_ids_to_tokens(input_ids)
    n_real_tokens  = sum(attention_mask)

    # Count original tokens without truncation to detect truncation
    original_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    was_truncated = len(original_ids) > max_length

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "tokens":         tokens,
        "n_tokens":       n_real_tokens,
        "was_truncated":  was_truncated,
    }
