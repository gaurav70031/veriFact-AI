"""
ML Model Registry.

Responsibilities
----------------
* Discover and load trained artefact files from ml/saved_models/.
* Provide a unified predict() interface to the service layer.
* Cache models in memory so they are loaded once at startup, not per-request.
* Never train, never modify artefacts.
* Raise ModelUnavailableError with a clear message when an artefact is missing,
  so the API can return HTTP 503 instead of a silent failure.

Design
------
The registry holds three baseline pipelines (LR, SVM, NB) and one
transformer (DistilBERT).  Callers request predictions by model_id string.
All inference calls are synchronous and run in a thread-pool executor when
called from async FastAPI routes (see analysis_service.py).

Model IDs
---------
  "logistic_regression"   TF-IDF + Logistic Regression
  "linear_svm"            TF-IDF + LinearSVC (calibrated)
  "naive_bayes"           TF-IDF + MultinomialNB
  "distilbert"            Fine-tuned DistilBERT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

settings = get_settings()

# Add project root to sys.path so ml.* packages are importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Lazy imports from ml package (only available after training) ──────────────
def _import_baseline_inference():
    try:
        from ml.baseline.inference import predict_single, predict_batch, load_all_pipelines
        return predict_single, predict_batch, load_all_pipelines
    except ImportError as e:
        raise ModelUnavailableError(
            "Baseline ML package not found. Ensure ml/ is on the Python path.",
            detail=str(e),
        )


def _import_transformer_inference():
    try:
        from ml.transformer.inference import predict_text, predict_texts_batch, load_model_and_tokenizer
        return predict_text, predict_texts_batch, load_model_and_tokenizer
    except ImportError as e:
        raise ModelUnavailableError(
            "Transformer ML package not found. Ensure ml/ is on the Python path.",
            detail=str(e),
        )


# ── Registry singleton ────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Singleton that owns all loaded model objects.

    Call ModelRegistry.initialise() once during FastAPI lifespan startup.
    Then use ModelRegistry.predict(model_id, text) anywhere.
    """

    _instance: Optional["ModelRegistry"] = None

    def __init__(self) -> None:
        self._baseline_ready:    bool = False
        self._transformer_ready: bool = False
        self._transformer_dir:   Optional[Path] = None

        self._baseline_ids = [
            "logistic_regression",
            "linear_svm",
            "naive_bayes",
        ]

    # ── Startup loading ───────────────────────────────────────────────────────

    def load_baseline_models(self) -> None:
        """Load all three baseline sklearn pipelines into memory."""
        try:
            predict_single, predict_batch, load_all = _import_baseline_inference()
            load_all()   # populates ml.baseline.inference._pipeline_cache
            self._baseline_ready = True
            logger.info("Baseline models loaded successfully.")
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load baseline models. Run scripts/train_baseline.py first.",
                detail=str(exc),
            )

    def load_transformer_model(self) -> None:
        """Load the DistilBERT model and tokenizer into memory."""
        try:
            predict_text, _, load_fn = _import_transformer_inference()
            from ml.transformer.config import VERSIONED_MODEL_DIR
            self._transformer_dir = VERSIONED_MODEL_DIR

            if not VERSIONED_MODEL_DIR.exists():
                raise ModelUnavailableError(
                    f"DistilBERT checkpoint not found at {VERSIONED_MODEL_DIR}. "
                    "Run scripts/train_transformer.py first.",
                )
            load_fn(VERSIONED_MODEL_DIR)   # populates ml.transformer.inference cache
            self._transformer_ready = True
            logger.info("Transformer model loaded from %s.", VERSIONED_MODEL_DIR)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load transformer model.",
                detail=str(exc),
            )

    def initialise(self, require_transformer: bool = False) -> None:
        """
        Load all available models at startup.

        Parameters
        ----------
        require_transformer : If True, startup fails when the transformer
                              checkpoint is missing.  Default False allows
                              the app to run baseline-only.
        """
        errors: list[str] = []

        try:
            self.load_baseline_models()
        except ModelUnavailableError as e:
            errors.append(f"Baseline: {e.message}")
            logger.warning("Baseline models unavailable: %s", e.message)

        try:
            self.load_transformer_model()
        except ModelUnavailableError as e:
            if require_transformer:
                raise
            errors.append(f"Transformer: {e.message}")
            logger.warning("Transformer model unavailable: %s", e.message)

        if errors:
            logger.warning(
                "ModelRegistry started with %d unavailable model(s): %s",
                len(errors), "; ".join(errors),
            )
        else:
            logger.info("All models loaded and ready.")

    # ── Prediction interface ──────────────────────────────────────────────────

    def predict_baseline(self, text: str, model_id: str) -> dict:
        """
        Run a single baseline model and return a structured prediction dict.

        Returns
        -------
        {label, is_fake, confidence, fake_probability, real_probability,
         model_id, inference_time_ms}
        """
        if not self._baseline_ready:
            raise ModelUnavailableError(
                f"Baseline model '{model_id}' is not loaded. "
                "Run scripts/train_baseline.py and restart the server."
            )
        if model_id not in self._baseline_ids:
            raise ModelUnavailableError(
                f"Unknown baseline model_id '{model_id}'. "
                f"Valid IDs: {self._baseline_ids}"
            )

        predict_single, _, _ = _import_baseline_inference()
        try:
            result = predict_single(text, model_id=model_id, clean=True)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Baseline model '{model_id}' inference failed.",
                detail=str(exc),
            )
        return result

    def predict_all_baseline(self, text: str) -> dict[str, dict]:
        """Run all three baseline models. Returns {model_id: prediction}."""
        if not self._baseline_ready:
            raise ModelUnavailableError(
                "Baseline models are not loaded. "
                "Run scripts/train_baseline.py and restart the server."
            )
        from ml.baseline.inference import predict_all_models
        try:
            return predict_all_models(text, clean=True)
        except Exception as exc:
            raise ModelUnavailableError(
                "Baseline model inference failed.", detail=str(exc)
            )

    def predict_transformer(self, text: str) -> dict:
        """
        Run DistilBERT on a text (handles long-article chunking internally).

        Returns
        -------
        {label, is_fake, confidence, fake_probability, real_probability,
         num_chunks, aggregation_strategy, inference_time_ms}
        """
        if not self._transformer_ready:
            raise ModelUnavailableError(
                "Transformer model is not loaded. "
                "Run scripts/train_transformer.py and restart the server."
            )
        predict_text, _, _ = _import_transformer_inference()
        try:
            return predict_text(text, model_dir=self._transformer_dir)
        except Exception as exc:
            raise ModelUnavailableError(
                "Transformer model inference failed.", detail=str(exc)
            )

    # ── Status reporting ──────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "baseline_ready":    self._baseline_ready,
            "transformer_ready": self._transformer_ready,
            "available_models":  (
                self._baseline_ids + (["distilbert"] if self._transformer_ready else [])
            ),
        }

    # ── Singleton accessor ────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Module-level convenience accessor
def get_registry() -> ModelRegistry:
    return ModelRegistry.get()
