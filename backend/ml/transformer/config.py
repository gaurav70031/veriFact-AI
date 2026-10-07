"""
Transformer pipeline configuration.

Rationale for DistilBERT
------------------------
DistilBERT-base-uncased was chosen over full BERT-base because:
  * 40% fewer parameters (66M vs 110M) — faster training and inference
  * 97% of BERT's performance on GLUE benchmarks
  * Fits in 4–8 GB GPU VRAM without mixed precision; runs on CPU for demo
  * Pre-trained on Wikipedia + BookCorpus — strong general English understanding
  * Well-supported by HuggingFace Transformers

For production upgrade: swap BASE_MODEL to "bert-base-uncased" or
"roberta-base" without changing any other code.

Chunking strategy
-----------------
DistilBERT has a 512-token limit.  Long news articles regularly exceed this.
Rather than truncating (which discards the body of long articles), we:
  1. Split the cleaned text into overlapping 512-token chunks.
  2. Run inference on each chunk independently.
  3. Aggregate chunk probabilities using a weighted mean:
       weight = softmax(max_probability_per_chunk)
     Chunks where the model is more confident get higher weight.
  4. The final label = argmax(weighted_mean_probabilities).

This is documented in ml/transformer/inference.py and tested in
tests/ml/test_transformer.py.
"""

import os
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ML_ROOT      = Path(__file__).resolve().parents[1]
SAVED_MODELS = ML_ROOT / "saved_models"
SAVED_MODELS.mkdir(parents=True, exist_ok=True)

TRANSFORMER_MODEL_DIR = SAVED_MODELS / "distilbert"
EVAL_RESULTS_DIR      = SAVED_MODELS / "eval"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMER_EVAL_PATH = EVAL_RESULTS_DIR / "distilbert_eval.json"

# ── Model versioning ──────────────────────────────────────────────────────────
# Each training run creates a subdirectory: distilbert/v{VERSION}/
# The "best" symlink (or config entry) points to the active version.
MODEL_VERSION = os.getenv("TRANSFORMER_VERSION", "1.0.0")
VERSIONED_MODEL_DIR = TRANSFORMER_MODEL_DIR / f"v{MODEL_VERSION}"

# ── Base model ────────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"   # HuggingFace model hub identifier

# ── Label mapping ─────────────────────────────────────────────────────────────
NUM_LABELS = 2
LABEL2ID   = {"FAKE": 0, "REAL": 1}
ID2LABEL   = {0: "FAKE", 1: "REAL"}

# ── Tokenization ──────────────────────────────────────────────────────────────
MAX_TOKEN_LENGTH = 512        # DistilBERT hard limit
TOKENIZER_PADDING   = True
TOKENIZER_TRUNCATION = True   # Applied per chunk — not to the full article

# ── Chunking ──────────────────────────────────────────────────────────────────
# CHUNK_SIZE: number of tokens per chunk (leave room for [CLS] and [SEP])
CHUNK_SIZE    = 510           # 512 - 2 special tokens
# CHUNK_OVERLAP: token overlap between consecutive chunks
# Overlap prevents boundary sentences from being split across two chunks
# without context.  50 tokens ≈ 2–3 sentences.
CHUNK_OVERLAP = 50
# Minimum tokens a chunk must contain to be sent through the model.
# Prevents near-empty trailing chunks from skewing aggregation.
MIN_CHUNK_TOKENS = 10

# Aggregation strategy: "weighted_mean" | "max_confidence" | "majority_vote"
# weighted_mean: softmax(max_prob_per_chunk) as weights — default
# max_confidence: use only the single highest-confidence chunk prediction
# majority_vote: count chunk votes, break ties by probability mean
CHUNK_AGGREGATION = "weighted_mean"

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE       = int(os.getenv("TRANSFORMER_BATCH_SIZE",  "16"))
NUM_EPOCHS       = int(os.getenv("TRANSFORMER_EPOCHS",      "3"))
LEARNING_RATE    = float(os.getenv("TRANSFORMER_LR",        "2e-5"))
WARMUP_RATIO     = 0.1        # fraction of total steps used for LR warmup
WEIGHT_DECAY     = 0.01
GRADIENT_CLIP    = 1.0
EVAL_BATCH_SIZE  = 32         # larger batch for eval (no gradient storage)
DATALOADER_WORKERS = 0        # 0 = use main process (safe on Windows)

# ── Early stopping ────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 2   # stop if val_loss doesn't improve for N epochs
SAVE_BEST_ONLY          = True

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Mixed precision ───────────────────────────────────────────────────────────
# Set to True only when a CUDA GPU is available — CPU training always uses FP32
USE_FP16 = False   # overridden at runtime in trainer.py if GPU detected
