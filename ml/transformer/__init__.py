"""
ml.transformer
==============
Transformer-based fine-tuning pipeline for fake news detection.

Supported models
----------------
  DistilBERT : distilbert-base-uncased (default)
  BERT       : bert-base-uncased (via --base-model flag in train_transformer.py)
  RoBERTa    : roberta-base (via --base-model flag)

The same training, evaluation, and inference code works for all HuggingFace
AutoModelForSequenceClassification-compatible models.

Public re-exports
-----------------
  FakeNewsDataset           PyTorch Dataset for training/eval
  ChunkedArticleDataset     PyTorch Dataset for long-article inference
  train(...)                Fine-tune a transformer model
  predict_text(...)         Predict a single article (with chunking)
  predict_texts_batch(...)  Predict a list of articles
  load_model_and_tokenizer  Load (and cache) a model checkpoint
  evaluate(...)             Compute metrics on test set
  evaluate_from_csv(...)    Load test.csv and evaluate
  VERSIONED_MODEL_DIR       Path to the active model checkpoint
  BASE_MODEL                Default base model identifier
"""

from ml.transformer.dataset import (
    FakeNewsDataset,
    ChunkedArticleDataset,
    tokenize_to_ids,
    chunk_token_ids,
    encode_chunk,
)

from ml.transformer.trainer import (
    train,
    get_device,
)

from ml.transformer.inference import (
    predict_text,
    predict_texts_batch,
    load_model_and_tokenizer,
    aggregate_chunk_probs,
    clear_model_cache,
)

from ml.transformer.evaluator import (
    evaluate,
    evaluate_from_csv,
    print_evaluation_summary,
)

from ml.transformer.config import (
    BASE_MODEL,
    VERSIONED_MODEL_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_AGGREGATION,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    MAX_TOKEN_LENGTH,
    LABEL2ID,
    ID2LABEL,
)

__all__ = [
    "FakeNewsDataset", "ChunkedArticleDataset",
    "tokenize_to_ids", "chunk_token_ids", "encode_chunk",
    "train", "get_device",
    "predict_text", "predict_texts_batch", "load_model_and_tokenizer",
    "aggregate_chunk_probs", "clear_model_cache",
    "evaluate", "evaluate_from_csv", "print_evaluation_summary",
    "BASE_MODEL", "VERSIONED_MODEL_DIR",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_AGGREGATION",
    "NUM_EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "MAX_TOKEN_LENGTH",
    "LABEL2ID", "ID2LABEL",
]
