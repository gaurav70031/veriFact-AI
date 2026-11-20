# VeriFact AI — Machine Learning Engine

## Overview

This directory contains a complete fake news classification engine using NLP, traditional ML, and deep learning. All predictions come from real trained model weights — no keyword matching, no simulated outputs, no LLM-generated verdicts.

---

## Model Architecture

### Traditional ML (Baseline)

Three sklearn Pipelines, each bundling TF-IDF vectorization with a classifier:

| Model ID | Algorithm | Notes |
|---|---|---|
| `logistic_regression` | TF-IDF + Logistic Regression | C=1.0, balanced class weights, lbfgs solver |
| `linear_svm` | TF-IDF + LinearSVC | C=1.0, Platt-scaled via CalibratedClassifierCV (cv=5) for `predict_proba` |
| `naive_bayes` | TF-IDF + MultinomialNB | alpha=0.1 Laplace smoothing |

**TF-IDF configuration:**
- `max_features=50,000` — top 50k by TF-IDF score
- `ngram_range=(1,2)` — unigrams and bigrams
- `sublinear_tf=True` — apply `1 + log(tf)` for sublinear scaling
- `min_df=2`, `max_df=0.95` — filter hapax legomena and stop-word-like tokens
- Fitted **only** on the training split — vectorizer and classifier share the same Pipeline object to prevent leakage

### Transformer Models

| Model | Base | Parameters | Chunking |
|---|---|---|---|
| DistilBERT | `distilbert-base-uncased` | 66M | 510-token windows, 50-token overlap |
| BERT | `bert-base-uncased` | 110M | Same chunking |
| Any HuggingFace model | `--base-model <id>` | varies | Same chunking |

**Long-article chunking (key design decision):**

News articles regularly exceed DistilBERT's 512-token limit. Rather than truncating:
1. Tokenise the full article (no special tokens)
2. Slide a 510-token window with 50-token overlap
3. Each chunk gets `[CLS]/[SEP]` added, padded to 512
4. Run forward pass on all chunks in one batched DataLoader
5. Aggregate chunk probabilities using **weighted mean**: `weight_i = softmax(max_confidence_per_chunk)`
   - Chunks the model is more confident about receive proportionally more weight
   - Alternative strategies: `max_confidence`, `majority_vote` (selectable via config)

**Why DistilBERT as default?**
- 40% fewer parameters than BERT-base (66M vs 110M)
- 97% of BERT's GLUE performance
- Fits in 4–8 GB VRAM without mixed precision; runs on CPU for demonstration
- Same code works for BERT, RoBERTa — just change `BASE_MODEL` in config

---

## Preprocessing Pipeline

### Baseline mode (TF-IDF models)

Heavy normalisation applied before TF-IDF vectorization:

1. Decode HTML entities (`&amp;` → `&`)
2. Strip HTML/XML tags
3. Remove Reuters dateline (`(REUTERS) —`)
4. Remove URLs
5. Remove email addresses
6. Unicode NFKC normalisation + smart quote/dash replacement
7. **Lowercase**
8. Expand contractions (`don't` → `do not`)
9. Remove non-alphabetic characters (keep spaces)
10. Compress repeated characters (`looool` → `lool`, max 3 repeats)
11. Collapse whitespace
12. Drop if `len(text) < 20` characters after cleaning

**Consistency guarantee:** The same `clean_for_baseline()` function is used both during training (in `pipeline.py`) and during inference (in `inference.py`). The sklearn `Pipeline` object wraps TF-IDF + classifier together — they are always saved and loaded as a unit.

### Transformer mode (DistilBERT/BERT)

Light normalisation that preserves natural language structure:

1. Decode HTML entities
2. Strip HTML/XML tags
3. Remove Reuters dateline
4. Remove URLs / emails
5. Unicode NFKC normalisation
6. Collapse whitespace

**Casing and punctuation preserved** — DistilBERT is case-insensitive (`-uncased` variant) but its pre-trained tokenizer handles subwords naturally and benefits from punctuation patterns.

---

## Explainability

### LIME (Local Interpretable Model-Agnostic Explanations)

**Used for:** TF-IDF baseline models (LR, SVM, NB)

**Method:**
- Perturbs the input text by randomly masking words (1,000 perturbation samples by default)
- Fits a local linear model to observe which words, when removed, change the prediction
- Returns signed weights: **positive = toward FAKE, negative = toward REAL**
- Normalised to `[-1, +1]`

**Why LIME and not SHAP?**
- SHAP's KernelExplainer is O(2^n) in the worst case; LIME is O(n_samples × n_features)
- LIME integrates directly with `predict_proba` that all three baseline pipelines already expose
- For text with 100+ words, LIME is 10–100× faster than KernelExplainer

**Fallback:** When LIME is unavailable (not installed), the module falls back to raw TF-IDF feature weights for the document — less interpretable but always available.

### Attention Rollout

**Used for:** DistilBERT / BERT

**Method:**
1. Single forward pass with `output_attentions=True`
2. Take the **last transformer layer** attention matrix `(1, n_heads, seq_len, seq_len)`
3. Mean over all heads → `(seq_len, seq_len)`
4. Take the CLS row (how the classification token attends to each other token)
5. Zero out special token positions (`[CLS]`, `[SEP]`, `[PAD]`)
6. Aggregate sub-word tokens (`##suffix`) back to whole words by taking `max` across sub-words
7. L1-normalise scores to sum = 1
8. Sign assignment: tokens the model attends to above-average get the sign of the predicted label

**Reference:** Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers."

### Critical disclaimer (always shown to users)

> These highlighted words show which parts of the text the model focused on when making its prediction. This reflects **statistical patterns** learned during training — it does **NOT** indicate that these words are factually incorrect or prove the article is fake.

---

## Prediction System

### Confidence threshold

Default: `0.55` (5 percentage points above chance — conservative)

Override via environment variable:
```bash
export VERIFACT_CONFIDENCE_THRESHOLD=0.65
```

When model confidence < threshold:
- Label is set to `UNVERIFIED` (not FAKE or REAL)
- `below_threshold: true` is included in the response
- A specific limitation note is added

### Ensemble weights

The backend combines all available models using weighted average probabilities:

| Model | Weight | Rationale |
|---|---|---|
| Logistic Regression | 0.20 | Interpretable baseline |
| Linear SVM | 0.30 | Strongest traditional classifier on ISOT |
| Naive Bayes | 0.10 | Fast, independent signal |
| DistilBERT | 0.40 | Highest accuracy; contextual understanding |

Weights sum to 1.0. If a model is unavailable (not trained), remaining weights are renormalised.

### MIXED verdict

When ensemble confidence is `< 0.55` (models disagree), the verdict is `MIXED` — not FAKE or REAL. This is better than forcing a low-confidence binary classification.

---

## Training Dataset

### ISOT Fake News Dataset

| Property | Value |
|---|---|
| Total articles | 44,898 |
| Fake articles | 23,481 (52.3%) |
| Real articles | 21,417 (47.7%) |
| Year range | 2015–2018 |
| Real source | Reuters.com |
| Fake sources | PolitiFact lists, Infowars, others |
| Citation | Ahmed, H., Traore, I., Saad, S. (2018). *Journal of Security and Privacy* |

**Download:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Splits (stratified, reproducible)

| Split | Fraction | Articles |
|---|---|---|
| Train | 80% | ~35,900 |
| Validation | 10% | ~4,490 |
| Test (held-out) | 10% | ~4,490 |

`random_state=42` throughout. Leakage check runs automatically after splitting.

### Known limitations

- Models trained on 2015–2018 US political news — may not generalise to scientific claims, non-English content, or events after 2018
- Real articles come exclusively from Reuters; fake from curated lists — models may learn source-domain features rather than factual misinformation patterns
- High accuracy on ISOT ≠ high accuracy on arbitrary new content

---

## Module Reference

### Top-level modules (public API)

| File | Purpose |
|---|---|
| `ml/data_loader.py` | `load_raw()`, `load_processed()`, `prepare_if_needed()` |
| `ml/preprocessing_api.py` | `clean_text()`, `clean_texts()`, `clean_for_inference()` |
| `ml/feature_extraction.py` | `TfidfFeatureExtractor`, `get_top_features()`, `tokenize_for_transformer()` |
| `ml/predict.py` | `predict()`, `predict_ensemble()`, `predict_batch()`, `predict_url()` |
| `ml/evaluation.py` | `evaluate_baseline()`, `evaluate_transformer()`, `load_saved_results()` |
| `ml/__init__.py` | Lazy re-exports of all public functions |

### Internal subpackages

| Package | Key files |
|---|---|
| `ml/preprocessing/` | `cleaner.py`, `loader.py`, `labels.py`, `deduplication.py`, `splitter.py`, `pipeline.py` |
| `ml/baseline/` | `trainer.py`, `inference.py`, `evaluator.py`, `config.py` |
| `ml/transformer/` | `trainer.py`, `dataset.py`, `inference.py`, `evaluator.py`, `config.py` |

### Training scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_dataset.py` | Load, clean, deduplicate, split ISOT → CSV |
| `scripts/train_baseline.py` | Train LR, SVM, NB and save `.pkl` artifacts |
| `scripts/evaluate_baseline.py` | Evaluate on test set → `eval/*.json` |
| `scripts/train_transformer.py` | Fine-tune DistilBERT (default) |
| `scripts/train_bert.py` | Fine-tune BERT or any HuggingFace model |
| `scripts/evaluate_transformer.py` | Evaluate transformer on test set |
| `scripts/register_models.py` | Seed evaluation metrics into PostgreSQL |

---

## Quick Start

### 1. Install dependencies

```bash
# CPU training
pip install -r ml/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU training (CUDA 12.x)
pip install -r ml/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download dataset

Download `Fake.csv` and `True.csv` from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place them in `ml/datasets/raw/`.

### 3. Preprocess

```bash
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv
```

Output: `ml/datasets/processed/train.csv`, `val.csv`, `test.csv`, `dataset_stats.json`

### 4. Train baseline models (~2 min on CPU)

```bash
python scripts/train_baseline.py
```

Output: `ml/saved_models/lr_pipeline.pkl`, `svm_pipeline.pkl`, `nb_pipeline.pkl`

### 5. Evaluate baseline models

```bash
python scripts/evaluate_baseline.py
```

Output: `ml/saved_models/eval/all_models_eval.json`

### 6. Train DistilBERT (GPU recommended — ~15–30 min/epoch on T4)

```bash
python scripts/train_transformer.py --epochs 3
```

Output: `ml/saved_models/distilbert/v1.0.0/`

### 7. Evaluate DistilBERT

```bash
python scripts/evaluate_transformer.py
```

Output: `ml/saved_models/eval/distilbert_eval.json`

### 8. Register metrics in database

```bash
python scripts/register_models.py
```

This populates the `model_versions` table with real accuracy/F1/ROC-AUC so the frontend Model Laboratory page shows actual evaluation results.

### 9. Run a prediction

```bash
python -c "
from ml.predict import predict_ensemble
result = predict_ensemble('Scientists claim that vaccines cause autism.')
print('Label:      ', result['label'])
print('Confidence: ', f\"{result['confidence']:.1%}\")
print('Fake prob:  ', f\"{result['fake_probability']:.1%}\")
print('Real prob:  ', f\"{result['real_probability']:.1%}\")
"
```

---

## Artifact Directory Layout

```
ml/saved_models/
  lr_pipeline.pkl                    TF-IDF + LR (sklearn Pipeline)
  svm_pipeline.pkl                   TF-IDF + SVM (sklearn Pipeline)
  nb_pipeline.pkl                    TF-IDF + NB  (sklearn Pipeline)
  logistic_regression_training_record.json
  linear_svm_training_record.json
  naive_bayes_training_record.json
  distilbert/
    v1.0.0/
      config.json                    HuggingFace model config
      pytorch_model.bin              Model weights (or model.safetensors)
      tokenizer/
        tokenizer_config.json
        vocab.txt
        tokenizer.json
      training_record.json           Hyperparams + per-epoch metrics
  eval/
    lr_eval.json                     Baseline evaluation metrics
    svm_eval.json
    nb_eval.json
    all_models_eval.json             Combined baseline metrics
    distilbert_eval.json             Transformer evaluation metrics
```

---

## Configuration

All hyperparameters are in module-level `config.py` files and can be overridden via environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `VERIFACT_CONFIDENCE_THRESHOLD` | `0.55` | Below this → `UNVERIFIED` verdict |
| `TRANSFORMER_VERSION` | `1.0.0` | Which model version the backend loads |
| `TRANSFORMER_BATCH_SIZE` | `16` | Training batch size |
| `TRANSFORMER_EPOCHS` | `3` | Training epochs |
| `TRANSFORMER_LR` | `2e-5` | Learning rate |

---

## Evaluation Methodology

All metrics are computed on the **held-out 10% test set** that was:
- Never seen during training
- Never seen during hyperparameter selection (which uses the validation set)
- Stratified to maintain class balance

Metrics reported:
- **Accuracy** — fraction of correct predictions
- **Precision** (macro) — TP/(TP+FP), averaged across classes
- **Recall** (macro) — TP/(TP+FN), averaged across classes
- **F1 Score** (macro) — harmonic mean of precision and recall
- **ROC-AUC** — area under the receiver operating characteristic curve
- **Confusion matrix** — [[TN, FP], [FN, TP]]
- **Inference time** — wall-clock ms/sample on the test set

> These are **offline evaluation metrics** on ISOT. They describe how the model performed on ISOT test data — not how it will perform on arbitrary real-world submissions. High ISOT accuracy does not guarantee correct classification of all inputs.

---

## Data Integrity Guarantees

1. **No leakage** — TF-IDF is fitted exclusively on training data inside the Pipeline object. Val/test sets are transformed only.
2. **Exact same preprocessing at training and inference** — `clean_for_baseline()` is called by both `pipeline.py` (training) and `inference.py` (inference).
3. **Deduplication** — exact and near-duplicate removal before splitting; cross-split leakage check after splitting.
4. **Reproducible splits** — `random_state=42` throughout; stratified on label.
5. **No hardcoded metrics** — every number displayed in the UI comes from a JSON file produced by running the actual models on actual test data.
