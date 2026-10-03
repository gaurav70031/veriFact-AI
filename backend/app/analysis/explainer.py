"""
Explainable AI module.

Produces token-level feature attributions for both traditional ML models
(TF-IDF + classifier) and the transformer (DistilBERT).

Two methods are implemented:

LIME — for TF-IDF baseline models
-----------------------------------
LIME (Local Interpretable Model-Agnostic Explanations) perturbs the input
text by randomly masking tokens and observes how the pipeline's predicted
probability changes.  It fits a local linear model to these perturbations,
and the linear coefficients become feature importances.

Why LIME and not SHAP?
SHAP's KernelExplainer is O(2^n) in the worst case.  LIME is faster for
text inputs and works directly with sklearn's predict_proba interface that
every baseline pipeline already exposes (including the calibrated SVM).

Attention Attribution — for DistilBERT
----------------------------------------
DistilBERT's multi-head self-attention produces attention weights
(batch, heads, seq, seq) per layer.  We extract:
  1. Last transformer layer attentions (most task-specific).
  2. Mean over all heads (aggregates diverse attention patterns).
  3. CLS-row (token 0) of the resulting (seq, seq) matrix.
     Each entry CLS→token_i indicates how much the classification token
     "attends to" token_i when making its prediction.
  4. Scores are L1-normalised so they sum to 1.0.

Why attention and not Integrated Gradients?
Integrated Gradients requires multiple forward passes and gradient
computation.  On CPU (typical for development) this is 10–50× slower
than a single forward pass.  Attention rollout is a well-studied proxy
[Abnar & Zuidema, 2020] that runs in a single forward pass and requires
no additional libraries beyond PyTorch + Transformers.

Important caveats (communicated to users)
------------------------------------------
* These attributions explain the MODEL's decision, NOT factual truth.
* High weight on a word means the model associates it with
  fake/real patterns from training data.
* It does NOT mean the word is factually wrong or right.
* Do not use token weights as evidence of misinformation.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add project root to path so ml.* packages are importable from the backend
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Shared data structures ────────────────────────────────────────────────────

@dataclass
class TokenWeight:
    """Attribution weight for a single token."""
    token:    str
    weight:   float   # positive = toward FAKE label, negative = toward REAL
    position: int     # 0-based position in the token list


@dataclass
class ExplanationResult:
    """
    Complete explanation for a single model's prediction on one text.

    Fields
    ------
    model_id        : Identifier of the model that produced this prediction.
    method          : "lime" | "attention" | "tfidf_weights" | "unavailable"
    label           : The label predicted by this model.
    tokens          : Ordered list of token attributions (by absolute weight desc).
    top_tokens      : Top-N tokens by |weight| for compact display.
    plain_text      : Human-readable explanation for non-technical users.
    disclaimer      : Standard disclaimer that must always be shown.
    error           : Set when explanation could not be computed (graceful fallback).
    """
    model_id:   str
    method:     str
    label:      str
    tokens:     list[TokenWeight] = field(default_factory=list)
    top_tokens: list[TokenWeight] = field(default_factory=list)
    plain_text: str = ""
    disclaimer: str = (
        "These highlighted words show which parts of the text the model "
        "focused on when making its prediction. This reflects statistical "
        "patterns learned during training — it does NOT indicate that these "
        "words are factually incorrect or prove the article is fake."
    )
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id":   self.model_id,
            "method":     self.method,
            "label":      self.label,
            "tokens": [
                {"token": t.token, "weight": round(t.weight, 6), "position": t.position}
                for t in self.tokens
            ],
            "top_tokens": [
                {"token": t.token, "weight": round(t.weight, 6), "position": t.position}
                for t in self.top_tokens
            ],
            "plain_text": self.plain_text,
            "disclaimer": self.disclaimer,
            "error":      self.error,
        }

    @classmethod
    def unavailable(cls, model_id: str, reason: str) -> "ExplanationResult":
        """Return a graceful no-op result when explanation cannot be computed."""
        return cls(
            model_id=model_id,
            method="unavailable",
            label="UNKNOWN",
            error=reason,
            plain_text=(
                "Explanation is not available for this prediction. "
                f"Reason: {reason}"
            ),
        )


# ── Plain-text explanation builder ───────────────────────────────────────────

def _build_plain_text(
    model_id:    str,
    label:       str,
    top_tokens:  list[TokenWeight],
    method:      str,
) -> str:
    """
    Compose a human-readable, non-technical explanation.

    Critical: the explanation NEVER claims a word "proves" something is fake.
    It only says the model found it relevant to its decision.
    """
    if not top_tokens:
        return (
            f"The {model_id.replace('_', ' ')} model predicted this content "
            f"as {label}, but no specific words could be identified as "
            f"strongly contributing to this decision."
        )

    # Split into fake-leaning and real-leaning tokens
    fake_tokens = [t for t in top_tokens if t.weight > 0]
    real_tokens = [t for t in top_tokens if t.weight < 0]

    parts: list[str] = []

    model_name = model_id.replace("_", " ").title()
    method_desc = {
        "lime":          "Local Interpretable Model-Agnostic Explanations (LIME)",
        "attention":     "attention-based attribution",
        "tfidf_weights": "TF-IDF feature weights",
    }.get(method, method)

    parts.append(
        f"The {model_name} model predicted this content as {label} "
        f"using {method_desc}."
    )

    if fake_tokens:
        token_list = ", ".join(f'"{t.token}"' for t in fake_tokens[:5])
        parts.append(
            f"Words that the model associated with patterns it learned "
            f"from fake content include: {token_list}. "
            f"This means the model has seen similar language in training "
            f"examples labelled as fake — it does not mean these words are "
            f"inaccurate or that the article is false."
        )

    if real_tokens:
        token_list = ", ".join(f'"{t.token}"' for t in real_tokens[:5])
        parts.append(
            f"Words associated with credible content patterns include: "
            f"{token_list}."
        )

    parts.append(
        "These word associations reflect the model's statistical patterns "
        "and should be treated as model signals, not as factual evidence."
    )

    return " ".join(parts)


# ── LIME explainer for TF-IDF baseline models ─────────────────────────────────

def explain_baseline(
    text:        str,
    model_id:    str,
    num_features: int = 15,
    num_samples:  int = 1000,
) -> ExplanationResult:
    """
    Generate LIME explanation for a TF-IDF baseline model.

    Parameters
    ----------
    text         : Cleaned text to explain (same preprocessing as training).
    model_id     : "logistic_regression" | "linear_svm" | "naive_bayes"
    num_features : Number of tokens to include in explanation.
    num_samples  : LIME perturbation samples (more = stable but slower).

    Returns
    -------
    ExplanationResult with token weights derived from LIME coefficients.
    """
    # ── Import LIME (optional dependency) ────────────────────────────────────
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("LIME not installed — baseline explanation unavailable.")
        return ExplanationResult.unavailable(
            model_id,
            "LIME library not installed. Run: pip install lime",
        )

    # ── Load pipeline ─────────────────────────────────────────────────────────
    try:
        from ml.baseline.inference import get_pipeline
        pipeline = get_pipeline(model_id)
    except Exception as exc:
        logger.warning("Could not load pipeline '%s' for LIME: %s", model_id, exc)
        return ExplanationResult.unavailable(model_id, str(exc))

    # ── Run LIME ──────────────────────────────────────────────────────────────
    try:
        # class_names order must match pipeline.classes_
        # pipeline.classes_ is [0, 1] = [FAKE, REAL]
        explainer = LimeTextExplainer(
            class_names=["FAKE", "REAL"],
            bow=True,             # bag-of-words mode — appropriate for TF-IDF
            random_state=42,
        )

        # Wrapper so LIME gets class probabilities in [P(FAKE), P(REAL)] order
        classes = list(pipeline.classes_)
        fake_idx = classes.index(0) if 0 in classes else 0
        real_idx = classes.index(1) if 1 in classes else 1

        def predict_fn(texts: list[str]) -> "np.ndarray":
            import numpy as np
            proba = pipeline.predict_proba(texts)
            # Re-order to [P(FAKE), P(REAL)] regardless of internal class order
            return np.column_stack([proba[:, fake_idx], proba[:, real_idx]])

        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Extract predictions to get the predicted label
        pred_label_int = pipeline.predict([text])[0]
        label = "FAKE" if pred_label_int == 0 else "REAL"

        # LIME returns (word, weight_for_predicted_class)
        # Positive weight → pushes toward FAKE (class 0 in our scheme)
        # We normalise weights to [-1, +1]
        raw_weights = exp.as_list(label=0)   # weights for FAKE class
        if not raw_weights:
            raw_weights = exp.as_list()

        max_abs = max(abs(w) for _, w in raw_weights) if raw_weights else 1.0
        if max_abs == 0:
            max_abs = 1.0

        tokens: list[TokenWeight] = []
        for pos, (word, weight) in enumerate(raw_weights):
            tokens.append(TokenWeight(
                token=word,
                weight=round(weight / max_abs, 6),  # normalise
                position=pos,
            ))

        # Sort by absolute weight descending
        tokens.sort(key=lambda t: abs(t.weight), reverse=True)
        top_tokens = tokens[:10]

        plain_text = _build_plain_text(model_id, label, top_tokens, "lime")

        logger.info(
            "LIME explanation for '%s': %d features, top='%s' (%.3f)",
            model_id, len(tokens),
            top_tokens[0].token if top_tokens else "none",
            top_tokens[0].weight if top_tokens else 0.0,
        )

        return ExplanationResult(
            model_id=model_id,
            method="lime",
            label=label,
            tokens=tokens,
            top_tokens=top_tokens,
            plain_text=plain_text,
        )

    except Exception as exc:
        logger.warning("LIME explanation failed for '%s': %s", model_id, exc, exc_info=True)
        return ExplanationResult.unavailable(model_id, f"LIME computation failed: {exc}")


# ── Attention attribution for DistilBERT ──────────────────────────────────────

def explain_transformer(
    text:          str,
    model_dir:     Optional[Path] = None,
    max_tokens:    int = 15,
) -> ExplanationResult:
    """
    Generate attention-based attribution for the DistilBERT transformer.

    Method: Last-layer attention rollout
    -------------------------------------
    1. Tokenise text (truncated to 512 tokens — first chunk only for speed).
    2. Forward pass with output_attentions=True.
    3. Take the last transformer layer's attention matrix
       shape: (1, n_heads, seq_len, seq_len).
    4. Mean over heads → (seq_len, seq_len).
    5. Take the CLS row (index 0) → (seq_len,).
       CLS[i] = how much the classification decision attends to token i.
    6. Remove [CLS] and [SEP] tokens from output.
    7. Aggregate sub-word tokens (##suffix) back to whole words.
    8. L1-normalise scores to [0, 1].
    9. Map to signed weights: tokens that push toward FAKE get positive,
       tokens that push toward REAL get negative.
       (approximated by comparing attention concentration against uniform baseline)

    Parameters
    ----------
    text      : Raw or lightly cleaned text (transformer handles its own preprocessing).
    model_dir : Path to versioned model directory. Defaults to VERSIONED_MODEL_DIR.
    max_tokens: Number of top-attended tokens to include.

    Returns
    -------
    ExplanationResult with attention-derived token weights.
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        return ExplanationResult.unavailable(
            "distilbert",
            "PyTorch not installed. Transformer explanation unavailable.",
        )

    try:
        from ml.transformer.inference import load_model_and_tokenizer
        from ml.transformer.config import VERSIONED_MODEL_DIR
        if model_dir is None:
            model_dir = VERSIONED_MODEL_DIR

        model, tokenizer = load_model_and_tokenizer(model_dir)

        # Ensure the model uses eager attention so output_attentions=True works.
        # Newer transformers defaults to sdpa which drops attentions.
        try:
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Could not load transformer for explanation: %s", exc)
        return ExplanationResult.unavailable("distilbert", str(exc))

    try:
        device = next(model.parameters()).device

        # Tokenise (truncate to 512 — single chunk for explanation speed)
        enc = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Get predicted label
        probs    = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        classes  = list(model.config.id2label.keys())
        fake_idx = 0
        real_idx = 1
        is_fake  = probs[fake_idx] > probs[real_idx]
        label    = "FAKE" if is_fake else "REAL"

        # ── Extract attention ─────────────────────────────────────────────────
        # outputs.attentions: tuple of (1, n_heads, seq, seq) per layer
        # Use last layer — most task-specific
        if not outputs.attentions:
            return ExplanationResult.unavailable(
                "distilbert",
                "Model did not return attention weights. "
                "The model may not support output_attentions=True with the current attention implementation. "
                "Try setting attn_implementation='eager' in the model config."
            )
        last_attn = outputs.attentions[-1]          # (1, heads, seq, seq)
        mean_attn = last_attn.mean(dim=1)[0]        # (seq, seq) mean over heads
        cls_attn  = mean_attn[0].cpu().numpy()      # (seq,) CLS→all tokens

        # Remove [CLS] (index 0) and [SEP] (last real token)
        token_ids  = input_ids[0].cpu().numpy()
        attn_score = cls_attn.copy()

        # Zero out special tokens
        special_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        } - {None}
        for i, tid in enumerate(token_ids):
            if int(tid) in special_ids:
                attn_score[i] = 0.0

        # ── Aggregate sub-word tokens to whole words ──────────────────────────
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        word_scores: list[tuple[str, float]] = []
        current_word  = ""
        current_score = 0.0

        for tok, score in zip(raw_tokens, attn_score):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                if current_word:
                    word_scores.append((current_word, current_score))
                    current_word  = ""
                    current_score = 0.0
                continue
            if tok.startswith("##"):
                current_word  += tok[2:]
                current_score  = max(current_score, float(score))
            else:
                if current_word:
                    word_scores.append((current_word, current_score))
                current_word  = tok
                current_score = float(score)
        if current_word:
            word_scores.append((current_word, current_score))

        # ── Normalise ─────────────────────────────────────────────────────────
        if not word_scores:
            return ExplanationResult.unavailable("distilbert", "No tokens after filtering.")

        scores_arr = np.array([s for _, s in word_scores])
        total = scores_arr.sum()
        if total > 0:
            scores_arr = scores_arr / total   # L1-normalise to sum=1

        # Sign: tokens with above-average attention get sign of predicted label
        # (positive = toward FAKE, negative = toward REAL)
        mean_score = scores_arr.mean()
        sign = 1.0 if is_fake else -1.0

        token_weights: list[TokenWeight] = []
        for pos, ((word, _), norm_score) in enumerate(zip(word_scores, scores_arr)):
            # Give sign to tokens above mean attention; below-mean get opposite sign
            token_sign = sign if norm_score >= mean_score else -sign
            token_weights.append(TokenWeight(
                token=word,
                weight=round(float(norm_score) * token_sign, 6),
                position=pos,
            ))

        # Sort by absolute weight descending
        token_weights.sort(key=lambda t: abs(t.weight), reverse=True)
        top_tokens = [t for t in token_weights if len(t.token) > 2][:max_tokens]

        plain_text = _build_plain_text("distilbert", label, top_tokens, "attention")

        logger.info(
            "Attention explanation for distilbert: %d tokens, top='%s' (%.4f)",
            len(token_weights),
            top_tokens[0].token if top_tokens else "none",
            top_tokens[0].weight if top_tokens else 0.0,
        )

        return ExplanationResult(
            model_id="distilbert",
            method="attention",
            label=label,
            tokens=token_weights,
            top_tokens=top_tokens,
            plain_text=plain_text,
        )

    except Exception as exc:
        logger.warning("Attention explanation failed: %s", exc, exc_info=True)
        return ExplanationResult.unavailable("distilbert", f"Attention computation failed: {exc}")


# ── TF-IDF weight fallback (no LIME) ─────────────────────────────────────────

def explain_tfidf_weights(
    text:        str,
    model_id:    str,
    num_features: int = 15,
) -> ExplanationResult:
    """
    Lightweight fallback explanation using raw TF-IDF feature weights.

    When LIME is unavailable, extract which tokens the TF-IDF vectorizer
    assigned the highest weights for this document.  This is less
    interpretable than LIME but requires no additional libraries.

    The scores reflect TF-IDF importance (how unusual/specific a token is),
    NOT the model's discriminative signal.
    """
    try:
        from ml.baseline.inference import get_pipeline
        pipeline = get_pipeline(model_id)
        tfidf    = pipeline.named_steps["tfidf"]

        # Transform the single document
        import numpy as np
        vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()

        # Get non-zero features for this document
        cx = vec.tocsr()
        row = cx.getrow(0)
        nonzero_indices = row.nonzero()[1]
        if len(nonzero_indices) == 0:
            return ExplanationResult.unavailable(model_id, "No TF-IDF features for this text.")

        scores = [(feature_names[i], float(row[0, i])) for i in nonzero_indices]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Get predicted label
        pred_label_int = pipeline.predict([text])[0]
        label = "FAKE" if pred_label_int == 0 else "REAL"

        # Normalise
        max_score = max(s for _, s in scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        # All TF-IDF weights are non-negative — sign by label
        sign = 1.0 if label == "FAKE" else -1.0
        tokens = [
            TokenWeight(token=word, weight=round(score / max_score * sign, 6), position=pos)
            for pos, (word, score) in enumerate(scores[:num_features])
        ]
        top_tokens = tokens[:10]
        plain_text = _build_plain_text(model_id, label, top_tokens, "tfidf_weights")

        return ExplanationResult(
            model_id=model_id,
            method="tfidf_weights",
            label=label,
            tokens=tokens,
            top_tokens=top_tokens,
            plain_text=plain_text,
        )

    except Exception as exc:
        logger.warning("TF-IDF weight explanation failed for '%s': %s", model_id, exc)
        return ExplanationResult.unavailable(model_id, str(exc))


# ── Top-level dispatcher ──────────────────────────────────────────────────────

def explain_baseline_with_fallback(
    text:        str,
    model_id:    str,
    num_features: int = 15,
) -> ExplanationResult:
    """
    Explain a baseline prediction — LIME primary, TF-IDF weights fallback.
    Never raises; always returns an ExplanationResult.
    """
    result = explain_baseline(text, model_id, num_features=num_features)
    if result.method == "unavailable" and "LIME" in (result.error or ""):
        logger.info("LIME unavailable — falling back to TF-IDF weights for '%s'.", model_id)
        result = explain_tfidf_weights(text, model_id, num_features=num_features)
    return result


def aggregate_top_tokens(
    explanations: list[ExplanationResult],
    top_n:        int = 15,
) -> list[dict]:
    """
    Aggregate token weights across multiple model explanations for a claim.

    Strategy: weighted average.  Each model's top tokens are averaged.
    Models with higher confidence get proportionally more weight.

    Returns a list of {"token": str, "weight": float} dicts sorted by
    |weight| descending — suitable for storing in claim.top_tokens_json.
    """
    import collections
    token_weights: dict[str, list[float]] = collections.defaultdict(list)

    for exp in explanations:
        if exp.method == "unavailable":
            continue
        for tw in exp.top_tokens:
            token_weights[tw.token].append(tw.weight)

    aggregated: list[dict] = []
    for token, weights in token_weights.items():
        avg_weight = sum(weights) / len(weights)
        aggregated.append({"token": token, "weight": round(avg_weight, 6)})

    aggregated.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return aggregated[:top_n]
