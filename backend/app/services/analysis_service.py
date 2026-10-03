"""
Analysis service.

Orchestrates the full analysis pipeline for every input type:
  text  → clean → run ML + evidence → persist → return
  url   → extract article → clean → run ML + evidence → persist → return
  claim → combine with context → clean → run ML + evidence → persist → return

Design rules
------------
* No ML training here — only inference via ModelRegistry.
* ML inference (CPU/GPU-bound) runs in asyncio thread-pool executor
  so it does not block the async event loop.
* Evidence retrieval runs concurrently with ML inference.
* Each analysis is persisted to PostgreSQL: one Analysis row +
  one Prediction row per model + EvidenceSource rows.
* Verdict is determined by weighted ensemble of available model outputs.
* If all models are unavailable the service raises ModelUnavailableError
  (HTTP 503) — never returns a hardcoded prediction.
* Evidence failure never blocks the analysis — it is logged and ignored.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from functools import partial
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.errors import ModelUnavailableError, ArticleExtractionError
from app.evidence.evidence_service import retrieve_evidence
from app.evidence.schema import EvidenceResult, EvidenceStatus
from app.extraction.article_extractor import extract_article
from app.ml.model_registry import get_registry
from app.models.analysis import Analysis, InputType, AnalysisStatus, FinalVerdict
from app.models.prediction import Prediction, PredictionLabel
from app.models.model_version import ModelVersion
from app.models.evidence_source import EvidenceSource, SourceType, EvidenceRelationship
from app.schemas.analyze import AnalysisResponse, ModelPrediction

logger = logging.getLogger(__name__)

# ── Verdict weights for ensemble ──────────────────────────────────────────────
# Higher weight = more influence on final verdict.
_MODEL_WEIGHTS: dict[str, float] = {
    "logistic_regression": 0.20,
    "linear_svm":          0.30,
    "naive_bayes":         0.10,
    "distilbert":          0.40,
}


# ── Ensemble logic ────────────────────────────────────────────────────────────

def _compute_ensemble_verdict(
    predictions: dict[str, dict],
) -> tuple[str, float]:
    """
    Compute final verdict from per-model predictions using weighted average.

    Parameters
    ----------
    predictions : {model_id: prediction_dict}
                  Each dict has fake_probability, real_probability.

    Returns
    -------
    (verdict_str, confidence_float)
    """
    if not predictions:
        return "UNVERIFIED", 0.5

    weighted_fake = 0.0
    weighted_real = 0.0
    total_weight  = 0.0

    for model_id, pred in predictions.items():
        w = _MODEL_WEIGHTS.get(model_id, 0.1)
        weighted_fake += w * pred.get("fake_probability", 0.5)
        weighted_real += w * pred.get("real_probability", 0.5)
        total_weight  += w

    if total_weight == 0:
        return "UNVERIFIED", 0.5

    avg_fake = weighted_fake / total_weight
    avg_real = weighted_real / total_weight
    confidence = max(avg_fake, avg_real)

    if confidence < 0.55:
        verdict = "MIXED"
    elif avg_fake > avg_real:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return verdict, round(confidence, 6)


# ── Model version helpers ─────────────────────────────────────────────────────

async def _get_or_create_model_version(
    db: AsyncSession,
    model_id: str,
) -> Optional[ModelVersion]:
    """
    Look up ModelVersion by model_name.  Returns None if not registered.
    Creates a stub row on first use so predictions always have a valid FK.
    """
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.model_name == model_id,
            ModelVersion.is_active.is_(True),
        )
    )
    mv = result.scalar_one_or_none()

    if mv is None:
        logger.debug("No ModelVersion row for '%s' — creating stub.", model_id)
        from app.models.model_version import ModelType, ModelAlgorithm
        algo_map = {
            "logistic_regression": ModelAlgorithm.LOGISTIC_REGRESSION,
            "linear_svm":          ModelAlgorithm.LINEAR_SVM,
            "naive_bayes":         ModelAlgorithm.NAIVE_BAYES,
            "distilbert":          ModelAlgorithm.DISTILBERT,
        }
        type_map = {
            "logistic_regression": ModelType.TRADITIONAL,
            "linear_svm":          ModelType.TRADITIONAL,
            "naive_bayes":         ModelType.TRADITIONAL,
            "distilbert":          ModelType.TRANSFORMER,
        }
        mv = ModelVersion(
            model_name=model_id,
            version="1.0.0",
            model_type=type_map.get(model_id, ModelType.TRADITIONAL),
            algorithm=algo_map.get(model_id, ModelAlgorithm.LOGISTIC_REGRESSION),
            artifact_path=f"ml/saved_models/{model_id}",
            is_active=True,
        )
        db.add(mv)
        await db.flush()   # get id without committing

    return mv


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def _run_inference_async(text: str) -> dict[str, dict]:
    """
    Run all available models in the thread-pool executor.
    Returns {model_id: prediction_dict}.
    Gracefully excludes any model that raises ModelUnavailableError.
    """
    registry = get_registry()
    results: dict[str, dict] = {}
    loop = asyncio.get_event_loop()

    # Baseline models
    if registry._baseline_ready:
        try:
            baseline_preds = await loop.run_in_executor(
                None,
                partial(registry.predict_all_baseline, text),
            )
            results.update(baseline_preds)
        except ModelUnavailableError:
            logger.warning("Baseline models unavailable during inference.")

    # Transformer model
    if registry._transformer_ready:
        try:
            transformer_pred = await loop.run_in_executor(
                None,
                partial(registry.predict_transformer, text),
            )
            results["distilbert"] = transformer_pred
        except ModelUnavailableError:
            logger.warning("Transformer model unavailable during inference.")

    if not results:
        raise ModelUnavailableError(
            "No trained models are available. "
            "Run scripts/train_baseline.py (and optionally "
            "scripts/train_transformer.py) then restart the server."
        )

    return results


async def _persist_analysis(
    db:             AsyncSession,
    analysis:       Analysis,
    predictions:    dict[str, dict],
    evidence:       Optional[EvidenceResult] = None,
) -> Analysis:
    """Save Prediction + EvidenceSource rows and update Analysis status/verdict in DB."""
    verdict, confidence = _compute_ensemble_verdict(predictions)

    analysis.final_verdict    = FinalVerdict(verdict)
    analysis.final_confidence = confidence
    analysis.status           = AnalysisStatus.COMPLETED

    summary_parts = []
    for model_id, pred in predictions.items():
        label = pred.get("label", "?")
        conf  = pred.get("confidence", 0.0)
        summary_parts.append(f"{model_id}: {label} ({conf:.0%})")
    analysis.summary = "Model predictions — " + ", ".join(summary_parts)

    for model_id, pred in predictions.items():
        mv = await _get_or_create_model_version(db, model_id)
        if mv is None:
            continue
        is_fake = pred.get("is_fake", pred.get("label") == "FAKE")
        prediction_row = Prediction(
            analysis_id=analysis.id,
            model_version_id=mv.id,
            label=PredictionLabel.FAKE if is_fake else PredictionLabel.REAL,
            confidence=pred.get("confidence", 0.5),
            fake_probability=pred.get("fake_probability", 0.5),
            real_probability=pred.get("real_probability", 0.5),
        )
        db.add(prediction_row)

    # ── Persist evidence sources ──────────────────────────────────────────────
    # Evidence belongs to the first (and only) Claim for this analysis.
    # We create the Claim row if it doesn't exist yet.
    if evidence and evidence.items:
        from app.models.claim import Claim, ClaimVerdict
        claim_row = Claim(
            analysis_id=analysis.id,
            claim_text=analysis.original_input[:2000],
            position=1,
            verdict=ClaimVerdict(verdict) if verdict in ("FAKE", "REAL", "UNVERIFIED", "MIXED") else ClaimVerdict.UNVERIFIED,
            confidence=confidence,
        )
        db.add(claim_row)
        await db.flush()

        _src_type_map = {
            "news_api":   SourceType.NEWS_API,
            "gnews":      SourceType.NEWS_API,
            "rss_feed":   SourceType.RSS_FEED,
            "web_search": SourceType.WEB_SEARCH,
        }

        for rank, ev_item in enumerate(evidence.items, start=1):
            src_type = _src_type_map.get(ev_item.source_type.value, SourceType.UNKNOWN)
            from datetime import datetime, timezone
            ev_row = EvidenceSource(
                claim_id=claim_row.id,
                source_name=ev_item.source_name[:200],
                title=ev_item.title[:500],
                url=ev_item.url[:2000],
                snippet=(ev_item.description or "")[:500] or None,
                source_type=src_type,
                published_at=ev_item.published_at,
                retrieved_at=ev_item.retrieved_at,
                relevance_score=ev_item.relevance_score,
                rank=rank,
                relationship_to_claim=EvidenceRelationship.INCONCLUSIVE,
            )
            db.add(ev_row)

    return analysis


def _build_response(
    analysis:    Analysis,
    predictions: dict[str, dict],
) -> AnalysisResponse:
    model_preds = []
    for model_id, pred in predictions.items():
        model_preds.append(ModelPrediction(
            model_id=model_id,
            model_name=model_id.replace("_", " ").title(),
            label=pred.get("label", "UNKNOWN"),
            is_fake=bool(pred.get("is_fake", False)),
            confidence=float(pred.get("confidence", 0.5)),
            fake_probability=float(pred.get("fake_probability", 0.5)),
            real_probability=float(pred.get("real_probability", 0.5)),
            inference_time_ms=float(pred.get("inference_time_ms", 0.0)),
        ))

    return AnalysisResponse(
        id=analysis.id,
        input_type=analysis.input_type.value,
        original_input=analysis.original_input,
        source_url=analysis.source_url,
        article_title=analysis.article_title,
        final_verdict=analysis.final_verdict.value if analysis.final_verdict else "UNVERIFIED",
        final_confidence=analysis.final_confidence,
        summary=analysis.summary,
        model_predictions=model_preds,
        status=analysis.status.value,
        processing_time_ms=analysis.processing_time_ms,
        created_at=analysis.created_at,
    )


async def _run_ml_and_evidence(
    text: str, claim: str
) -> tuple[dict[str, dict], Optional[EvidenceResult]]:
    """Run ML inference and evidence retrieval concurrently."""
    ml_coro  = _run_inference_async(text)
    ev_coro  = _safe_retrieve_evidence(claim)
    ml_result, ev_result = await asyncio.gather(ml_coro, ev_coro)
    return ml_result, ev_result


async def _safe_retrieve_evidence(claim: str) -> Optional[EvidenceResult]:
    """Retrieve evidence without blocking the analysis if it fails."""
    try:
        return await retrieve_evidence(claim)
    except Exception as exc:
        logger.warning("Evidence retrieval failed (non-blocking): %s", exc)
        return None


# ── Public service functions ──────────────────────────────────────────────────

async def analyse_text(
    db:    AsyncSession,
    text:  str,
    title: Optional[str] = None,
) -> AnalysisResponse:
    """Analyse raw text or article body."""
    combined = f"{title} {text}".strip() if title else text

    t0 = time.perf_counter()

    analysis = Analysis(
        input_type=InputType.TEXT,
        original_input=combined[:2000],
        article_title=title,
        status=AnalysisStatus.PROCESSING,
    )
    db.add(analysis)
    await db.flush()

    try:
        predictions, evidence = await _run_ml_and_evidence(combined, combined[:500])
        analysis = await _persist_analysis(db, analysis, predictions, evidence)
        analysis.processing_time_ms = int((time.perf_counter() - t0) * 1000)
        return _build_response(analysis, predictions)
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models available."
        raise


async def analyse_url(
    db:  AsyncSession,
    url: str,
) -> AnalysisResponse:
    """Extract article from URL, then analyse."""
    t0 = time.perf_counter()

    analysis = Analysis(
        input_type=InputType.URL,
        original_input=url,
        source_url=url,
        status=AnalysisStatus.PROCESSING,
    )
    db.add(analysis)
    await db.flush()

    try:
        article = await extract_article(url)
    except ArticleExtractionError as exc:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = exc.message
        raise

    combined = f"{article.title or ''} {article.text}".strip()
    analysis.article_title = article.title
    analysis.article_text  = article.text[:10000]

    try:
        predictions, evidence = await _run_ml_and_evidence(combined, combined[:500])
        analysis = await _persist_analysis(db, analysis, predictions, evidence)
        analysis.processing_time_ms = int((time.perf_counter() - t0) * 1000)
        return _build_response(analysis, predictions)
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models available."
        raise


async def analyse_claim(
    db:      AsyncSession,
    claim:   str,
    context: Optional[str] = None,
) -> AnalysisResponse:
    """Analyse a short factual claim, optionally with surrounding context."""
    combined = f"{claim} {context or ''}".strip()

    t0 = time.perf_counter()

    analysis = Analysis(
        input_type=InputType.TEXT,
        original_input=claim[:500],
        status=AnalysisStatus.PROCESSING,
    )
    db.add(analysis)
    await db.flush()

    try:
        # For claims, use the raw claim (not context) as the evidence query
        predictions, evidence = await _run_ml_and_evidence(combined, claim)
        analysis = await _persist_analysis(db, analysis, predictions, evidence)
        analysis.processing_time_ms = int((time.perf_counter() - t0) * 1000)
        return _build_response(analysis, predictions)
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models available."
        raise
