"""
Analysis service — full pipeline orchestrator.

Pipeline for every input type
------------------------------
  text  → ML inference ║ claim extraction → per-claim evidence → compare → verdict
  url   → extract article → same as text
  claim → ML inference ║ per-claim evidence → compare → verdict

Concurrency model
-----------------
ML inference and evidence retrieval run concurrently via asyncio.gather.
ML is CPU/GPU-bound → thread-pool executor.
Evidence retrieval is I/O-bound → pure async.
Claim extraction + comparison are synchronous but fast (< 50 ms).

Two independent assessments
----------------------------
ml_verdict       — probability-based, from model weights.
evidence_verdict — based on actual retrieved sources.
Neither overrides the other.  Both are stored and returned.

Evidence failure is non-blocking
---------------------------------
If all providers fail or time out the analysis still completes with
evidence_verdict = INSUFFICIENT_EVIDENCE.  The ML verdict is unaffected.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.claim_extractor   import extract_claims, ExtractedClaim
from app.analysis.evidence_comparator import compare as compare_evidence, ComparisonResult
from app.analysis.verdict_engine    import (
    assess_claim, aggregate_verdict,
    ClaimVerdictResult, AnalysisVerdict,
    DEFAULT_CONFIG,
)
from app.core.errors                import ModelUnavailableError, ArticleExtractionError
from app.evidence.evidence_service  import retrieve_evidence
from app.evidence.query_extractor   import extract_queries
from app.evidence.schema            import EvidenceResult, EvidenceStatus
from app.extraction.article_extractor import extract_article
from app.ml.model_registry          import get_registry
from app.models.analysis            import Analysis, InputType, AnalysisStatus, FinalVerdict
from app.models.claim               import Claim, ClaimVerdict, EvidenceAssessment
from app.models.evidence_source     import EvidenceSource, SourceType, EvidenceRelationship
from app.models.model_version       import ModelVersion
from app.models.prediction          import Prediction, PredictionLabel
from app.schemas.analyze            import (
    AnalysisResponse, ModelPrediction,
    ClaimResult, EvidenceSourceResult, EvidenceSummary,
)

logger = logging.getLogger(__name__)

# ── ML ensemble weights ───────────────────────────────────────────────────────
_MODEL_WEIGHTS: dict[str, float] = {
    "logistic_regression": 0.20,
    "linear_svm":          0.30,
    "naive_bayes":         0.10,
    "distilbert":          0.40,
}

# Source type mapping from evidence schema → ORM enum
_SRC_TYPE_MAP: dict[str, SourceType] = {
    "news_api":   SourceType.NEWS_API,
    "gnews":      SourceType.NEWS_API,
    "rss_feed":   SourceType.RSS_FEED,
    "web_search": SourceType.WEB_SEARCH,
    "official":   SourceType.OFFICIAL,
    "fact_check": SourceType.FACT_CHECK,
}


# =============================================================================
# ML ensemble
# =============================================================================

def _compute_ensemble_verdict(predictions: dict[str, dict]) -> tuple[str, float, float, float]:
    """
    Weighted average over all model predictions.

    Returns
    -------
    (verdict, confidence, avg_fake_prob, avg_real_prob)
    """
    if not predictions:
        return "UNVERIFIED", 0.5, 0.5, 0.5

    weighted_fake = weighted_real = total_weight = 0.0
    for model_id, pred in predictions.items():
        w = _MODEL_WEIGHTS.get(model_id, 0.1)
        weighted_fake += w * pred.get("fake_probability", 0.5)
        weighted_real += w * pred.get("real_probability", 0.5)
        total_weight  += w

    if total_weight == 0:
        return "UNVERIFIED", 0.5, 0.5, 0.5

    avg_fake = weighted_fake / total_weight
    avg_real = weighted_real / total_weight
    confidence = max(avg_fake, avg_real)

    if confidence < 0.55:
        verdict = "MIXED"
    elif avg_fake > avg_real:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return verdict, round(confidence, 6), round(avg_fake, 6), round(avg_real, 6)


# =============================================================================
# ML inference
# =============================================================================

async def _run_inference_async(text: str) -> dict[str, dict]:
    """Run all available models in a thread-pool executor."""
    registry = get_registry()
    results: dict[str, dict] = {}
    loop = asyncio.get_event_loop()

    if registry._baseline_ready:
        try:
            baseline_preds = await loop.run_in_executor(
                None, partial(registry.predict_all_baseline, text)
            )
            results.update(baseline_preds)
        except ModelUnavailableError:
            logger.warning("Baseline models unavailable during inference.")

    if registry._transformer_ready:
        try:
            transformer_pred = await loop.run_in_executor(
                None, partial(registry.predict_transformer, text)
            )
            results["distilbert"] = transformer_pred
        except ModelUnavailableError:
            logger.warning("Transformer model unavailable during inference.")

    if not results:
        raise ModelUnavailableError(
            "No trained models are available. "
            "Run scripts/train_baseline.py then restart the server."
        )
    return results


# =============================================================================
# Evidence retrieval (per-claim)
# =============================================================================

async def _safe_retrieve_evidence(claim: str) -> Optional[EvidenceResult]:
    """Retrieve evidence for a single claim; never raises."""
    try:
        return await retrieve_evidence(claim)
    except Exception as exc:
        logger.warning("Evidence retrieval failed for claim '%s…': %s", claim[:60], exc)
        return None


async def _retrieve_all_evidence(
    claims: list[ExtractedClaim],
) -> list[Optional[EvidenceResult]]:
    """
    Retrieve evidence for every extracted claim concurrently.
    Returns a list aligned with `claims`.
    """
    tasks = [_safe_retrieve_evidence(c.text) for c in claims]
    return await asyncio.gather(*tasks)


# =============================================================================
# Evidence comparison (per claim × evidence item)
# =============================================================================

def _compare_claim_to_evidence(
    claim:          ExtractedClaim,
    ev_result:      Optional[EvidenceResult],
    ml_label:       Optional[str],
    ml_fake_prob:   Optional[float],
    ml_real_prob:   Optional[float],
    ml_confidence:  Optional[float],
) -> tuple[ClaimVerdictResult, list[tuple[EvidenceSource, ComparisonResult]]]:
    """
    Compare one claim against all its evidence items.

    Returns
    -------
    (ClaimVerdictResult, list of (EvidenceSource-to-persist, ComparisonResult))
    """
    # Extract keywords once for reuse in comparator
    qs = extract_queries(claim.text)
    keywords = qs.keywords

    relationship_counts: dict[EvidenceRelationship, int] = defaultdict(int)
    ev_source_pairs: list[tuple[EvidenceSource, ComparisonResult]] = []

    total_ev = len(ev_result.items) if ev_result and ev_result.items else 0

    if ev_result and ev_result.items:
        for rank, ev_item in enumerate(ev_result.items, start=1):
            # Build text to compare against (title + snippet)
            snippet_text = " ".join(filter(None, [
                ev_item.title,
                ev_item.description or "",
            ])).strip()

            comparison = compare_evidence(
                claim_text=claim.text,
                snippet=snippet_text,
                claim_keywords=keywords,
            )

            relationship_counts[comparison.relationship] += 1

            # Map evidence schema SourceType → ORM SourceType
            src_type_val = ev_item.source_type.value if hasattr(ev_item.source_type, "value") else str(ev_item.source_type)
            orm_src_type = _SRC_TYPE_MAP.get(src_type_val, SourceType.NEWS_API)

            ev_src_row = EvidenceSource(
                # claim_id filled in later after Claim is flushed
                source_name=ev_item.source_name[:200],
                title=ev_item.title[:500],
                url=ev_item.url[:2000],
                snippet=(ev_item.description or "")[:500] or None,
                source_type=orm_src_type,
                published_at=ev_item.published_at,
                retrieved_at=ev_item.retrieved_at,
                relevance_score=ev_item.relevance_score,
                comparison_score=comparison.comparison_score,
                rank=rank,
                relationship_to_claim=comparison.relationship,
            )
            ev_source_pairs.append((ev_src_row, comparison))

    # Assess this claim using the verdict engine
    claim_verdict_result = assess_claim(
        claim_text=claim.text,
        position=claim.position,
        relationship_counts=dict(relationship_counts),
        total_evidence=total_ev,
        ml_label=ml_label,
        ml_confidence=ml_confidence,
        ml_fake_prob=ml_fake_prob,
        ml_real_prob=ml_real_prob,
        config=DEFAULT_CONFIG,
    )

    return claim_verdict_result, ev_source_pairs


# =============================================================================
# Model version helper
# =============================================================================

async def _get_or_create_model_version(
    db: AsyncSession, model_id: str
) -> Optional[ModelVersion]:
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.model_name == model_id,
            ModelVersion.is_active.is_(True),
        )
    )
    mv = result.scalar_one_or_none()
    if mv is None:
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
        await db.flush()
    return mv


# =============================================================================
# Persistence
# =============================================================================

async def _persist_full_analysis(
    db:               AsyncSession,
    analysis:         Analysis,
    predictions:      dict[str, dict],
    extracted_claims: list[ExtractedClaim],
    claim_results:    list[ClaimVerdictResult],
    ev_pairs_per_claim: list[list[tuple[EvidenceSource, ComparisonResult]]],
    verdict:          AnalysisVerdict,
) -> Analysis:
    """
    Persist Prediction rows, Claim rows, and EvidenceSource rows.
    Updates Analysis with both ML verdict and evidence verdict.
    """
    ml_label, ml_conf, ml_fake, ml_real = _compute_ensemble_verdict(predictions)

    # ── Update Analysis row ───────────────────────────────────────────────────
    analysis.final_verdict    = FinalVerdict(ml_label)
    analysis.final_confidence = ml_conf
    analysis.status           = AnalysisStatus.COMPLETED

    # Build plain-text summary that includes both verdicts
    ml_parts = [
        f"{mid}: {p.get('label','?')} ({p.get('confidence',0):.0%})"
        for mid, p in predictions.items()
    ]
    ev_verdict_str = verdict.overall_evidence_assessment.value
    analysis.summary = (
        f"ML: {ml_label} ({ml_conf:.0%}) — "
        f"Evidence: {ev_verdict_str}. "
        f"{verdict.overall_explanation}"
    )

    # ── Persist Prediction rows ───────────────────────────────────────────────
    for model_id, pred in predictions.items():
        mv = await _get_or_create_model_version(db, model_id)
        if mv is None:
            continue
        is_fake = pred.get("is_fake", pred.get("label") == "FAKE")
        db.add(Prediction(
            analysis_id=analysis.id,
            model_version_id=mv.id,
            label=PredictionLabel.FAKE if is_fake else PredictionLabel.REAL,
            confidence=pred.get("confidence", 0.5),
            fake_probability=pred.get("fake_probability", 0.5),
            real_probability=pred.get("real_probability", 0.5),
        ))

    # ── Persist Claim + EvidenceSource rows ───────────────────────────────────
    for i, (ec, cvr) in enumerate(zip(extracted_claims, claim_results)):
        # Map ML verdict → ClaimVerdict enum
        ml_verdict_for_claim = ml_label if ml_label in ("FAKE", "REAL", "UNVERIFIED", "MIXED") else "UNVERIFIED"

        claim_row = Claim(
            analysis_id=analysis.id,
            claim_text=ec.text[:2000],
            position=ec.position,
            verdict=ClaimVerdict(ml_verdict_for_claim),
            confidence=ml_conf,
            evidence_verdict=cvr.evidence_assessment,
            explanation=(
                f"ML: {ml_label} ({ml_conf:.0%})"
            ),
            evidence_explanation=cvr.evidence_explanation,
        )
        db.add(claim_row)
        await db.flush()   # need claim_row.id for FK

        for ev_src_row, _ in ev_pairs_per_claim[i]:
            ev_src_row.claim_id = claim_row.id
            db.add(ev_src_row)

    return analysis


# =============================================================================
# Response builder
# =============================================================================

def _build_response(
    analysis:         Analysis,
    predictions:      dict[str, dict],
    extracted_claims: list[ExtractedClaim],
    claim_results:    list[ClaimVerdictResult],
    ev_results:       list[Optional[EvidenceResult]],
    ev_pairs_per_claim: list[list[tuple[EvidenceSource, ComparisonResult]]],
    verdict:          AnalysisVerdict,
) -> AnalysisResponse:
    """Construct the full AnalysisResponse from all pipeline outputs."""

    ml_label, ml_conf, _, _ = _compute_ensemble_verdict(predictions)

    # Per-model ML predictions
    model_preds = [
        ModelPrediction(
            model_id=mid,
            model_name=mid.replace("_", " ").title(),
            label=p.get("label", "UNKNOWN"),
            is_fake=bool(p.get("is_fake", False)),
            confidence=float(p.get("confidence", 0.5)),
            fake_probability=float(p.get("fake_probability", 0.5)),
            real_probability=float(p.get("real_probability", 0.5)),
            inference_time_ms=float(p.get("inference_time_ms", 0.0)),
        )
        for mid, p in predictions.items()
    ]

    # Per-claim results
    claim_responses: list[ClaimResult] = []
    for i, (ec, cvr) in enumerate(zip(extracted_claims, claim_results)):
        # Build EvidenceSourceResult list for this claim
        ev_source_results: list[EvidenceSourceResult] = []
        for ev_src_row, _ in ev_pairs_per_claim[i]:
            ev_source_results.append(EvidenceSourceResult(
                source_name=ev_src_row.source_name,
                title=ev_src_row.title,
                url=ev_src_row.url,
                snippet=ev_src_row.snippet,
                source_type=ev_src_row.source_type.value,
                published_at=ev_src_row.published_at,
                retrieved_at=ev_src_row.retrieved_at,
                relevance_score=ev_src_row.relevance_score,
                comparison_score=ev_src_row.comparison_score,
                rank=ev_src_row.rank,
                relationship_to_claim=ev_src_row.relationship_to_claim.value,
            ))

        claim_responses.append(ClaimResult(
            position=ec.position,
            claim_text=ec.text,
            ml_verdict=ml_label,
            ml_confidence=ml_conf,
            evidence_verdict=cvr.evidence_assessment.value,
            evidence_explanation=cvr.evidence_explanation,
            evidence_sources=ev_source_results,
            supporting_count=cvr.supporting_count,
            contradicting_count=cvr.contradicting_count,
            inconclusive_count=cvr.inconclusive_count,
            not_relevant_count=cvr.not_relevant_count,
            total_evidence=cvr.total_evidence,
            assessment_conflict=cvr.assessment_conflict,
            evidence_limitations=cvr.evidence_limitations,
        ))

    # Aggregate evidence summary
    all_providers_used:   list[str] = []
    all_providers_failed: list[str] = []
    for ev_r in ev_results:
        if ev_r:
            all_providers_used.extend(ev_r.providers_used)
            all_providers_failed.extend(ev_r.providers_failed)
    all_providers_used   = list(dict.fromkeys(all_providers_used))    # deduplicate, keep order
    all_providers_failed = list(dict.fromkeys(all_providers_failed))

    ev_summary = EvidenceSummary(
        total_evidence=verdict.total_evidence,
        supporting_count=verdict.total_supporting,
        contradicting_count=verdict.total_contradicting,
        inconclusive_count=verdict.total_inconclusive,
        not_relevant_count=verdict.total_not_relevant,
        providers_used=all_providers_used,
        providers_failed=all_providers_failed,
        evidence_limitations=list(dict.fromkeys(verdict.evidence_limitations)),
        all_providers_failed=(
            len(all_providers_used) == 0 and len(all_providers_failed) > 0
        ),
    )

    return AnalysisResponse(
        id=analysis.id,
        input_type=analysis.input_type.value,
        original_input=analysis.original_input,
        source_url=analysis.source_url,
        article_title=analysis.article_title,
        ml_verdict=ml_label,
        ml_confidence=ml_conf,
        evidence_verdict=verdict.overall_evidence_assessment.value,
        evidence_explanation=verdict.overall_explanation,
        model_predictions=model_preds,
        claims=claim_responses,
        evidence_summary=ev_summary,
        summary=analysis.summary,
        status=analysis.status.value,
        processing_time_ms=analysis.processing_time_ms,
        created_at=analysis.created_at,
    )


# =============================================================================
# Core orchestrator
# =============================================================================

async def _run_full_pipeline(
    db:              AsyncSession,
    analysis:        Analysis,
    full_text:       str,
    evidence_seed:   str,
    max_claims:      int = 5,
) -> AnalysisResponse:
    """
    Shared pipeline for all three input types.

    Steps
    -----
    1. ML inference + claim extraction (concurrent)
    2. Per-claim evidence retrieval (all claims concurrent)
    3. Per-claim evidence comparison (synchronous, fast)
    4. Verdict aggregation
    5. DB persistence
    6. Response construction
    """
    t0 = time.perf_counter()

    # ── Step 1: ML inference + claim extraction (concurrent) ─────────────────
    ml_coro    = _run_inference_async(full_text)
    # Claim extraction is synchronous but wrapped to run in executor for safety
    loop = asyncio.get_event_loop()
    claims_coro = loop.run_in_executor(
        None, lambda: extract_claims(full_text, max_claims=max_claims)
    )
    try:
        (predictions, extracted_claims) = await asyncio.gather(ml_coro, claims_coro)
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models are available."
        raise

    ml_label, ml_conf, ml_fake, ml_real = _compute_ensemble_verdict(predictions)
    logger.info(
        "ML verdict: %s (%.1f%%) from %d models | %d claims extracted",
        ml_label, ml_conf * 100, len(predictions), len(extracted_claims),
    )

    # ── Step 2: Per-claim evidence retrieval (all concurrent) ────────────────
    ev_results: list[Optional[EvidenceResult]] = await _retrieve_all_evidence(
        extracted_claims
    )

    # ── Step 3: Evidence comparison ───────────────────────────────────────────
    claim_results:      list[ClaimVerdictResult]                        = []
    ev_pairs_per_claim: list[list[tuple[EvidenceSource, ComparisonResult]]] = []

    for ec, ev_result in zip(extracted_claims, ev_results):
        cvr, ev_pairs = _compare_claim_to_evidence(
            claim=ec,
            ev_result=ev_result,
            ml_label=ml_label,
            ml_fake_prob=ml_fake,
            ml_real_prob=ml_real,
            ml_confidence=ml_conf,
        )
        claim_results.append(cvr)
        ev_pairs_per_claim.append(ev_pairs)

    # ── Step 4: Aggregate verdict ─────────────────────────────────────────────
    all_providers_used   = []
    all_providers_failed = []
    for ev_r in ev_results:
        if ev_r:
            all_providers_used.extend(ev_r.providers_used)
            all_providers_failed.extend(ev_r.providers_failed)

    verdict = aggregate_verdict(
        claim_results=claim_results,
        ml_ensemble_label=ml_label,
        ml_ensemble_confidence=ml_conf,
        providers_used=list(dict.fromkeys(all_providers_used)),
        providers_failed=list(dict.fromkeys(all_providers_failed)),
    )

    logger.info(
        "Evidence verdict: %s | supporting=%d contradicting=%d inconclusive=%d",
        verdict.overall_evidence_assessment.value,
        verdict.total_supporting,
        verdict.total_contradicting,
        verdict.total_inconclusive,
    )

    # ── Step 5: Persist ───────────────────────────────────────────────────────
    analysis = await _persist_full_analysis(
        db=db,
        analysis=analysis,
        predictions=predictions,
        extracted_claims=extracted_claims,
        claim_results=claim_results,
        ev_pairs_per_claim=ev_pairs_per_claim,
        verdict=verdict,
    )
    analysis.processing_time_ms = int((time.perf_counter() - t0) * 1000)

    # ── Step 6: Build response ────────────────────────────────────────────────
    return _build_response(
        analysis=analysis,
        predictions=predictions,
        extracted_claims=extracted_claims,
        claim_results=claim_results,
        ev_results=ev_results,
        ev_pairs_per_claim=ev_pairs_per_claim,
        verdict=verdict,
    )


# =============================================================================
# Public entry points
# =============================================================================

async def analyse_text(
    db:    AsyncSession,
    text:  str,
    title: Optional[str] = None,
) -> AnalysisResponse:
    """Analyse raw article text or multi-sentence content."""
    combined = f"{title} {text}".strip() if title else text

    analysis = Analysis(
        input_type=InputType.TEXT,
        original_input=combined[:2000],
        article_title=title,
        status=AnalysisStatus.PROCESSING,
    )
    db.add(analysis)
    await db.flush()

    try:
        return await _run_full_pipeline(
            db=db,
            analysis=analysis,
            full_text=combined,
            evidence_seed=combined[:500],
        )
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models available."
        raise


async def analyse_url(
    db:  AsyncSession,
    url: str,
) -> AnalysisResponse:
    """Extract article from URL then analyse."""
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
    analysis.article_text  = article.text[:10_000]

    try:
        return await _run_full_pipeline(
            db=db,
            analysis=analysis,
            full_text=combined,
            evidence_seed=(article.title or combined)[:500],
        )
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models available."
        raise


async def analyse_claim(
    db:      AsyncSession,
    claim:   str,
    context: Optional[str] = None,
) -> AnalysisResponse:
    """Analyse a short factual claim with optional context."""
    combined = f"{claim} {context or ''}".strip()

    analysis = Analysis(
        input_type=InputType.TEXT,
        original_input=claim[:500],
        status=AnalysisStatus.PROCESSING,
    )
    db.add(analysis)
    await db.flush()

    try:
        return await _run_full_pipeline(
            db=db,
            analysis=analysis,
            full_text=combined,
            evidence_seed=claim,   # search on the raw claim, not padded context
            max_claims=1,           # claim input = already one claim
        )
    except ModelUnavailableError:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "No ML models available."
        raise
