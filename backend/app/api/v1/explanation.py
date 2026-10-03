"""
Explanation endpoint.

GET /api/v1/explanation/{analysis_id}

Returns the full explainability output for a completed analysis:
  - Per-model token attributions (LIME coefficients or attention weights)
  - Per-claim aggregate token weights
  - Plain-language explanations
  - Signal-vs-evidence warning (always shown)

This endpoint is separate from the main analysis response so that:
  1. The analysis endpoint stays fast — explanation can be fetched lazily.
  2. Clients can request explanation detail only when the user wants it.
  3. Heavy JSON (all tokens) does not bloat the default response payload.

Important: All responses include the signal_vs_evidence_warning field.
Clients MUST display this to users alongside any highlighted tokens.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.db.session import get_db
from app.models.analysis import Analysis
from app.models.claim import Claim
from app.models.prediction import Prediction
from app.models.model_version import ModelVersion
from app.schemas.analyze import (
    ExplanationResponse,
    PerModelExplanation,
    ExplanationTokenWeight,
    ClaimExplanation,
    AggregateTokenWeight,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/explanation", tags=["Explainability"])


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _get_analysis_or_404(db: AsyncSession, analysis_id: int) -> Analysis:
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise NotFoundError(f"Analysis {analysis_id} not found.")
    return analysis


def _parse_explanation_json(raw: Optional[str]) -> Optional[dict]:
    """Safely deserialise explanation_json from the DB; returns None on failure."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse explanation_json — value truncated or corrupt.")
        return None


def _tokens_from_list(token_list: list[dict]) -> list[ExplanationTokenWeight]:
    """Convert raw [{"token":..., "weight":..., "position":...}] → schema list."""
    result = []
    for i, item in enumerate(token_list):
        result.append(ExplanationTokenWeight(
            token=str(item.get("token", "")),
            weight=float(item.get("weight", 0.0)),
            position=int(item.get("position", i)),
        ))
    return result


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.get(
    "/{analysis_id}",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Explainability output for a completed analysis",
    description=(
        "Returns token-level feature attributions for every ML model that "
        "ran on this analysis.\n\n"
        "**Baseline models (LR, SVM, NB):** LIME perturbation-based importance "
        "— positive weight = model associates token with FAKE patterns from training.\n\n"
        "**DistilBERT:** Last-layer attention attribution — CLS→token attention weight.\n\n"
        "**Important:** Token weights reflect statistical patterns the model "
        "learned during training. They are **model signals**, not factual evidence. "
        "A high-weight word does not mean it is factually incorrect.\n\n"
        "Returns HTTP 404 if the analysis does not exist.\n"
        "Returns HTTP 200 with `method='unavailable'` per model if explanation "
        "was not computed (e.g. LIME not installed, or analysis still pending)."
    ),
)
async def get_explanation(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
) -> ExplanationResponse:
    # ── Verify analysis exists ────────────────────────────────────────────────
    analysis = await _get_analysis_or_404(db, analysis_id)

    # ── Fetch prediction rows (with explanation_json) ─────────────────────────
    pred_result = await db.execute(
        select(Prediction, ModelVersion.model_name, ModelVersion.algorithm)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.analysis_id == analysis_id)
    )
    pred_rows = pred_result.all()

    model_explanations: list[PerModelExplanation] = []

    for pred, model_name, algorithm in pred_rows:
        exp_data = _parse_explanation_json(pred.explanation_json)

        if exp_data is None:
            # Explanation was not stored (LIME not installed, or model never ran it)
            model_explanations.append(PerModelExplanation(
                model_id=model_name,
                model_name=model_name.replace("_", " ").title(),
                method="unavailable",
                label=pred.label.value,
                confidence=pred.confidence,
                error="Explanation not available for this prediction.",
            ))
            continue

        # Determine method from stored data (may be "lime", "attention",
        # "tfidf_weights", or "unavailable")
        method     = exp_data.get("method", "unavailable")
        error      = exp_data.get("error")
        label      = exp_data.get("label", pred.label.value)
        plain_text = exp_data.get("plain_text", "")
        disclaimer = exp_data.get("disclaimer", "")

        tokens     = _tokens_from_list(exp_data.get("tokens", []))
        top_tokens = _tokens_from_list(exp_data.get("top_tokens", []))

        model_explanations.append(PerModelExplanation(
            model_id=model_name,
            model_name=model_name.replace("_", " ").title(),
            method=method,
            label=label,
            confidence=pred.confidence,
            tokens=tokens,
            top_tokens=top_tokens,
            plain_text=plain_text,
            disclaimer=disclaimer,
            error=error,
        ))

    # ── Fetch claim rows (with top_tokens_json) ───────────────────────────────
    claims_result = await db.execute(
        select(Claim)
        .where(Claim.analysis_id == analysis_id)
        .order_by(Claim.position)
    )
    claims = claims_result.scalars().all()

    claim_explanations: list[ClaimExplanation] = []
    for claim in claims:
        agg_tokens: list[AggregateTokenWeight] = []
        top_tokens_data = _parse_explanation_json(claim.top_tokens_json)
        if top_tokens_data and isinstance(top_tokens_data, list):
            for item in top_tokens_data:
                agg_tokens.append(AggregateTokenWeight(
                    token=str(item.get("token", "")),
                    weight=float(item.get("weight", 0.0)),
                ))

        claim_explanations.append(ClaimExplanation(
            position=claim.position,
            claim_text=claim.claim_text,
            aggregate_tokens=agg_tokens,
        ))

    # ── Derive overall ML / evidence verdicts for context ─────────────────────
    ml_verdict   = analysis.final_verdict.value if analysis.final_verdict else "UNVERIFIED"
    ml_confidence = analysis.final_confidence

    # Evidence verdict lives in the summary string (we don't repeat the full
    # breakdown here — use GET /analysis/{id} for that)
    evidence_verdict: Optional[str] = None
    if claims:
        # Take evidence verdict from the first claim that has one
        for claim in claims:
            if claim.evidence_verdict:
                evidence_verdict = claim.evidence_verdict.value
                break

    return ExplanationResponse(
        analysis_id=analysis_id,
        ml_verdict=ml_verdict,
        ml_confidence=ml_confidence,
        evidence_verdict=evidence_verdict,
        model_explanations=model_explanations,
        claim_explanations=claim_explanations,
    )
