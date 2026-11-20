"""
Explanation endpoint.

GET /api/v1/explanation/{analysis_id}            — full token-level attributions
GET /api/v1/explanation/{analysis_id}/report     — structured XAI report (download)
GET /api/v1/explanation/{analysis_id}/annotated  — annotated text for inline highlighting
GET /api/v1/explanation/{analysis_id}/summary    — cross-model token comparison

...rest of original docstring...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse, Response
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


# ── Annotated text endpoint ───────────────────────────────────────────────────

@router.get(
    "/{analysis_id}/annotated",
    summary="Annotated article text for inline highlighting",
    description=(
        "Returns the original article text annotated with per-word explanation weights.\n\n"
        "Each word is tagged with its SHAP/LIME/attention weight so the frontend "
        "can render an inline colour-coded heatmap.\n\n"
        "Words not in the explanation token set get weight 0.0 (neutral)."
    ),
)
async def get_annotated_text(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    analysis = await _get_analysis_or_404(db, analysis_id)

    # Get aggregate token weights from the first claim or all predictions
    pred_result = await db.execute(
        select(Prediction)
        .where(Prediction.analysis_id == analysis_id)
    )
    preds = pred_result.scalars().all()

    all_tokens: list[dict] = []
    for pred in preds:
        exp_data = _parse_explanation_json(pred.explanation_json)
        if exp_data and isinstance(exp_data.get("top_tokens"), list):
            all_tokens.extend(exp_data["top_tokens"])

    # Average across models
    from collections import defaultdict
    token_sums: dict[str, list[float]] = defaultdict(list)
    for item in all_tokens:
        key = str(item.get("token", ""))
        if key:
            token_sums[key].append(float(item.get("weight", 0.0)))
    agg = [
        {"token": k, "weight": sum(v) / len(v)}
        for k, v in token_sums.items()
    ]
    agg.sort(key=lambda x: abs(x["weight"]), reverse=True)

    from app.analysis.explainer import annotate_text
    text = analysis.original_input or ""
    annotated = annotate_text(text, agg, max_length=3000)

    return {
        "analysis_id": analysis_id,
        "annotated":   annotated,
        "token_count": len([t for t in annotated if not t.get("is_space")]),
    }


# ── SHAP summary / cross-model comparison endpoint ───────────────────────────

@router.get(
    "/{analysis_id}/summary",
    summary="Cross-model token explanation comparison",
    description=(
        "Returns a cross-model comparison of the most important tokens, "
        "showing each model's weight for each token side-by-side.\n\n"
        "Useful for the 'Explanation Comparison' chart in the frontend."
    ),
)
async def get_explanation_summary(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    analysis = await _get_analysis_or_404(db, analysis_id)

    pred_result = await db.execute(
        select(Prediction, ModelVersion.model_name)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.analysis_id == analysis_id)
    )
    pred_rows = pred_result.all()

    from app.analysis.explainer import ExplanationResult, TokenWeight
    from app.analysis.shap_explainer import build_shap_summary

    exp_list: list[ExplanationResult] = []
    for pred, model_name in pred_rows:
        exp_data = _parse_explanation_json(pred.explanation_json)
        if not exp_data:
            continue
        tokens = [
            TokenWeight(
                token=str(t.get("token", "")),
                weight=float(t.get("weight", 0.0)),
                position=int(t.get("position", 0)),
            )
            for t in exp_data.get("top_tokens", [])
        ]
        exp_list.append(ExplanationResult(
            model_id=model_name,
            method=exp_data.get("method", "unavailable"),
            label=exp_data.get("label", "UNKNOWN"),
            tokens=tokens,
            top_tokens=tokens[:10],
        ))

    summary = build_shap_summary(exp_list)
    summary["analysis_id"] = analysis_id
    return summary


# ── Full XAI report download ──────────────────────────────────────────────────

@router.get(
    "/{analysis_id}/report",
    summary="Download structured XAI report",
    description=(
        "Returns a complete explainability report as a JSON document "
        "suitable for download or further processing.\n\n"
        "Contains:\n"
        "- Original article preview\n"
        "- Predicted class and confidence\n"
        "- Per-model LIME/SHAP/attention explanations\n"
        "- Annotated article text\n"
        "- Top influential words\n"
        "- Cross-model contribution summary\n"
        "- Explanation limitations\n"
        "- Academic references\n\n"
        "All explanation data is derived from the actual model predictions "
        "stored in the database — not generated by an LLM."
    ),
)
async def get_explanation_report(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
) -> Response:
    analysis = await _get_analysis_or_404(db, analysis_id)

    pred_result = await db.execute(
        select(Prediction, ModelVersion.model_name)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.analysis_id == analysis_id)
    )
    pred_rows = pred_result.all()

    from app.analysis.explainer import ExplanationResult, TokenWeight, build_explanation_report

    exp_list: list[ExplanationResult] = []
    model_preds: list[dict] = []

    for pred, model_name in pred_rows:
        exp_data = _parse_explanation_json(pred.explanation_json)
        tokens = []
        if exp_data:
            tokens = [
                TokenWeight(
                    token=str(t.get("token", "")),
                    weight=float(t.get("weight", 0.0)),
                    position=int(t.get("position", 0)),
                )
                for t in exp_data.get("top_tokens", [])
            ]
        exp_list.append(ExplanationResult(
            model_id=model_name,
            method=(exp_data or {}).get("method", "unavailable"),
            label=(exp_data or {}).get("label", pred.label.value),
            tokens=tokens,
            top_tokens=tokens[:10],
            plain_text=(exp_data or {}).get("plain_text", ""),
            disclaimer=(exp_data or {}).get("disclaimer", ""),
            error=(exp_data or {}).get("error"),
        ))
        model_preds.append({
            "model_id":         model_name,
            "label":            pred.label.value,
            "confidence":       pred.confidence,
            "fake_probability": pred.fake_probability,
            "real_probability": pred.real_probability,
        })

    ml_verdict   = analysis.final_verdict.value if analysis.final_verdict else "UNVERIFIED"
    ml_confidence = analysis.final_confidence

    # Get evidence verdict from first claim
    claims_result = await db.execute(
        select(Claim).where(Claim.analysis_id == analysis_id).order_by(Claim.position)
    )
    claims = claims_result.scalars().all()
    evidence_verdict: Optional[str] = None
    for claim in claims:
        if claim.evidence_verdict:
            evidence_verdict = claim.evidence_verdict.value
            break

    report = build_explanation_report(
        analysis_id=analysis_id,
        original_text=analysis.original_input or "",
        article_title=analysis.article_title,
        ml_verdict=ml_verdict,
        ml_confidence=ml_confidence,
        evidence_verdict=evidence_verdict,
        model_explanations=exp_list,
        model_predictions=model_preds,
        created_at=analysis.created_at.isoformat(),
    )

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    return Response(
        content=report_json.encode("utf-8"),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="xai_report_{analysis_id}.json"',
        },
    )