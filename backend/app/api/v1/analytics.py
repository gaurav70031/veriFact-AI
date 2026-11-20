"""
Misinformation Pattern Analytics API.

GET /api/v1/analytics/patterns
    Full pattern analysis — word freq, n-grams, TF-IDF, sentiment, LDA,
    clusters, repeated claims, temporal distribution, research insights.

GET /api/v1/analytics/wordfreq
    Word frequency only (fast, used for word cloud).

GET /api/v1/analytics/temporal
    Temporal distribution only (fast, used for time-series chart).

GET /api/v1/analytics/sentiment
    Sentiment distribution only.

GET /api/v1/analytics/topics
    LDA topic modeling only.

GET /api/v1/analytics/insights
    Research insights only (re-runs full analysis but returns only insights).

All endpoints accept common filter query parameters:
  date_from, date_to, verdict, input_type, confidence_min, confidence_max,
  limit, source (live|dataset|both)

Data integrity
--------------
* All statistics computed from real records — none fabricated.
* Insufficient-data states returned explicitly rather than empty arrays.
* Insights distinguish correlations from causal conclusions.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.analytics_service import AnalyticsFilters, run_analytics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Misinformation Analytics"])


# ── Common filter parameters ──────────────────────────────────────────────────

def _filters(
    date_from:      Optional[str]   = Query(None, description="ISO date lower bound (YYYY-MM-DD)"),
    date_to:        Optional[str]   = Query(None, description="ISO date upper bound (YYYY-MM-DD)"),
    verdict:        Optional[str]   = Query(None, pattern="^(FAKE|REAL|UNVERIFIED|MIXED)$"),
    input_type:     Optional[str]   = Query(None, pattern="^(text|url|query)$"),
    confidence_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    confidence_max: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit:          int             = Query(500, ge=1, le=2000),
    source:         str             = Query("live", pattern="^(live|dataset|both)$"),
) -> AnalyticsFilters:
    return AnalyticsFilters(
        date_from=date_from,
        date_to=date_to,
        verdict=verdict,
        input_type=input_type,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
        limit=limit,
        source=source,
    )


# ── Full analytics ────────────────────────────────────────────────────────────

@router.get(
    "/patterns",
    summary="Full misinformation pattern analysis",
    description=(
        "Runs the complete analytics pipeline:\n\n"
        "- Verdict distribution\n"
        "- Word frequency (all / FAKE / REAL)\n"
        "- Bigram and trigram analysis\n"
        "- TF-IDF discriminating keywords\n"
        "- Sentiment distribution (VADER)\n"
        "- Text length statistics by label\n"
        "- Linguistic features\n"
        "- LDA topic modeling\n"
        "- Article clustering (K-Means on TF-IDF)\n"
        "- Repeated / near-duplicate claims\n"
        "- Temporal distribution\n"
        "- Source and evidence distribution\n"
        "- Research insights\n\n"
        "**All statistics are computed from real data — nothing fabricated.**\n"
        "Heavy NLP runs in a thread pool. Response time: 2–15s depending on corpus size.\n\n"
        "Use `source=dataset` to analyse the ISOT training data.\n"
        "Use `source=live` (default) to analyse submitted articles."
    ),
)
async def full_patterns(
    n_topics:   int = Query(5, ge=2, le=10, description="Number of LDA topics"),
    n_clusters: int = Query(5, ge=2, le=10, description="Number of K-Means clusters"),
    filters: AnalyticsFilters = Depends(_filters),
    db: AsyncSession = Depends(get_db),
) -> dict:
    return await run_analytics(db, filters, n_topics=n_topics, n_clusters=n_clusters)


# ── Word frequency ────────────────────────────────────────────────────────────

@router.get(
    "/wordfreq",
    summary="Word frequency for word cloud",
    description=(
        "Returns top-N word frequencies for all / FAKE / REAL texts.\n"
        "Fast endpoint designed for rendering word clouds."
    ),
)
async def word_frequency(
    top_n:   int = Query(80, ge=10, le=200),
    filters: AnalyticsFilters = Depends(_filters),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.analytics_service import _load_live_analyses, _load_dataset_records
    from app.analysis.pattern_engine import word_frequency as _wf

    records = []
    if filters.source in ("live", "both"):
        records.extend(await _load_live_analyses(db, filters))
    if filters.source in ("dataset", "both"):
        records.extend(_load_dataset_records(limit=filters.limit, verdict_filter=filters.verdict))

    if not records:
        return {"all": [], "FAKE": [], "REAL": [], "document_count": 0}

    import asyncio
    from functools import partial

    def _compute(records, top_n):
        texts_all  = [r["text"] for r in records if r.get("text")]
        texts_fake = [r["text"] for r in records if r.get("text") and r.get("final_verdict") == "FAKE"]
        texts_real = [r["text"] for r in records if r.get("text") and r.get("final_verdict") == "REAL"]
        return {
            "all":           _wf(texts_all,  top_n=top_n),
            "FAKE":          _wf(texts_fake, top_n=top_n),
            "REAL":          _wf(texts_real, top_n=top_n),
            "document_count": len(texts_all),
        }

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(_compute, records, top_n))


# ── Temporal distribution ─────────────────────────────────────────────────────

@router.get(
    "/temporal",
    summary="Date-wise analysis trend",
    description="Returns article counts by date, week, or month — broken down by verdict.",
)
async def temporal(
    granularity: str = Query("day", pattern="^(day|week|month)$"),
    filters: AnalyticsFilters = Depends(_filters),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.analytics_service import _load_live_analyses
    from app.analysis.pattern_engine import temporal_distribution

    records = await _load_live_analyses(db, filters)
    series  = temporal_distribution(records, granularity=granularity)
    return {
        "series":       series,
        "granularity":  granularity,
        "document_count": len(records),
    }


# ── Sentiment ─────────────────────────────────────────────────────────────────

@router.get(
    "/sentiment",
    summary="Sentiment distribution",
    description=(
        "Sentiment analysis using VADER (rule-based). "
        "Falls back to lexicon if VADER not installed.\n\n"
        "Returns distribution split by label (FAKE/REAL/all) and average compound score."
    ),
)
async def sentiment(
    filters: AnalyticsFilters = Depends(_filters),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.analytics_service import _load_live_analyses, _load_dataset_records
    from app.analysis.pattern_engine import sentiment_analysis

    records = []
    if filters.source in ("live", "both"):
        records.extend(await _load_live_analyses(db, filters))
    if filters.source in ("dataset", "both"):
        records.extend(_load_dataset_records(limit=filters.limit))

    if not records:
        return {"error": "No records found."}

    import asyncio
    from functools import partial

    def _compute(records):
        texts_all  = [r["text"] for r in records if r.get("text")]
        texts_fake = [r["text"] for r in records if r.get("text") and r.get("final_verdict") == "FAKE"]
        texts_real = [r["text"] for r in records if r.get("text") and r.get("final_verdict") == "REAL"]
        return {
            "all":  sentiment_analysis(texts_all),
            "FAKE": sentiment_analysis(texts_fake) if texts_fake else {},
            "REAL": sentiment_analysis(texts_real) if texts_real else {},
            "document_count": len(texts_all),
        }

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(_compute, records))


# ── LDA topics ────────────────────────────────────────────────────────────────

@router.get(
    "/topics",
    summary="LDA topic modeling",
    description=(
        "Discovers latent topics using LDA (sklearn LatentDirichletAllocation).\n\n"
        "Requires at least 20 documents. Falls back gracefully with an explanatory message.\n\n"
        "Topics are statistical word co-occurrence patterns — not human-defined categories."
    ),
)
async def topics(
    n_topics: int = Query(5, ge=2, le=10),
    filters: AnalyticsFilters = Depends(_filters),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.analytics_service import _load_live_analyses, _load_dataset_records
    from app.analysis.pattern_engine import lda_topics

    records = []
    if filters.source in ("live", "both"):
        records.extend(await _load_live_analyses(db, filters))
    if filters.source in ("dataset", "both"):
        records.extend(_load_dataset_records(limit=filters.limit))

    texts = [r["text"] for r in records if r.get("text")]

    import asyncio
    from functools import partial

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, partial(lda_topics, texts, n_topics)
    )
    result["document_count"] = len(texts)
    return result


# ── Research insights ─────────────────────────────────────────────────────────

@router.get(
    "/insights",
    summary="Research insights",
    description=(
        "Runs a full analytics pass and returns only the derived research observations.\n\n"
        "All insights are explicitly labelled as correlations, not causal claims.\n"
        "Only generated from actual computed statistics — never invented."
    ),
)
async def insights(
    filters: AnalyticsFilters = Depends(_filters),
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await run_analytics(db, filters, n_topics=5, n_clusters=5)
    return {
        "insights":       result.get("insights", []),
        "document_count": result.get("document_count", 0),
        "computed_at":    result.get("computed_at"),
        "source":         result.get("source"),
    }
