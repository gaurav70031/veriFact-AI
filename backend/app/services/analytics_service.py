"""
backend/app/services/analytics_service.py
==========================================
Misinformation analytics service.

Orchestrates all pattern analysis computations:
  1. Loads data from PostgreSQL (live analyses) and/or processed dataset CSVs.
  2. Applies requested filters (date range, verdict, confidence, input_type).
  3. Runs NLP computations via pattern_engine.py in a thread pool.
  4. Returns a structured response ready for the API endpoint.

Data sources
------------
  "live"    : PostgreSQL analyses table (real submitted articles)
  "dataset" : ml/datasets/processed/train.csv (ISOT training data)
  "both"    : Combined

Filter parameters (all optional)
---------------------------------
  date_from       : ISO date string — lower bound on created_at
  date_to         : ISO date string — upper bound on created_at
  verdict         : "FAKE" | "REAL" | "UNVERIFIED" | "MIXED"
  input_type      : "text" | "url" | "query"
  confidence_min  : float 0.0–1.0
  confidence_max  : float 0.0–1.0
  limit           : max records to analyse (default 500, max 2000)
  source          : "live" | "dataset" | "both" (default "live")

Performance notes
-----------------
  Heavy NLP (LDA, clustering) is skipped when fewer than 20 documents.
  All heavy work runs in a thread-pool executor so the event loop stays free.
  Results are NOT cached between requests — each call recomputes from DB.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis, AnalysisStatus, FinalVerdict
from app.models.evidence_source import EvidenceSource

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Filter dataclass ──────────────────────────────────────────────────────────

class AnalyticsFilters:
    def __init__(
        self,
        date_from:      Optional[str]   = None,
        date_to:        Optional[str]   = None,
        verdict:        Optional[str]   = None,
        input_type:     Optional[str]   = None,
        confidence_min: Optional[float] = None,
        confidence_max: Optional[float] = None,
        limit:          int             = 500,
        source:         str             = "live",
    ):
        self.date_from      = date_from
        self.date_to        = date_to
        self.verdict        = verdict
        self.input_type     = input_type
        self.confidence_min = confidence_min
        self.confidence_max = confidence_max
        self.limit          = min(limit, 2000)
        self.source         = source  # "live" | "dataset" | "both"


# ── Database query ────────────────────────────────────────────────────────────

async def _load_live_analyses(
    db:      AsyncSession,
    filters: AnalyticsFilters,
) -> list[dict]:
    """Load analyses from PostgreSQL applying all filters."""
    conditions = [Analysis.status == AnalysisStatus.COMPLETED]

    if filters.date_from:
        try:
            dt = datetime.fromisoformat(filters.date_from).replace(tzinfo=timezone.utc)
            conditions.append(Analysis.created_at >= dt)
        except ValueError:
            pass

    if filters.date_to:
        try:
            dt = datetime.fromisoformat(filters.date_to).replace(tzinfo=timezone.utc)
            conditions.append(Analysis.created_at <= dt)
        except ValueError:
            pass

    if filters.verdict:
        try:
            v = FinalVerdict(filters.verdict.upper())
            conditions.append(Analysis.final_verdict == v)
        except ValueError:
            pass

    if filters.input_type:
        from app.models.analysis import InputType
        try:
            it = InputType(filters.input_type.lower())
            conditions.append(Analysis.input_type == it)
        except ValueError:
            pass

    if filters.confidence_min is not None:
        conditions.append(Analysis.final_confidence >= filters.confidence_min)

    if filters.confidence_max is not None:
        conditions.append(Analysis.final_confidence <= filters.confidence_max)

    stmt = (
        select(
            Analysis.id,
            Analysis.input_type,
            Analysis.original_input,
            Analysis.article_title,
            Analysis.final_verdict,
            Analysis.final_confidence,
            Analysis.processing_time_ms,
            Analysis.created_at,
        )
        .where(and_(*conditions))
        .order_by(Analysis.created_at.desc())
        .limit(filters.limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "id":                r.id,
            "input_type":        r.input_type.value,
            "text":              r.article_title or r.original_input or "",
            "final_verdict":     r.final_verdict.value if r.final_verdict else None,
            "final_confidence":  r.final_confidence,
            "processing_time_ms": r.processing_time_ms,
            "created_at":        r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def _load_evidence_sources(
    db: AsyncSession,
    analysis_ids: list[int],
) -> list[dict]:
    """Load evidence sources for a set of analysis IDs."""
    if not analysis_ids:
        return []
    from app.models.claim import Claim
    stmt = (
        select(
            EvidenceSource.source_name,
            EvidenceSource.relationship_to_claim,
            EvidenceSource.relevance_score,
        )
        .join(Claim, EvidenceSource.claim_id == Claim.id)
        .where(Claim.analysis_id.in_(analysis_ids))
    )
    result = await db.execute(stmt)
    return [
        {
            "source_name":           r.source_name,
            "relationship_to_claim": r.relationship_to_claim.value,
            "relevance_score":       r.relevance_score,
        }
        for r in result.all()
    ]


# ── Dataset CSV loader ────────────────────────────────────────────────────────

def _load_dataset_records(
    split:  str = "train",
    limit:  int = 500,
    verdict_filter: Optional[str] = None,
) -> list[dict]:
    """Load records from processed dataset CSV (ISOT or similar)."""
    try:
        import pandas as pd
        path = _PROJECT_ROOT / "ml" / "datasets" / "processed" / f"{split}.csv"
        if not path.exists():
            return []

        df = pd.read_csv(path, nrows=limit)
        label_map = {0: "FAKE", 1: "REAL"}

        records = []
        for i, row in df.iterrows():
            label_int = int(row.get("label", -1))
            verdict = label_map.get(label_int, "UNKNOWN")
            if verdict_filter and verdict != verdict_filter.upper():
                continue
            text = str(row.get("cleaned_text") or row.get("text") or "")
            records.append({
                "id":               i,
                "input_type":       "text",
                "text":             text,
                "final_verdict":    verdict,
                "final_confidence": None,
                "processing_time_ms": None,
                "created_at":       None,
            })

        return records[:limit]

    except Exception as exc:
        logger.info("Dataset CSV load failed: %s", exc)
        return []


# ── Main analytics runner ─────────────────────────────────────────────────────

def _run_analytics_sync(
    records:    list[dict],
    ev_sources: list[dict],
    n_topics:   int,
    n_clusters: int,
) -> dict:
    """
    Run all NLP analytics synchronously.
    Called via loop.run_in_executor to avoid blocking the event loop.
    """
    from app.analysis.pattern_engine import (
        word_frequency, ngram_analysis, tfidf_keywords,
        sentiment_analysis, lda_topics, text_length_analysis,
        linguistic_features, temporal_distribution,
        repeated_claims, cluster_articles, generate_insights,
    )

    texts_all   = [r["text"] for r in records if r.get("text")]
    ids_all     = [r.get("id", 0) for r in records if r.get("text")]
    verdicts    = [r.get("final_verdict") for r in records]

    if not texts_all:
        return {"error": "No text content found in records.", "filters_applied": True}

    # ── Group by label ────────────────────────────────────────────────────────
    texts_by_label: dict[str, list[str]] = {}
    for r in records:
        v = r.get("final_verdict") or "UNKNOWN"
        if r.get("text"):
            texts_by_label.setdefault(v, []).append(r["text"])

    # ── Verdict distribution ──────────────────────────────────────────────────
    total = len(records)
    verdict_counts = {}
    for v in verdicts:
        k = str(v or "UNKNOWN")
        verdict_counts[k] = verdict_counts.get(k, 0) + 1
    verdict_distribution = {
        "total":  total,
        "counts": verdict_counts,
        "fake":   {"count": verdict_counts.get("FAKE", 0),
                   "pct":   round(verdict_counts.get("FAKE", 0) / max(total, 1) * 100, 2)},
        "real":   {"count": verdict_counts.get("REAL", 0),
                   "pct":   round(verdict_counts.get("REAL", 0) / max(total, 1) * 100, 2)},
    }

    # ── Word frequency ────────────────────────────────────────────────────────
    wf_all  = word_frequency(texts_all, top_n=50)
    wf_fake = word_frequency(texts_by_label.get("FAKE", []), top_n=30)
    wf_real = word_frequency(texts_by_label.get("REAL", []), top_n=30)

    # ── N-gram analysis ───────────────────────────────────────────────────────
    bigrams   = ngram_analysis(texts_all, n=2, top_n=20)
    trigrams  = ngram_analysis(texts_all, n=3, top_n=15)

    # ── TF-IDF keywords ───────────────────────────────────────────────────────
    tfidf_kw = tfidf_keywords(
        {k: v for k, v in texts_by_label.items() if v},
        top_n=20,
    )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    sent_all  = sentiment_analysis(texts_all)
    sent_by_label = {
        label: sentiment_analysis(txts)
        for label, txts in texts_by_label.items()
        if txts
    }

    # ── Text length ───────────────────────────────────────────────────────────
    length_data = text_length_analysis(
        {k: v for k, v in texts_by_label.items() if v}
    )

    # ── Linguistic features ───────────────────────────────────────────────────
    ling_all  = linguistic_features(texts_all, label="all")
    ling_by_label = {
        label: linguistic_features(txts, label=label)
        for label, txts in texts_by_label.items()
        if txts
    }

    # ── Temporal distribution ─────────────────────────────────────────────────
    temporal = temporal_distribution(records, granularity="day")

    # ── Source distribution ───────────────────────────────────────────────────
    source_dist: dict[str, int] = {}
    for src in ev_sources:
        name = src.get("source_name", "Unknown")
        source_dist[name] = source_dist.get(name, 0) + 1
    source_distribution = [
        {"source": k, "count": v, "pct": round(v / max(len(ev_sources), 1) * 100, 2)}
        for k, v in sorted(source_dist.items(), key=lambda x: x[1], reverse=True)[:20]
    ]

    # ── Relationship distribution (evidence) ──────────────────────────────────
    rel_dist: dict[str, int] = {}
    for src in ev_sources:
        rel = str(src.get("relationship_to_claim", "unknown"))
        rel_dist[rel] = rel_dist.get(rel, 0) + 1

    # ── Heavy NLP (only when enough data) ────────────────────────────────────
    lda_result      = {}
    clusters_result = {}
    repeated        = []

    if len(texts_all) >= 10:
        lda_result = lda_topics(texts_all, n_topics=min(n_topics, len(texts_all) // 3))

    if len(texts_all) >= 10:
        clusters_result = cluster_articles(texts_all, ids_all, n_clusters=min(n_clusters, len(texts_all) // 2))

    if len(texts_all) >= 3:
        repeated = repeated_claims(texts_all, ids_all, threshold=0.5, top_n=10)

    # ── Assemble full result ──────────────────────────────────────────────────
    result = {
        "document_count":       total,
        "verdict_distribution": verdict_distribution,
        "word_frequency":       {"all": wf_all, "FAKE": wf_fake, "REAL": wf_real},
        "ngrams":               {"bigrams": bigrams, "trigrams": trigrams},
        "tfidf_keywords":       tfidf_kw,
        "sentiment":            sent_all,
        "sentiment_by_label":   sent_by_label,
        "text_length":          length_data,
        "linguistic_features":  {"all": ling_all, "by_label": ling_by_label},
        "temporal_distribution": temporal,
        "source_distribution":  source_distribution,
        "evidence_relationships": [
            {"relationship": k, "count": v}
            for k, v in sorted(rel_dist.items(), key=lambda x: x[1], reverse=True)
        ],
        "lda_topics":           lda_result,
        "clusters":             clusters_result,
        "repeated_claims":      repeated,
    }

    # ── Insights ──────────────────────────────────────────────────────────────
    result["insights"] = generate_insights(result)

    return result


async def run_analytics(
    db:        AsyncSession,
    filters:   AnalyticsFilters,
    n_topics:  int = 5,
    n_clusters: int = 5,
) -> dict:
    """
    Main entry point: load data, run analytics, return structured response.
    """
    records:    list[dict] = []
    ev_sources: list[dict] = []

    # ── Load data ─────────────────────────────────────────────────────────────
    if filters.source in ("live", "both"):
        live_records = await _load_live_analyses(db, filters)
        records.extend(live_records)
        if live_records:
            analysis_ids = [r["id"] for r in live_records]
            ev_sources = await _load_evidence_sources(db, analysis_ids)

    if filters.source in ("dataset", "both"):
        split = "train"
        dataset_records = _load_dataset_records(
            split=split,
            limit=filters.limit,
            verdict_filter=filters.verdict,
        )
        records.extend(dataset_records)

    if not records:
        return {
            "document_count": 0,
            "source":         filters.source,
            "message": (
                "No records found matching the specified filters. "
                "Submit articles through the analysis pipeline or "
                "place ISOT dataset files in ml/datasets/raw/ and run "
                "python scripts/prepare_dataset.py."
            ),
        }

    # ── Run heavy analytics in thread pool ───────────────────────────────────
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            _run_analytics_sync,
            records,
            ev_sources,
            n_topics,
            n_clusters,
        ),
    )

    # ── Add metadata ──────────────────────────────────────────────────────────
    result["source"]        = filters.source
    result["computed_at"]   = datetime.now(timezone.utc).isoformat()
    result["filters_applied"] = {
        "date_from":      filters.date_from,
        "date_to":        filters.date_to,
        "verdict":        filters.verdict,
        "input_type":     filters.input_type,
        "confidence_min": filters.confidence_min,
        "confidence_max": filters.confidence_max,
        "limit":          filters.limit,
    }

    return result
