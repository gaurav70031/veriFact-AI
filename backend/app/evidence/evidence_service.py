"""
Evidence service.

Orchestrates the full evidence retrieval pipeline:
  1. Extract search queries from the claim.
  2. Run each configured provider concurrently.
  3. Skip unconfigured providers.
  4. Catch per-provider errors; log them; continue with remaining providers.
  5. Deduplicate results by URL.
  6. Rank results by relevance score.
  7. Return an EvidenceResult.

If one provider fails, the service continues with the others.
If ALL providers fail, the status is ALL_PROVIDERS_FAILED.
If providers succeed but find nothing, the status is INSUFFICIENT_EVIDENCE.

Production vs test
------------------
In production, concrete provider instances (NewsAPI, GNews, RSS, Search) are
used.  Tests inject mock providers via the `providers` parameter of
`retrieve_evidence()`.  No mocking happens inside this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import get_settings
from app.evidence.base_provider import EvidenceProvider, ProviderError
from app.evidence.query_extractor import extract_queries
from app.evidence.schema import EvidenceItem, EvidenceResult, EvidenceStatus, SourceType

logger   = logging.getLogger(__name__)
settings = get_settings()


# ── Relevance ranking ─────────────────────────────────────────────────────────

def _compute_relevance(item: EvidenceItem, keywords: list[str]) -> float:
    """
    Compute a relevance score in [0, 1] for a single EvidenceItem.

    Factors
    -------
    keyword_overlap : fraction of query keywords that appear in title + description
    recency_boost   : exponential decay — articles published today score highest
    """
    text = f"{item.title} {item.description or ''}".lower()

    # Keyword overlap
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in text)
        keyword_score = hits / len(keywords)
    else:
        keyword_score = 0.5

    # Recency boost — exponential decay with 7-day half-life
    recency_score = 0.5   # neutral when date unknown
    if item.published_at:
        now     = datetime.now(timezone.utc)
        pub     = item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - pub).total_seconds() / 86400)
        recency_score = 2 ** (-age_days / 7)   # 1.0 today → 0.5 at 7 days

    # Weighted combination
    score = 0.65 * keyword_score + 0.35 * recency_score
    return round(min(max(score, 0.0), 1.0), 4)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _url_fingerprint(url: str) -> str:
    """MD5 of normalised URL (lowercase, strip trailing slash and utm params)."""
    clean = url.lower().split("?")[0].rstrip("/")
    return hashlib.md5(clean.encode()).hexdigest()


def _deduplicate(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen:   set[str]          = set()
    result: list[EvidenceItem] = []
    for item in items:
        fp = _url_fingerprint(item.url)
        if fp not in seen:
            seen.add(fp)
            result.append(item)
    return result


# ── Default provider registry ─────────────────────────────────────────────────

def _build_default_providers() -> list[EvidenceProvider]:
    """
    Instantiate the production provider set from environment variables.
    Providers with missing keys are included but will be skipped at runtime
    (is_configured returns False).
    """
    from app.evidence.providers.newsapi_provider import NewsAPIProvider
    from app.evidence.providers.gnews_provider   import GNewsProvider
    from app.evidence.providers.rss_provider     import RSSFeedProvider
    from app.evidence.providers.search_provider  import SearchProvider

    return [
        NewsAPIProvider(),
        GNewsProvider(),
        RSSFeedProvider(),
        SearchProvider(),
    ]


# ── Single-provider search ────────────────────────────────────────────────────

async def _search_one_provider(
    provider:    EvidenceProvider,
    query:       str,
    max_results: int,
    from_date:   Optional[datetime],
) -> tuple[list[EvidenceItem], Optional[str]]:
    """
    Run one provider and return (items, error_message).
    Never raises — all errors are caught and returned as error_message.
    """
    if not provider.is_configured:
        logger.debug("[%s] Skipping — not configured.", provider.name)
        return [], f"{provider.name}: not configured (missing API key)"

    try:
        items = await provider.search(query, max_results=max_results, from_date=from_date)
        logger.info("[%s] %d items retrieved.", provider.name, len(items))
        return items, None
    except ProviderError as exc:
        logger.warning("[%s] %s", provider.name, exc)
        return [], str(exc)
    except Exception as exc:
        logger.error("[%s] Unexpected error: %s", provider.name, exc, exc_info=True)
        return [], f"{provider.name}: unexpected error — {exc}"


# ── Main service function ─────────────────────────────────────────────────────

async def retrieve_evidence(
    claim:        str,
    max_results:  int                        = 10,
    from_days:    Optional[int]              = 30,
    providers:    Optional[list[EvidenceProvider]] = None,
) -> EvidenceResult:
    """
    Retrieve evidence for a claim from all configured providers.

    Parameters
    ----------
    claim       : The factual claim to search for.
    max_results : Maximum total evidence items to return.
    from_days   : Only include articles published within this many days.
                  None = no date restriction.
    providers   : Inject a custom provider list (used in tests).
                  Defaults to the production provider set from env vars.

    Returns
    -------
    EvidenceResult with items, status, and provider logs.

    Status values
    -------------
    FOUND               : ≥ 1 item returned.
    INSUFFICIENT_EVIDENCE: all providers responded but found nothing.
    ALL_PROVIDERS_FAILED : every provider errored or was unconfigured.

    Notes
    -----
    * Absence of evidence is not evidence of absence — the caller must not
      interpret INSUFFICIENT_EVIDENCE as proof the claim is false.
    * All providers run concurrently (asyncio.gather).
    * If one provider fails, others are unaffected.
    """
    t0 = time.perf_counter()

    # ── Query extraction ──────────────────────────────────────────────────────
    query_set = extract_queries(claim)
    primary_query = query_set.primary_query
    keywords      = query_set.keywords

    logger.info(
        "Evidence search | claim=%r | query=%r | keywords=%s",
        claim[:100], primary_query, keywords,
    )

    # ── Date window ───────────────────────────────────────────────────────────
    from_date: Optional[datetime] = None
    if from_days is not None:
        from_date = datetime.now(timezone.utc) - timedelta(days=from_days)

    # ── Provider selection ────────────────────────────────────────────────────
    active_providers = providers if providers is not None else _build_default_providers()

    # ── Concurrent search ─────────────────────────────────────────────────────
    tasks = [
        _search_one_provider(p, primary_query, max_results, from_date)
        for p in active_providers
    ]
    raw_results: list[tuple[list[EvidenceItem], Optional[str]]] = (
        await asyncio.gather(*tasks)
    )

    all_items:        list[EvidenceItem] = []
    providers_used:   list[str]          = []
    providers_failed: list[str]          = []

    for provider, (items, err) in zip(active_providers, raw_results):
        if err:
            providers_failed.append(f"{provider.name}: {err}")
        else:
            if items:
                providers_used.append(provider.name)
                all_items.extend(items)

    # ── Fallback queries (if primary returned nothing) ────────────────────────
    if not all_items and query_set.fallback_queries:
        for fallback_q in query_set.fallback_queries:
            logger.info("Trying fallback query: %r", fallback_q)
            fb_tasks = [
                _search_one_provider(p, fallback_q, max_results, from_date)
                for p in active_providers
            ]
            fb_results = await asyncio.gather(*fb_tasks)
            for provider, (items, err) in zip(active_providers, fb_results):
                if not err and items:
                    providers_used.append(f"{provider.name}(fallback)")
                    all_items.extend(items)
            if all_items:
                break

    total_found = len(all_items)

    # ── Deduplication ─────────────────────────────────────────────────────────
    deduped = _deduplicate(all_items)

    # ── Relevance scoring ─────────────────────────────────────────────────────
    for item in deduped:
        item.relevance_score = _compute_relevance(item, keywords)

    # ── Sort: relevance desc, then recency ────────────────────────────────────
    deduped.sort(key=lambda x: (x.relevance_score, x.published_at or datetime.min), reverse=True)

    # ── Truncate to max_results ───────────────────────────────────────────────
    final_items = deduped[:max_results]

    # ── Status determination ──────────────────────────────────────────────────
    configured_count = sum(1 for p in active_providers if p.is_configured)
    all_failed       = len(providers_used) == 0 and configured_count > 0

    if final_items:
        status = EvidenceStatus.FOUND
    elif all_failed and not providers_used:
        status = EvidenceStatus.ALL_PROVIDERS_FAILED
    else:
        status = EvidenceStatus.INSUFFICIENT

    search_time_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Evidence search complete | status=%s | found=%d | deduped=%d | "
        "returned=%d | providers_used=%s | time=%.1fms",
        status.value, total_found, len(deduped), len(final_items),
        providers_used, search_time_ms,
    )

    return EvidenceResult(
        claim=claim,
        query=primary_query,
        items=final_items,
        status=status,
        providers_used=providers_used,
        providers_failed=providers_failed,
        total_found=total_found,
        after_dedup=len(deduped),
        search_time_ms=search_time_ms,
    )
