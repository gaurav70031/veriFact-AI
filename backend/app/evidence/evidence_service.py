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
    keyword_overlap : fraction of query keywords that appear in title + description.
                      Title matches count double — the title is the most signal-dense
                      part of a news article.
    entity_bonus    : multi-word proper noun phrases that appear verbatim score higher.
    recency_boost   : exponential decay — articles published today score highest.

    Scoring is strict: partial keyword overlap no longer gives a passable score.
    An article matching only 1 out of 5 keywords scores close to 0.
    """
    title_text = (item.title or "").lower()
    body_text  = (item.description or "").lower()
    full_text  = f"{title_text} {body_text}"

    if not keywords:
        keyword_score = 0.5
    else:
        # Title hits count 2×, body hits count 1×
        title_hits = sum(1 for kw in keywords if kw.lower() in title_text)
        body_hits  = sum(1 for kw in keywords if kw.lower() in body_text)
        # Weighted overlap over a denominator that rewards title presence
        weighted_hits = title_hits * 2 + body_hits
        max_possible  = len(keywords) * 2   # all in title
        raw_overlap   = weighted_hits / max_possible if max_possible else 0

        # Apply a strictness curve: overlap must be >50% to score above 0.5
        # This prevents single common-word matches from ranking highly
        keyword_score = raw_overlap ** 1.5   # squash low overlaps harder

    # Recency boost — exponential decay with 7-day half-life
    recency_score = 0.5
    if item.published_at:
        now     = datetime.now(timezone.utc)
        pub     = item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_days      = max(0.0, (now - pub).total_seconds() / 86400)
        recency_score = 2 ** (-age_days / 7)

    # Keyword overlap is the dominant signal (80%), recency is secondary (20%)
    score = 0.80 * keyword_score + 0.20 * recency_score
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
    query_set     = extract_queries(claim)
    keywords      = query_set.keywords

    # Search APIs (NewsAPI, GNews) perform best with short keyword queries —
    # sending a full question sentence like "Has Gyanesh Kumar stepped down?"
    # returns zero results because no article title contains that exact phrasing.
    # We build a compact query: named entities + key nouns, max 6 words.
    if keywords:
        api_query = " ".join(keywords[:6])
    else:
        # Last resort: strip question words and use the cleaned claim
        api_query = query_set.primary_query.lstrip("Has Did Is Are Was Were Do Does Can Could Should Would ").strip()
        api_query = api_query[:100]

    # RSS feed filtering uses the full primary query (phrase matching is better
    # in the feed-filter code, not the API).
    rss_query = query_set.primary_query

    logger.info(
        "Evidence search | claim=%r | api_query=%r | rss_query=%r | keywords=%s",
        claim[:100], api_query, rss_query[:60], keywords,
    )

    # ── Date window ───────────────────────────────────────────────────────────
    from_date: Optional[datetime] = None
    if from_days is not None:
        from_date = datetime.now(timezone.utc) - timedelta(days=from_days)

    # ── Provider selection ────────────────────────────────────────────────────
    active_providers = providers if providers is not None else _build_default_providers()

    # ── Concurrent search — use api_query for search APIs, rss_query for RSS ─
    from app.evidence.providers.rss_provider import RSSFeedProvider

    tasks = [
        _search_one_provider(
            p,
            rss_query if isinstance(p, RSSFeedProvider) else api_query,
            max_results,
            from_date,
        )
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
            # Shorten fallback queries too — take first 6 words max
            fb_api_q = " ".join(fallback_q.split()[:6])
            logger.info("Trying fallback query: %r (api: %r)", fallback_q, fb_api_q)
            fb_tasks = [
                _search_one_provider(
                    p,
                    fallback_q if isinstance(p, RSSFeedProvider) else fb_api_q,
                    max_results,
                    from_date,
                )
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

    # ── Drop items with very low relevance (likely noise from RSS feeds) ──────
    # Keep at least 1 item if we have any — but filter out clear mismatches.
    MIN_RELEVANCE = 0.15
    relevant = [i for i in deduped if i.relevance_score >= MIN_RELEVANCE]
    if not relevant and deduped:
        # All scored below threshold — keep the best one so we return something
        relevant = deduped[:1]

    # ── Truncate to max_results ───────────────────────────────────────────────
    final_items = relevant[:max_results]

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
        query=api_query,
        items=final_items,
        status=status,
        providers_used=providers_used,
        providers_failed=providers_failed,
        total_found=total_found,
        after_dedup=len(deduped),
        search_time_ms=search_time_ms,
    )
