"""
Web search provider adapter.

Supports two backends, selected by the SEARCH_PROVIDER env var:
  serpapi  — SerpAPI Google Search (https://serpapi.com/search-api)
             Required env var: SERPAPI_KEY
  brave    — Brave Search API (https://api.search.brave.com/res/v1/news/search)
             Required env var: BRAVE_SEARCH_KEY

If SEARCH_PROVIDER is not set or the key is missing, this provider is
skipped by the orchestrator (is_configured returns False).

This provider is the lowest-priority fallback — it runs only when
NewsAPI and GNews have returned fewer than the requested results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.evidence.base_provider import (
    EvidenceProvider,
    ProviderAuthError,
    ProviderNoResultsError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.evidence.schema import EvidenceItem, SourceType
from app.core.config import get_settings

logger = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(12.0, connect=5.0)


# ── SerpAPI backend ───────────────────────────────────────────────────────────

async def _search_serpapi(
    query:       str,
    api_key:     str,
    max_results: int,
    from_date:   Optional[datetime],
) -> list[EvidenceItem]:
    params: dict = {
        "q":      query,
        "num":    min(max_results, 10),
        "tbm":    "nws",          # news search
        "api_key": api_key,
    }
    if from_date:
        params["tbs"] = f"cdr:1,cd_min:{from_date.strftime('%m/%d/%Y')}"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get("https://serpapi.com/search", params=params)
    except httpx.TimeoutException:
        raise ProviderTimeoutError("search_serpapi", "Request timed out.")
    except httpx.RequestError as exc:
        raise ProviderUnavailableError("search_serpapi", f"Network error: {exc}")

    if resp.status_code == 401:
        raise ProviderAuthError("search_serpapi", "Invalid SERPAPI_KEY.")
    if resp.status_code == 429:
        raise ProviderRateLimitError("search_serpapi", "Rate limit exceeded.")
    if resp.status_code >= 500:
        raise ProviderUnavailableError("search_serpapi", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception as exc:
        raise ProviderUnavailableError("search_serpapi", f"Malformed JSON: {exc}")

    if "error" in data:
        err = data["error"]
        if "invalid api key" in str(err).lower():
            raise ProviderAuthError("search_serpapi", str(err))
        raise ProviderUnavailableError("search_serpapi", str(err))

    news_results = data.get("news_results", [])
    items = []
    for r in news_results:
        if not r.get("link") or not r.get("title"):
            continue
        snippet = (r.get("snippet") or "").strip()[:500] or None

        published_at = None
        date_str = r.get("date", "")
        if date_str:
            try:
                from dateutil import parser as dateutil_parser
                published_at = dateutil_parser.parse(date_str, fuzzy=True)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except Exception:
                pass

        items.append(EvidenceItem(
            source_name=r.get("source", "Web Search"),
            title=r["title"].strip()[:500],
            url=r["link"],
            source_type=SourceType.WEB_SEARCH,
            description=snippet,
            published_at=published_at,
            provider_name="search_serpapi",
        ))
    return items


# ── Brave Search backend ──────────────────────────────────────────────────────

async def _search_brave(
    query:       str,
    api_key:     str,
    max_results: int,
    from_date:   Optional[datetime],
) -> list[EvidenceItem]:
    headers = {
        "Accept":              "application/json",
        "Accept-Encoding":     "gzip",
        "X-Subscription-Token": api_key,
    }
    params: dict = {
        "q":     query,
        "count": min(max_results, 20),
    }
    if from_date:
        params["freshness"] = "pw"   # past week as minimum recency filter

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/news/search",
                headers=headers,
                params=params,
            )
    except httpx.TimeoutException:
        raise ProviderTimeoutError("search_brave", "Request timed out.")
    except httpx.RequestError as exc:
        raise ProviderUnavailableError("search_brave", f"Network error: {exc}")

    if resp.status_code == 401:
        raise ProviderAuthError("search_brave", "Invalid BRAVE_SEARCH_KEY.")
    if resp.status_code == 429:
        raise ProviderRateLimitError("search_brave", "Rate limit exceeded.")
    if resp.status_code >= 500:
        raise ProviderUnavailableError("search_brave", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception as exc:
        raise ProviderUnavailableError("search_brave", f"Malformed JSON: {exc}")

    results = data.get("results", [])
    items = []
    for r in results:
        if not r.get("url") or not r.get("title"):
            continue
        desc = (r.get("description") or "").strip()[:500] or None

        published_at = None
        age = r.get("age", "")
        if age:
            try:
                from dateutil import parser as dateutil_parser
                published_at = dateutil_parser.parse(age, fuzzy=True)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except Exception:
                pass

        items.append(EvidenceItem(
            source_name=r.get("meta_url", {}).get("hostname", "Brave Search"),
            title=r["title"].strip()[:500],
            url=r["url"],
            source_type=SourceType.WEB_SEARCH,
            description=desc,
            published_at=published_at,
            provider_name="search_brave",
        ))
    return items


# ── Unified adapter ───────────────────────────────────────────────────────────

class SearchProvider(EvidenceProvider):
    """
    Configurable web search adapter.

    Selects SerpAPI or Brave Search based on SEARCH_PROVIDER env var.
    Skipped gracefully when no key is configured.
    """

    name        = "web_search"
    source_type = SourceType.WEB_SEARCH

    def __init__(
        self,
        provider:    Optional[str] = None,
        api_key:     Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._backend = (
            (provider or settings.search_provider).lower()
        )
        if self._backend == "serpapi":
            self._api_key = api_key or settings.serpapi_key
        elif self._backend == "brave":
            self._api_key = api_key or settings.brave_search_key
        else:
            logger.warning(
                "Unknown SEARCH_PROVIDER '%s'. Valid options: serpapi, brave.",
                self._backend,
            )
            self._api_key = ""

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def search(
        self,
        query:       str,
        max_results: int               = 10,
        from_date:   Optional[datetime] = None,
    ) -> list[EvidenceItem]:
        if not self.is_configured:
            raise ProviderAuthError(
                self.name,
                f"No API key configured for search backend '{self._backend}'. "
                "Set SERPAPI_KEY or BRAVE_SEARCH_KEY.",
            )

        logger.info(
            "[%s/%s] Searching: %r (max=%d)",
            self.name, self._backend, query, max_results,
        )

        if self._backend == "serpapi":
            items = await _search_serpapi(query, self._api_key, max_results, from_date)
        elif self._backend == "brave":
            items = await _search_brave(query, self._api_key, max_results, from_date)
        else:
            raise ProviderUnavailableError(
                self.name, f"Unsupported backend: {self._backend}"
            )

        if not items:
            raise ProviderNoResultsError(
                self.name, f"No results for query: {query!r}"
            )

        logger.info("[%s] Returned %d items.", self.name, len(items))
        return items
