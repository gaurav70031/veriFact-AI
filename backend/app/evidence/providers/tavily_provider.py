"""
Tavily Search provider adapter.

API documentation: https://docs.tavily.com/docs/tavily-api/rest_api
Required env var : TAVILY_API_KEY

Tavily is an AI-optimised search engine built for RAG applications.
It is selected as the primary web-search provider when available because:

1. Returns search results with AI-extracted content snippets — better
   context than raw meta descriptions from SerpAPI/Brave.
2. Supports `search_depth="advanced"` which performs deeper crawling
   and returns more relevant content for fact-checking.
3. Supports domain filtering via `include_domains` and `exclude_domains`
   — used to prioritise credible sources.
4. Returns `score` per result (Tavily's own relevance estimate).
5. Free tier: 1,000 API credits/month (each search = 1 credit).
   Advanced search = 2 credits per call.

Domain priority tiers
---------------------
Tier 1 (official / government):
  gov, .gov.in, .gov.uk, who.int, un.org, whitehouse.gov, ...
Tier 2 (reputable news / research):
  reuters.com, apnews.com, bbc.com, nytimes.com, washingtonpost.com,
  theguardian.com, nature.com, science.org, pubmed.ncbi.nlm.nih.gov
Tier 3 (mainstream media):
  cnn.com, foxnews.com, nbcnews.com, thehindu.com, ndtv.com, ...

The `include_domains` parameter is not sent when empty — Tavily searches
all domains in that case, which is the correct fallback when no specific
domains are matched.

Important: domain tier does NOT replace relevance scoring.
A result from reuters.com about an unrelated topic scores lower than
a result from an unknown site that directly discusses the claim.
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

_BASE_URL = "https://api.tavily.com/search"
_TIMEOUT  = httpx.Timeout(15.0, connect=5.0)

# Source credibility tiers — higher tier = higher credibility bonus in ranking
# These are used to annotate results, NOT to filter them.
SOURCE_TIERS: dict[str, int] = {
    # Tier 1 — official / government / intergovernmental
    "who.int":                       1,
    "un.org":                        1,
    "cdc.gov":                       1,
    "nih.gov":                       1,
    "fda.gov":                       1,
    "whitehouse.gov":                1,
    "state.gov":                     1,
    "europa.eu":                     1,
    "eci.gov.in":                    1,  # Election Commission of India
    "mea.gov.in":                    1,  # Ministry of External Affairs India
    "pib.gov.in":                    1,  # Press Information Bureau India
    "mohfw.gov.in":                  1,  # Ministry of Health India
    "mha.gov.in":                    1,
    "parliament.uk":                 1,
    # Tier 2 — peer-reviewed / research
    "pubmed.ncbi.nlm.nih.gov":       2,
    "nature.com":                    2,
    "science.org":                   2,
    "thelancet.com":                 2,
    "nejm.org":                      2,
    # Tier 3 — established international news
    "reuters.com":                   3,
    "apnews.com":                    3,
    "bbc.com":                       3,
    "bbc.co.uk":                     3,
    "theguardian.com":               3,
    "nytimes.com":                   3,
    "washingtonpost.com":            3,
    "wsj.com":                       3,
    "ft.com":                        3,
    "economist.com":                 3,
    # Tier 4 — regional / other reputable
    "thehindu.com":                  4,
    "ndtv.com":                      4,
    "indianexpress.com":             4,
    "hindustantimes.com":            4,
    "livemint.com":                  4,
    "timesofindia.com":              4,
    "aljazeera.com":                 4,
    "cnn.com":                       4,
    "nbcnews.com":                   4,
}

# Domains known to publish low-credibility content — results are kept
# (never removed) but flagged so the comparator can weight them lower.
LOW_CREDIBILITY_INDICATORS: frozenset[str] = frozenset({
    "infowars.com", "breitbart.com", "naturalnews.com",
    "worldnewsdailyreport.com", "empirenews.net",
})


def _classify_source(url: str) -> tuple[SourceType, int, bool]:
    """
    Classify a URL by source type, trust tier, and low-credibility flag.

    Returns
    -------
    (source_type, trust_tier, is_low_credibility)
      trust_tier 1 = official, 2 = research, 3-4 = established news, 0 = unknown
    """
    from urllib.parse import urlparse
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lstrip("www.").lower()
    except Exception:
        return SourceType.WEB_SEARCH, 0, False

    is_low = hostname in LOW_CREDIBILITY_INDICATORS

    # Government domains
    if hostname.endswith((".gov", ".gov.in", ".gov.uk", ".gov.au", ".mil")):
        return SourceType.OFFICIAL, 1, False

    # Known tiers
    for domain, tier in SOURCE_TIERS.items():
        if hostname == domain or hostname.endswith("." + domain):
            if tier <= 2:
                return SourceType.OFFICIAL, tier, False
            return SourceType.WEB_SEARCH, tier, is_low

    return SourceType.WEB_SEARCH, 0, is_low


class TavilyProvider(EvidenceProvider):
    """
    Tavily Search provider — AI-optimised search with content extraction.

    Uses `search_depth="basic"` for speed; set `advanced=True` on the
    constructor for deeper crawling (costs 2 credits per call).
    """

    name        = "tavily"
    source_type = SourceType.WEB_SEARCH

    def __init__(
        self,
        api_key:  Optional[str] = None,
        advanced: bool           = False,
    ) -> None:
        self._api_key  = api_key or get_settings().tavily_api_key
        self._advanced = advanced

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
            raise ProviderAuthError(self.name, "TAVILY_API_KEY is not set.")

        payload: dict = {
            "api_key":      self._api_key,
            "query":        query,
            "search_depth": "advanced" if self._advanced else "basic",
            "max_results":  min(max_results, 10),
            "include_answer": False,     # we don't want the AI answer
            "include_raw_content": False,  # saves credits; snippets are enough
        }

        logger.info(
            "[%s] Searching: %r (depth=%s, max=%d)",
            self.name, query, payload["search_depth"], max_results,
        )

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(_BASE_URL, json=payload)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, f"Request timed out for: {query!r}")
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(self.name, f"Network error: {exc}")

        if resp.status_code == 401:
            raise ProviderAuthError(self.name, "Invalid TAVILY_API_KEY.")
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.name, "Rate limit / quota exceeded.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.name, f"HTTP {resp.status_code}.")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.name, f"Unexpected HTTP {resp.status_code}.")

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderUnavailableError(self.name, f"Malformed JSON: {exc}")

        results = data.get("results", [])
        if not results:
            raise ProviderNoResultsError(self.name, f"No results for: {query!r}")

        items: list[EvidenceItem] = []
        for r in results:
            url   = (r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            if not url or not title:
                continue

            content = (r.get("content") or "").strip()
            snippet = content[:500] if content else None

            # Parse published date — Tavily doesn't always provide this
            published_at: Optional[datetime] = None
            published_date = r.get("published_date") or r.get("published") or ""
            if published_date:
                try:
                    from dateutil import parser as dateutil_parser
                    published_at = dateutil_parser.parse(published_date, fuzzy=True)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            source_type, trust_tier, is_low = _classify_source(url)

            # Extract source name from URL domain
            try:
                from urllib.parse import urlparse
                source_name = urlparse(url).hostname or url
                source_name = source_name.lstrip("www.")
            except Exception:
                source_name = url

            # Encode trust tier and low-credibility flag in provider_name
            # so the comparator can use it for weighting
            provider_tag = self.name
            if trust_tier > 0:
                provider_tag += f":tier{trust_tier}"
            if is_low:
                provider_tag += ":low_credibility"

            items.append(EvidenceItem(
                source_name=source_name,
                title=title[:500],
                url=url,
                source_type=source_type,
                description=snippet,
                published_at=published_at,
                provider_name=provider_tag,
            ))

        logger.info("[%s] Returned %d items.", self.name, len(items))
        return items
