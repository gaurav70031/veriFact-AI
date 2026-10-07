"""
NewsAPI provider adapter.

API documentation: https://newsapi.org/docs/endpoints/everything
Rate limits     : Free tier — 100 req/day; Developer — 500 req/day
Required env var: NEWSAPI_KEY

Endpoint used   : GET https://newsapi.org/v2/everything
Returned fields : source.name, title, description, url, publishedAt

Copyright note
--------------
Only title, description (≤ 500 chars), and URL are stored.
Full article content from `content` field is discarded.
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

_BASE_URL = "https://newsapi.org/v2/everything"
_TIMEOUT  = httpx.Timeout(10.0, connect=5.0)


class NewsAPIProvider(EvidenceProvider):
    """
    Adapter for the NewsAPI 'everything' endpoint.

    API key is read from the NEWSAPI_KEY environment variable.
    Never passes the key to callers or logs it.
    """

    name        = "newsapi"
    source_type = SourceType.NEWS_API

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or get_settings().newsapi_key

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
            raise ProviderAuthError(self.name, "NEWSAPI_KEY is not set.")

        params: dict = {
            "q":        query,
            "pageSize": min(max_results, 100),
            "sortBy":   "relevancy",
            "language": "en",
            "apiKey":   self._api_key,
        }
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info("[%s] Searching: %r (max=%d)", self.name, query, max_results)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(_BASE_URL, params=params)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, f"Request timed out for query: {query!r}")
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(self.name, f"Network error: {exc}")

        if response.status_code == 401:
            raise ProviderAuthError(self.name, "Invalid or missing API key.")
        if response.status_code == 429:
            raise ProviderRateLimitError(self.name, "Rate limit exceeded.")
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                self.name, f"Server error HTTP {response.status_code}."
            )
        if response.status_code != 200:
            raise ProviderUnavailableError(
                self.name, f"Unexpected HTTP {response.status_code}."
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ProviderUnavailableError(self.name, f"Malformed JSON response: {exc}")

        # NewsAPI returns status="error" with a code/message on logical errors
        if data.get("status") == "error":
            code = data.get("code", "")
            msg  = data.get("message", "Unknown error")
            if code == "apiKeyInvalid":
                raise ProviderAuthError(self.name, f"API key invalid: {msg}")
            if code == "rateLimited":
                raise ProviderRateLimitError(self.name, msg)
            raise ProviderUnavailableError(self.name, f"API error [{code}]: {msg}")

        articles = data.get("articles", [])
        if not articles:
            raise ProviderNoResultsError(self.name, f"No results for query: {query!r}")

        items: list[EvidenceItem] = []
        for art in articles:
            if not art.get("url") or not art.get("title"):
                continue

            description = (art.get("description") or "").strip()
            description = description[:500] if description else None

            published_at = None
            raw_date = art.get("publishedAt")
            if raw_date:
                try:
                    published_at = datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            source_name = (
                (art.get("source") or {}).get("name")
                or art.get("author")
                or "Unknown"
            )

            items.append(EvidenceItem(
                source_name=source_name,
                title=art["title"].strip()[:500],
                url=art["url"],
                source_type=self.source_type,
                description=description,
                published_at=published_at,
                provider_name=self.name,
            ))

        logger.info("[%s] Returned %d items.", self.name, len(items))
        return items
