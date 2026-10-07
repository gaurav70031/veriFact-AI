"""
GNews provider adapter.

API documentation: https://gnews.io/docs/v4
Rate limits     : Free tier — 100 req/day, 10 articles/req
Required env var: GNEWS_API_KEY

Endpoint used   : GET https://gnews.io/api/v4/search
Returned fields : source.name, title, description, url, publishedAt

Copyright note
--------------
Only title, description (≤ 500 chars), and URL are stored.
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

_BASE_URL = "https://gnews.io/api/v4/search"
_TIMEOUT  = httpx.Timeout(10.0, connect=5.0)


class GNewsProvider(EvidenceProvider):
    """
    Adapter for the GNews search endpoint.

    API key is read from the GNEWS_API_KEY environment variable.
    Never passes the key to callers or logs it.
    """

    name        = "gnews"
    source_type = SourceType.GNEWS

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or get_settings().gnews_api_key

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
            raise ProviderAuthError(self.name, "GNEWS_API_KEY is not set.")

        params: dict = {
            "q":        query,
            "max":      min(max_results, 10),   # free tier cap
            "lang":     "en",
            "sortby":   "relevance",
            "apikey":   self._api_key,
        }
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info("[%s] Searching: %r (max=%d)", self.name, query, max_results)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(_BASE_URL, params=params)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, f"Timed out for query: {query!r}")
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(self.name, f"Network error: {exc}")

        if response.status_code == 403:
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
            raise ProviderUnavailableError(self.name, f"Malformed JSON: {exc}")

        # GNews returns errors as {"errors": [...]}
        if "errors" in data:
            errors = data["errors"]
            err_msg = "; ".join(errors) if isinstance(errors, list) else str(errors)
            if "forbidden" in err_msg.lower() or "api key" in err_msg.lower():
                raise ProviderAuthError(self.name, err_msg)
            raise ProviderUnavailableError(self.name, f"API error: {err_msg}")

        articles = data.get("articles", [])
        if not articles:
            raise ProviderNoResultsError(self.name, f"No results for: {query!r}")

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

            source_name = (art.get("source") or {}).get("name") or "Unknown"

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
