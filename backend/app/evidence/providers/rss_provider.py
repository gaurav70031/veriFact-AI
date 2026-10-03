"""
RSS feed provider adapter.

Reads a configurable list of RSS/Atom feed URLs from the environment variable:
  RSS_FEED_URLS=https://feeds.reuters.com/reuters/topNews,https://rss.bbc.co.uk/...

No API key required — RSS is an open standard.

For each feed:
  1. Fetches the XML with httpx.
  2. Parses with feedparser (graceful on malformed feeds).
  3. Filters entries by keyword match (title / summary contain query terms).
  4. Returns at most max_results items across all feeds.

Copyright note
--------------
RSS <description> or <summary> fields are truncated to ≤ 500 characters.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from app.evidence.base_provider import (
    EvidenceProvider,
    ProviderNoResultsError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.evidence.schema import EvidenceItem, SourceType

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# Curated default feed list used when RSS_FEED_URLS is not set.
# Only includes feeds from organisations that distribute via open RSS.
_DEFAULT_FEEDS: list[str] = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.theguardian.com/theguardian/world/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.who.int/feeds/entity/mediacentre/news/en",
]


def _parse_date(entry: dict) -> Optional[datetime]:
    """Try to extract a UTC-aware datetime from a feedparser entry."""
    # feedparser provides published_parsed (time.struct_time)
    published = entry.get("published_parsed")
    if published:
        try:
            return datetime(*published[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    # fallback: raw published string (RFC 2822)
    raw = entry.get("published", "")
    if raw:
        try:
            return parsedate_to_datetime(raw).astimezone(timezone.utc)
        except Exception:
            pass
    return None


def _entry_matches_query(entry: dict, query_terms: list[str]) -> bool:
    """Return True if any query term appears in the entry title or summary."""
    haystack = (
        (entry.get("title") or "") + " " + (entry.get("summary") or "")
    ).lower()
    return any(term.lower() in haystack for term in query_terms)


class RSSFeedProvider(EvidenceProvider):
    """
    Adapter that searches a list of RSS/Atom feeds for query-relevant entries.

    Feed URLs are read from the RSS_FEED_URLS environment variable
    (comma-separated).  Falls back to a curated default list.
    """

    name        = "rss_feed"
    source_type = SourceType.RSS_FEED

    def __init__(self, feed_urls: Optional[list[str]] = None) -> None:
        env_val = os.getenv("RSS_FEED_URLS", "")
        if feed_urls:
            self._feed_urls = feed_urls
        elif env_val.strip():
            self._feed_urls = [u.strip() for u in env_val.split(",") if u.strip()]
        else:
            self._feed_urls = _DEFAULT_FEEDS

    @property
    def is_configured(self) -> bool:
        return bool(self._feed_urls)

    async def search(
        self,
        query:       str,
        max_results: int               = 10,
        from_date:   Optional[datetime] = None,
    ) -> list[EvidenceItem]:
        try:
            import feedparser
        except ImportError:
            raise ProviderUnavailableError(
                self.name,
                "feedparser is not installed. Run: pip install feedparser",
            )

        query_terms = [t.strip() for t in query.split() if len(t.strip()) > 3]
        if not query_terms:
            query_terms = query.split()

        logger.info(
            "[%s] Searching %d feeds for: %r",
            self.name, len(self._feed_urls), query,
        )

        all_items: list[EvidenceItem] = []
        errors: list[str] = []

        for feed_url in self._feed_urls:
            if len(all_items) >= max_results:
                break
            try:
                items = await self._fetch_feed(feed_url, query_terms, from_date, feedparser)
                all_items.extend(items)
            except ProviderTimeoutError:
                errors.append(f"Timeout: {feed_url}")
                logger.warning("[%s] Timeout fetching %s", self.name, feed_url)
            except ProviderUnavailableError as exc:
                errors.append(str(exc))
                logger.warning("[%s] %s", self.name, exc)

        if not all_items:
            if errors:
                raise ProviderUnavailableError(
                    self.name,
                    f"All feeds failed ({len(errors)} errors). First: {errors[0]}",
                )
            raise ProviderNoResultsError(
                self.name, f"No RSS entries matched query: {query!r}"
            )

        logger.info("[%s] Returned %d items from feeds.", self.name, len(all_items))
        return all_items[:max_results]

    async def _fetch_feed(
        self,
        feed_url:    str,
        query_terms: list[str],
        from_date:   Optional[datetime],
        feedparser,
    ) -> list[EvidenceItem]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    feed_url,
                    headers={"User-Agent": "FakeNewsDetector/1.0 feed reader"},
                    follow_redirects=True,
                )
            if resp.status_code >= 400:
                raise ProviderUnavailableError(
                    self.name,
                    f"HTTP {resp.status_code} from {feed_url}",
                )
            raw_xml = resp.text
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, f"Timeout: {feed_url}")
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(self.name, f"Network error {feed_url}: {exc}")

        feed    = feedparser.parse(raw_xml)
        entries = feed.get("entries", [])
        source  = feed.feed.get("title", feed_url)

        items: list[EvidenceItem] = []
        for entry in entries:
            if not _entry_matches_query(entry, query_terms):
                continue

            published_at = _parse_date(entry)
            if from_date and published_at and published_at < from_date:
                continue

            link    = entry.get("link", "")
            title   = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or "").strip()[:500] or None

            if not link or not title:
                continue

            items.append(EvidenceItem(
                source_name=source,
                title=title[:500],
                url=link,
                source_type=self.source_type,
                description=summary,
                published_at=published_at,
                provider_name=self.name,
            ))

        return items
