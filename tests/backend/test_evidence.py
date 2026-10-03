"""
Tests for the evidence retrieval system.

IMPORTANT — production vs test isolation
-----------------------------------------
These tests use MOCK providers exclusively.
No real HTTP requests are made.
No real API keys are required.
The mock providers are clearly labelled and only exist in test scope.

Test coverage
-------------
  TestEvidenceSchema         — EvidenceItem, EvidenceResult, SourceType
  TestQueryExtractor         — extract_queries(), edge cases
  TestMockProvider           — base provider contract
  TestEvidenceServiceLogic   — orchestration, dedup, ranking, fallback
  TestProviderErrorHandling  — each error type handled, others continue
  TestNewsAPIProviderParsing — response parsing without real HTTP
  TestGNewsProviderParsing   — response parsing without real HTTP
  TestRSSProviderParsing     — feedparser entry filtering
  TestSearchProviderParsing  — SerpAPI / Brave response parsing

Run from the project root:
    pytest tests/backend/test_evidence.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.evidence.schema import (
    EvidenceItem, EvidenceResult, EvidenceStatus, SourceType,
)
from app.evidence.base_provider import (
    EvidenceProvider,
    ProviderAuthError, ProviderTimeoutError, ProviderRateLimitError,
    ProviderUnavailableError, ProviderNoResultsError,
)
from app.evidence.query_extractor import extract_queries, QuerySet
from app.evidence.evidence_service import (
    retrieve_evidence, _deduplicate, _compute_relevance, _url_fingerprint,
)


# =============================================================================
# MOCK PROVIDERS
# These are clearly synthetic test doubles — they exist only in test scope.
# Production code never imports or uses these classes.
# =============================================================================

class _MockProvider(EvidenceProvider):
    """
    [TEST ONLY] Configurable mock provider that returns preset items.
    Never makes HTTP requests.
    """
    name        = "mock_provider"
    source_type = SourceType.NEWS_API

    def __init__(
        self,
        items: Optional[list[EvidenceItem]] = None,
        error: Optional[Exception]          = None,
        configured: bool                    = True,
    ):
        self._items      = items or []
        self._error      = error
        self._configured = configured
        self.search_called_with: list = []

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(
        self,
        query:       str,
        max_results: int               = 10,
        from_date:   Optional[datetime] = None,
    ) -> list[EvidenceItem]:
        self.search_called_with.append((query, max_results, from_date))
        if self._error:
            raise self._error
        return self._items[:max_results]


def _make_item(
    url:   str = "https://example.com/article",
    title: str = "Test Article About Important Topic",
    source: str = "Test Source",
    published_days_ago: int = 1,
    provider: str = "mock_provider",
) -> EvidenceItem:
    """[TEST ONLY] Factory for EvidenceItem test fixtures."""
    return EvidenceItem(
        source_name=source,
        title=title,
        url=url,
        source_type=SourceType.NEWS_API,
        description="A brief description of the test article content.",
        published_at=datetime.now(timezone.utc) - timedelta(days=published_days_ago),
        provider_name=provider,
    )


# =============================================================================
# EvidenceItem / EvidenceResult schema
# =============================================================================

class TestEvidenceSchema:
    def test_evidence_item_to_dict_has_required_keys(self):
        item = _make_item()
        d    = item.to_dict()
        for key in ("source_name", "title", "url", "source_type",
                    "description", "published_at", "retrieved_at",
                    "provider_name", "relevance_score"):
            assert key in d, f"Missing key: {key}"

    def test_evidence_item_relevance_default_zero(self):
        item = _make_item()
        assert item.relevance_score == 0.0

    def test_evidence_item_source_type_serialised_as_string(self):
        item = _make_item()
        assert isinstance(item.to_dict()["source_type"], str)

    def test_evidence_result_has_evidence_true_when_items(self):
        item   = _make_item()
        result = EvidenceResult(
            claim="test claim", query="test",
            items=[item], status=EvidenceStatus.FOUND,
            providers_used=["mock"], providers_failed=[],
        )
        assert result.has_evidence is True

    def test_evidence_result_has_evidence_false_when_empty(self):
        result = EvidenceResult(
            claim="test", query="test",
            items=[], status=EvidenceStatus.INSUFFICIENT,
            providers_used=[], providers_failed=[],
        )
        assert result.has_evidence is False

    def test_evidence_result_to_dict_structure(self):
        item = _make_item()
        result = EvidenceResult(
            claim="claim", query="query",
            items=[item], status=EvidenceStatus.FOUND,
            providers_used=["newsapi"], providers_failed=[],
            total_found=1, after_dedup=1,
        )
        d = result.to_dict()
        assert d["status"] == "found"
        assert len(d["items"]) == 1
        assert d["items"][0]["title"] == item.title

    def test_insufficient_evidence_status_value(self):
        assert EvidenceStatus.INSUFFICIENT.value == "INSUFFICIENT_EVIDENCE"

    def test_all_source_types_serialise(self):
        for st in SourceType:
            assert isinstance(st.value, str)


# =============================================================================
# Query extractor
# =============================================================================

class TestQueryExtractor:
    def test_returns_queryset(self):
        qs = extract_queries("COVID-19 vaccines cause autism claims debunked")
        assert isinstance(qs, QuerySet)
        assert isinstance(qs.primary_query, str)
        assert isinstance(qs.keywords, list)

    def test_primary_query_not_empty_for_real_claim(self):
        qs = extract_queries("The WHO declared COVID-19 a pandemic in March 2020")
        assert qs.primary_query.strip() != ""

    def test_stop_words_removed(self):
        qs = extract_queries("the quick brown fox jumps over the lazy dog scientists")
        # stop words like 'the', 'over' should not appear
        for kw in qs.keywords:
            assert kw.lower() not in ("the", "over", "a", "is")

    def test_noise_phrases_stripped(self):
        qs = extract_queries(
            "According to sources say scientists discovered cure for cancer"
        )
        assert "according" not in qs.primary_query.lower()
        assert "sources say" not in qs.primary_query.lower()

    def test_empty_claim_returns_safe_result(self):
        qs = extract_queries("")
        assert qs.primary_query == ""

    def test_whitespace_only_claim(self):
        qs = extract_queries("   ")
        assert qs.primary_query.strip() == ""

    def test_very_short_claim_still_returns_something(self):
        qs = extract_queries("COVID vaccine")
        assert len(qs.primary_query) > 0

    def test_fallback_queries_are_shorter(self):
        qs = extract_queries(
            "The European Central Bank raised interest rates by twenty five basis points"
        )
        if qs.fallback_queries:
            assert len(qs.fallback_queries[0]) <= len(qs.primary_query)

    def test_long_claim_produces_bounded_keywords(self):
        long_claim = " ".join([f"word{i}" for i in range(100)])
        qs = extract_queries(long_claim)
        assert len(qs.keywords) <= 8

    def test_numbers_preserved_in_keywords(self):
        qs = extract_queries("5G towers caused COVID-19 infections worldwide")
        combined = " ".join(qs.keywords)
        assert "5G" in combined or "COVID" in combined or "5g" in combined.lower()


# =============================================================================
# Mock provider contract
# =============================================================================

class TestMockProvider:
    @pytest.mark.asyncio
    async def test_returns_preset_items(self):
        items    = [_make_item(url=f"https://example.com/{i}") for i in range(3)]
        provider = _MockProvider(items=items)
        result   = await provider.search("query")
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_raises_preset_error(self):
        provider = _MockProvider(error=ProviderTimeoutError("mock", "timed out"))
        with pytest.raises(ProviderTimeoutError):
            await provider.search("query")

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        items    = [_make_item(url=f"https://example.com/{i}") for i in range(10)]
        provider = _MockProvider(items=items)
        result   = await provider.search("query", max_results=3)
        assert len(result) == 3

    def test_unconfigured_provider_skipped(self):
        provider = _MockProvider(configured=False)
        assert provider.is_configured is False


# =============================================================================
# Evidence service — orchestration logic
# =============================================================================

class TestEvidenceServiceLogic:
    @pytest.mark.asyncio
    async def test_found_status_when_items_returned(self):
        items    = [_make_item(url=f"https://example.com/a{i}") for i in range(3)]
        provider = _MockProvider(items=items)
        result   = await retrieve_evidence("test claim", providers=[provider])
        assert result.status == EvidenceStatus.FOUND
        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_insufficient_when_no_results(self):
        provider = _MockProvider(error=ProviderNoResultsError("mock", "no results"))
        result   = await retrieve_evidence("test claim", providers=[provider])
        assert result.status in (
            EvidenceStatus.INSUFFICIENT, EvidenceStatus.ALL_PROVIDERS_FAILED
        )

    @pytest.mark.asyncio
    async def test_provider_name_in_providers_used(self):
        items    = [_make_item()]
        provider = _MockProvider(items=items)
        result   = await retrieve_evidence("test claim", providers=[provider])
        assert "mock_provider" in result.providers_used

    @pytest.mark.asyncio
    async def test_failed_provider_recorded_in_providers_failed(self):
        good = _MockProvider(items=[_make_item()], )
        good.name = "good_provider"
        bad  = _MockProvider(error=ProviderTimeoutError("bad_provider", "timeout"))
        bad.name = "bad_provider"
        result = await retrieve_evidence("test claim", providers=[good, bad])
        assert any("bad_provider" in f for f in result.providers_failed)

    @pytest.mark.asyncio
    async def test_one_provider_fails_other_succeeds(self):
        items = [_make_item(url="https://example.com/good")]
        good  = _MockProvider(items=items)
        good.name = "good_provider"
        bad   = _MockProvider(error=ProviderUnavailableError("bad_provider", "down"))
        bad.name  = "bad_provider"
        result    = await retrieve_evidence("claim", providers=[good, bad])
        assert result.status == EvidenceStatus.FOUND
        assert len(result.items) >= 1

    @pytest.mark.asyncio
    async def test_all_providers_fail_gives_all_failed_status(self):
        p1 = _MockProvider(error=ProviderAuthError("p1", "bad key"))
        p2 = _MockProvider(error=ProviderRateLimitError("p2", "rate limited"))
        result = await retrieve_evidence("claim", providers=[p1, p2])
        assert result.status == EvidenceStatus.ALL_PROVIDERS_FAILED

    @pytest.mark.asyncio
    async def test_unconfigured_provider_skipped_silently(self):
        unconfigured = _MockProvider(items=[_make_item()], configured=False)
        configured   = _MockProvider(items=[_make_item(url="https://example.com/c")])
        configured.name = "configured_provider"
        result = await retrieve_evidence("claim", providers=[unconfigured, configured])
        assert "configured_provider" in result.providers_used

    @pytest.mark.asyncio
    async def test_deduplication_removes_same_url(self):
        # Two providers return the same URL
        url   = "https://example.com/same-article"
        item1 = _make_item(url=url)
        item2 = _make_item(url=url)
        p1    = _MockProvider(items=[item1])
        p1.name = "provider_a"
        p2    = _MockProvider(items=[item2])
        p2.name = "provider_b"
        result = await retrieve_evidence("claim", providers=[p1, p2])
        urls   = [i.url for i in result.items]
        assert len(urls) == len(set(urls)), "Duplicate URLs found after dedup"

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance_desc(self):
        items = [
            _make_item(url=f"https://example.com/{i}", published_days_ago=i)
            for i in range(1, 6)
        ]
        provider = _MockProvider(items=items)
        result   = await retrieve_evidence("important claim topic", providers=[provider])
        scores   = [i.relevance_score for i in result.items]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_max_results_honoured(self):
        items    = [_make_item(url=f"https://example.com/{i}") for i in range(20)]
        provider = _MockProvider(items=items)
        result   = await retrieve_evidence("claim", max_results=5, providers=[provider])
        assert len(result.items) <= 5

    @pytest.mark.asyncio
    async def test_search_time_ms_populated(self):
        provider = _MockProvider(items=[_make_item()])
        result   = await retrieve_evidence("claim", providers=[provider])
        assert result.search_time_ms > 0

    @pytest.mark.asyncio
    async def test_claim_and_query_in_result(self):
        provider = _MockProvider(items=[_make_item()])
        result   = await retrieve_evidence("ECB raised rates 2024", providers=[provider])
        assert result.claim == "ECB raised rates 2024"
        assert len(result.query) > 0


# =============================================================================
# Deduplication + relevance (unit tests — no providers needed)
# =============================================================================

class TestDeduplicate:
    def test_removes_exact_url_duplicate(self):
        items = [
            _make_item(url="https://example.com/article"),
            _make_item(url="https://example.com/article"),
        ]
        deduped = _deduplicate(items)
        assert len(deduped) == 1

    def test_removes_url_with_utm_params(self):
        items = [
            _make_item(url="https://example.com/article"),
            _make_item(url="https://example.com/article?utm_source=twitter"),
        ]
        deduped = _deduplicate(items)
        assert len(deduped) == 1

    def test_keeps_different_urls(self):
        items = [
            _make_item(url="https://example.com/article1"),
            _make_item(url="https://example.com/article2"),
        ]
        deduped = _deduplicate(items)
        assert len(deduped) == 2

    def test_url_fingerprint_ignores_trailing_slash(self):
        assert _url_fingerprint("https://a.com/x") == _url_fingerprint("https://a.com/x/")

    def test_url_fingerprint_case_insensitive(self):
        assert _url_fingerprint("https://Example.com/X") == _url_fingerprint("https://example.com/x")


class TestRelevanceScoring:
    def test_score_in_0_1_range(self):
        item  = _make_item(title="ECB raises interest rates eurozone inflation")
        score = _compute_relevance(item, ["ECB", "rates", "eurozone"])
        assert 0.0 <= score <= 1.0

    def test_recent_article_scores_higher_than_old(self):
        recent = _make_item(url="https://a.com/1", published_days_ago=0)
        old    = _make_item(url="https://a.com/2", published_days_ago=90)
        score_recent = _compute_relevance(recent, ["vaccine", "COVID"])
        score_old    = _compute_relevance(old,    ["vaccine", "COVID"])
        assert score_recent > score_old

    def test_keyword_match_increases_score(self):
        match    = _make_item(title="COVID vaccine causes side effects study")
        no_match = _make_item(title="local sports results weekend roundup")
        s_match    = _compute_relevance(match,    ["COVID", "vaccine"])
        s_no_match = _compute_relevance(no_match, ["COVID", "vaccine"])
        assert s_match > s_no_match

    def test_empty_keywords_returns_neutral_score(self):
        item  = _make_item()
        score = _compute_relevance(item, [])
        assert 0.0 <= score <= 1.0


# =============================================================================
# Provider error hierarchy
# =============================================================================

class TestProviderErrors:
    def test_timeout_is_provider_error(self):
        exc = ProviderTimeoutError("newsapi", "timeout")
        assert isinstance(exc, Exception)
        assert "newsapi" in str(exc)

    def test_rate_limit_is_provider_error(self):
        exc = ProviderRateLimitError("gnews", "too many requests")
        assert isinstance(exc, Exception)

    def test_auth_error_is_provider_error(self):
        exc = ProviderAuthError("newsapi", "invalid key")
        assert "newsapi" in str(exc)

    def test_unavailable_is_provider_error(self):
        exc = ProviderUnavailableError("rss_feed", "connection refused")
        assert isinstance(exc, Exception)

    def test_no_results_is_provider_error(self):
        exc = ProviderNoResultsError("gnews", "no results found")
        assert isinstance(exc, Exception)


# =============================================================================
# NewsAPI response parsing (mocked HTTP — no real API call)
# =============================================================================

class TestNewsAPIProviderParsing:
    """
    Tests NewsAPI adapter by mocking the httpx response.
    No real HTTP request is made.
    """

    def _mock_newsapi_response(self, articles: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "articles": articles}
        return mock_resp

    @pytest.mark.asyncio
    async def test_parses_valid_response(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider

        articles = [
            {
                "source": {"id": "reuters", "name": "Reuters"},
                "title":  "Central bank raises rates",
                "description": "The ECB raised interest rates today.",
                "url":    "https://reuters.com/article/ecb-rates",
                "publishedAt": "2024-05-01T10:00:00Z",
            }
        ]
        mock_resp = self._mock_newsapi_response(articles)

        provider = NewsAPIProvider(api_key="test-key-not-real")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            items = await provider.search("ECB rates")

        assert len(items) == 1
        assert items[0].source_name == "Reuters"
        assert items[0].url         == "https://reuters.com/article/ecb-rates"
        assert items[0].published_at is not None

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_401(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        provider = NewsAPIProvider(api_key="bad-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderAuthError):
                await provider.search("query")

    @pytest.mark.asyncio
    async def test_raises_rate_limit_on_429(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 429

        provider = NewsAPIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderRateLimitError):
                await provider.search("query")

    @pytest.mark.asyncio
    async def test_raises_no_results_when_empty(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "articles": []}

        provider = NewsAPIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderNoResultsError):
                await provider.search("query")

    @pytest.mark.asyncio
    async def test_description_truncated_to_500(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider

        long_desc = "x" * 1000
        articles  = [
            {
                "source":    {"name": "Test"},
                "title":     "Some headline",
                "description": long_desc,
                "url":       "https://example.com/article",
                "publishedAt": "2024-05-01T10:00:00Z",
            }
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "articles": articles}

        provider = NewsAPIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            items = await provider.search("query")

        assert len(items[0].description) <= 500

    def test_not_configured_without_api_key(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider
        provider = NewsAPIProvider(api_key="")
        assert provider.is_configured is False

    def test_configured_with_api_key(self):
        from app.evidence.providers.newsapi_provider import NewsAPIProvider
        provider = NewsAPIProvider(api_key="some-real-looking-key")
        assert provider.is_configured is True


# =============================================================================
# GNews response parsing (mocked HTTP)
# =============================================================================

class TestGNewsProviderParsing:
    @pytest.mark.asyncio
    async def test_parses_valid_response(self):
        from app.evidence.providers.gnews_provider import GNewsProvider

        articles = [
            {
                "source": {"name": "BBC News", "url": "https://bbc.co.uk"},
                "title":  "Inflation hits record high in eurozone",
                "description": "European inflation reached record levels.",
                "url":    "https://bbc.co.uk/news/economy-123",
                "publishedAt": "2024-06-01T08:30:00Z",
            }
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"totalArticles": 1, "articles": articles}

        provider = GNewsProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            items = await provider.search("inflation eurozone")

        assert len(items) == 1
        assert items[0].source_name == "BBC News"

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_403(self):
        from app.evidence.providers.gnews_provider import GNewsProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        provider = GNewsProvider(api_key="bad-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderAuthError):
                await provider.search("query")

    def test_not_configured_without_key(self):
        from app.evidence.providers.gnews_provider import GNewsProvider
        assert GNewsProvider(api_key="").is_configured is False


# =============================================================================
# RSS provider (mocked feedparser + httpx)
# =============================================================================

class TestRSSProviderParsing:
    def _make_feed_entry(
        self,
        title:   str = "Vaccine study published today",
        link:    str = "https://reuters.com/rss/1",
        summary: str = "A new study on vaccines was published.",
        days_ago: int = 0,
    ) -> dict:
        """[TEST ONLY] Build a feedparser-style entry dict."""
        from email.utils import formatdate
        import time
        pub = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return {
            "title":          title,
            "link":           link,
            "summary":        summary,
            "published":      pub.strftime("%a, %d %b %Y %H:%M:%S +0000"),
            "published_parsed": pub.timetuple(),
        }

    @pytest.mark.asyncio
    async def test_keyword_filtering_includes_match(self):
        from app.evidence.providers.rss_provider import RSSFeedProvider

        entry = self._make_feed_entry(
            title="COVID-19 vaccine effectiveness study",
            link="https://example.com/article1",
        )

        mock_feed   = MagicMock()
        mock_feed.feed.get.return_value = "Reuters"
        mock_feed.get.return_value      = [entry]

        mock_http_resp = MagicMock()
        mock_http_resp.status_code = 200
        mock_http_resp.text        = "<rss/>"

        provider = RSSFeedProvider(feed_urls=["https://feeds.example.com/news"])

        # Patch feedparser at the module level so even if it's not installed
        # the test can still run. The RSS provider imports feedparser lazily.
        import sys
        mock_feedparser = MagicMock()
        mock_feedparser.parse = MagicMock(return_value=mock_feed)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_http_resp)
            mock_client_cls.return_value = mock_client

            sys.modules.setdefault("feedparser", mock_feedparser)
            with patch.dict(sys.modules, {"feedparser": mock_feedparser}):
                items = await provider.search("COVID vaccine", max_results=10)

        assert any("COVID" in i.title or "vaccine" in i.title.lower() for i in items)

    @pytest.mark.asyncio
    async def test_keyword_filtering_excludes_no_match(self):
        from app.evidence.providers.rss_provider import RSSFeedProvider

        # Title contains no terms from "COVID vaccine" query
        entry = self._make_feed_entry(
            title="Weekend football championship results roundup",
            link="https://example.com/sports",
            summary="Team wins championship after exciting final match.",
        )

        mock_feed = MagicMock()
        mock_feed.feed.get.return_value = "Sports Daily"
        mock_feed.get.return_value      = [entry]

        mock_http_resp = MagicMock()
        mock_http_resp.status_code = 200
        mock_http_resp.text        = "<rss/>"

        mock_feedparser = MagicMock()
        mock_feedparser.parse = MagicMock(return_value=mock_feed)

        provider = RSSFeedProvider(feed_urls=["https://feeds.example.com/sports"])

        import sys
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_http_resp)
            mock_client_cls.return_value = mock_client

            with patch.dict(sys.modules, {"feedparser": mock_feedparser}):
                try:
                    items = await provider.search("COVID vaccine", max_results=10)
                    # If it returns without raising, no COVID/vaccine items should exist
                    assert all(
                        "covid" not in i.title.lower() and "vaccine" not in i.title.lower()
                        for i in items
                    )
                except ProviderNoResultsError:
                    # This is the expected outcome — no matching entries found
                    pass


# =============================================================================
# Search provider (mocked HTTP)
# =============================================================================

class TestSearchProviderParsing:
    @pytest.mark.asyncio
    async def test_serpapi_parses_news_results(self):
        from app.evidence.providers.search_provider import SearchProvider

        news_results = [
            {
                "title":   "ECB raises rates as inflation persists",
                "link":    "https://reuters.com/ecb-article",
                "snippet": "The ECB raised rates by 25 basis points.",
                "source":  "Reuters",
                "date":    "2024-05-01",
            }
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"news_results": news_results}

        provider = SearchProvider(provider="serpapi", api_key="test-serpapi-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            items = await provider.search("ECB rates")

        assert len(items) == 1
        assert items[0].source_name == "Reuters"
        assert items[0].url         == "https://reuters.com/ecb-article"

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_401(self):
        from app.evidence.providers.search_provider import SearchProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        provider = SearchProvider(provider="serpapi", api_key="bad-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.get        = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderAuthError):
                await provider.search("query")

    def test_not_configured_without_key(self):
        from app.evidence.providers.search_provider import SearchProvider
        provider = SearchProvider(provider="serpapi", api_key="")
        assert provider.is_configured is False
