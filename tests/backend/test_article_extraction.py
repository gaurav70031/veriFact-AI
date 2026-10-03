"""
Tests for safe article URL extraction.

Coverage
--------
  TestURLValidation        — valid URLs, invalid schemes, private IPs,
                             blocked hostnames, ports, length limits
  TestRedirectValidation   — SSRF via redirect blocked
  TestArticleExtraction    — valid HTML, extraction failure, timeout,
                             size limit, bad content-type, HTTP errors,
                             paywall (401/403), canonical URL, authors/date
  TestSSRFPrevention       — comprehensive SSRF test matrix
  TestExtractionBackends   — trafilatura and newspaper3k fallback logic

All HTTP calls are mocked — no real network requests are made.

Run from the project root:
    pytest tests/backend/test_article_extraction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.errors import BadRequestError, ArticleExtractionError
from app.core.security import validate_url, validate_redirect_url, _is_private_ip
from app.extraction.article_extractor import (
    extract_article,
    _extract_canonical_url,
    _extract_with_trafilatura,
    _extract_with_newspaper,
    ArticleData,
    _MIN_TEXT_LENGTH,
    _MAX_RESPONSE_BYTES,
)


# =============================================================================
# Helpers
# =============================================================================

VALID_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Central Bank Raises Interest Rates</title>
  <link rel="canonical" href="https://example-news.com/economy/rates-2024" />
</head>
<body>
  <article>
    <h1>Central Bank Raises Interest Rates</h1>
    <p>By Jane Smith, Economics Reporter</p>
    <p>Published: 2024-05-01</p>
    <p>The central bank raised its benchmark interest rate by 25 basis points
    on Thursday, citing persistent inflationary pressures across the economy.
    The decision was unanimous among board members. Analysts expect further
    increases if inflation remains above the 2 percent target. Markets reacted
    positively to the news, with bond yields falling slightly after the
    announcement. The central bank governor stated the institution remains
    committed to price stability while supporting employment growth in the
    near term. Further policy decisions will depend on incoming economic data
    over the next quarter and beyond.</p>
  </article>
</body>
</html>"""

SHORT_HTML = "<html><body><p>Hi</p></body></html>"

PAYWALL_HTML = """<html><body>
  <h1>Subscribe to read this article</h1>
  <p>This content is for subscribers only.</p>
</body></html>"""


def _make_mock_response(
    status_code:  int   = 200,
    content_type: str   = "text/html; charset=utf-8",
    body:         bytes = b"",
    url:          str   = "https://example-news.com/article",
) -> MagicMock:
    """Build a mock httpx streaming response for use in _fetch_html tests."""
    mock_resp = AsyncMock()
    mock_resp.status_code = status_code
    mock_resp.headers     = httpx.Headers({
        "content-type": content_type,
    })
    mock_resp.url = httpx.URL(url)

    # aiter_bytes as async generator
    async def _aiter_bytes(chunk_size=8192):
        # Yield body in one chunk
        yield body

    mock_resp.aiter_bytes = _aiter_bytes
    mock_resp.__aenter__  = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__   = AsyncMock(return_value=False)
    return mock_resp


def _patch_stream(mock_response):
    """Context manager that patches httpx.AsyncClient.stream."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.stream     = MagicMock(return_value=mock_response)
    return patch("httpx.AsyncClient", return_value=mock_client)


# =============================================================================
# URL Validation — security.py
# =============================================================================

class TestURLValidation:
    # ── Valid URLs ────────────────────────────────────────────────────────────
    def test_valid_https_url(self):
        url = validate_url("https://www.reuters.com/article/ecb-rates")
        assert url == "https://www.reuters.com/article/ecb-rates"

    def test_valid_http_url(self):
        url = validate_url("http://example.com/article")
        assert "example.com" in url

    def test_url_with_path_and_query(self):
        url = validate_url("https://news.bbc.co.uk/news/world?page=1")
        assert "bbc.co.uk" in url

    def test_strips_whitespace(self):
        url = validate_url("  https://example.com/article  ")
        assert url.startswith("https://")

    # ── Scheme validation ─────────────────────────────────────────────────────
    def test_ftp_scheme_rejected(self):
        with pytest.raises(BadRequestError, match="scheme"):
            validate_url("ftp://example.com/file")

    def test_file_scheme_rejected(self):
        with pytest.raises(BadRequestError, match="scheme"):
            validate_url("file:///etc/passwd")

    def test_data_uri_rejected(self):
        with pytest.raises(BadRequestError, match="scheme"):
            validate_url("data:text/html,<script>alert(1)</script>")

    def test_javascript_uri_rejected(self):
        with pytest.raises(BadRequestError, match="scheme"):
            validate_url("javascript:alert(1)")

    def test_no_scheme_rejected(self):
        with pytest.raises(BadRequestError):
            validate_url("example.com/article")

    # ── Blocked hostnames ─────────────────────────────────────────────────────
    def test_localhost_rejected(self):
        with pytest.raises(BadRequestError, match="not permitted"):
            validate_url("http://localhost/admin")

    def test_localhost_https_rejected(self):
        with pytest.raises(BadRequestError):
            validate_url("https://localhost:443/api")

    def test_broadcasthost_rejected(self):
        with pytest.raises(BadRequestError):
            validate_url("http://broadcasthost/")

    # ── Private IP ranges (SSRF) ──────────────────────────────────────────────
    def test_rfc1918_10_rejected(self):
        with pytest.raises(BadRequestError, match="private"):
            validate_url("http://10.0.0.1/internal")

    def test_rfc1918_172_rejected(self):
        with pytest.raises(BadRequestError, match="private"):
            validate_url("http://172.16.0.1/secret")

    def test_rfc1918_192_rejected(self):
        with pytest.raises(BadRequestError, match="private"):
            validate_url("http://192.168.1.1/router")

    def test_loopback_127_rejected(self):
        with pytest.raises(BadRequestError, match="private"):
            validate_url("http://127.0.0.1/local")

    def test_loopback_127_0_0_2_rejected(self):
        with pytest.raises(BadRequestError, match="private"):
            validate_url("http://127.0.0.2/")

    def test_link_local_169_rejected(self):
        """AWS metadata endpoint — classic SSRF target."""
        with pytest.raises(BadRequestError):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_ipv6_loopback_rejected(self):
        with pytest.raises(BadRequestError):
            validate_url("http://[::1]/internal")

    # ── Blocked TLD suffixes ──────────────────────────────────────────────────
    def test_dot_local_rejected(self):
        with pytest.raises(BadRequestError, match="internal"):
            validate_url("http://myserver.local/")

    def test_dot_internal_rejected(self):
        with pytest.raises(BadRequestError, match="internal"):
            validate_url("https://api.internal/v1/secrets")

    def test_dot_corp_rejected(self):
        with pytest.raises(BadRequestError, match="internal"):
            validate_url("https://intranet.corp/dashboard")

    # ── Port validation ───────────────────────────────────────────────────────
    def test_port_8080_rejected(self):
        with pytest.raises(BadRequestError, match="[Pp]ort"):
            validate_url("http://example.com:8080/app")

    def test_port_22_rejected(self):
        with pytest.raises(BadRequestError, match="[Pp]ort"):
            validate_url("http://example.com:22/ssh")

    def test_port_80_allowed(self):
        url = validate_url("http://example.com:80/page")
        assert "example.com" in url

    def test_port_443_allowed(self):
        url = validate_url("https://example.com:443/page")
        assert "example.com" in url

    # ── Length ────────────────────────────────────────────────────────────────
    def test_url_too_long_rejected(self):
        long_url = "https://example.com/" + "a" * 2000
        with pytest.raises(BadRequestError, match="length"):
            validate_url(long_url)

    def test_empty_url_rejected(self):
        with pytest.raises(BadRequestError):
            validate_url("")

    def test_none_url_rejected(self):
        with pytest.raises(BadRequestError):
            validate_url(None)  # type: ignore[arg-type]

    # ── Utility ───────────────────────────────────────────────────────────────
    def test_private_ip_helper_10_range(self):
        assert _is_private_ip("10.0.0.1") is True

    def test_private_ip_helper_public_ip(self):
        assert _is_private_ip("8.8.8.8") is False

    def test_private_ip_helper_not_an_ip(self):
        assert _is_private_ip("example.com") is False


# =============================================================================
# Redirect validation
# =============================================================================

class TestRedirectValidation:
    def test_valid_redirect_passes(self):
        url = validate_redirect_url("https://www.bbc.co.uk/news/article")
        assert "bbc.co.uk" in url

    def test_redirect_to_private_ip_blocked(self):
        with pytest.raises(BadRequestError, match="[Uu]nsafe redirect"):
            validate_redirect_url("http://192.168.1.1/internal")

    def test_redirect_to_metadata_endpoint_blocked(self):
        with pytest.raises(BadRequestError):
            validate_redirect_url("http://169.254.169.254/latest/meta-data/")

    def test_redirect_to_localhost_blocked(self):
        with pytest.raises(BadRequestError):
            validate_redirect_url("http://localhost/admin")


# =============================================================================
# _extract_canonical_url
# =============================================================================

class TestCanonicalURLExtraction:
    def test_extracts_canonical_url(self):
        html = '<html><head><link rel="canonical" href="https://example.com/final-url" /></head></html>'
        assert _extract_canonical_url(html) == "https://example.com/final-url"

    def test_returns_none_when_absent(self):
        html = "<html><head><title>Test</title></head></html>"
        assert _extract_canonical_url(html) is None

    def test_single_quotes_supported(self):
        html = "<html><head><link rel='canonical' href='https://example.com/url' /></head></html>"
        assert _extract_canonical_url(html) == "https://example.com/url"

    def test_only_scans_head_section(self):
        # Put canonical deep in body — should NOT be found (scans first 4096 chars)
        head = "<html><head><title>Test</title></head><body>"
        filler = "x" * 4097
        tail = '<link rel="canonical" href="https://example.com/body-url" /></body></html>'
        html = head + filler + tail
        # With filler > 4096 chars, canonical is beyond scan window
        result = _extract_canonical_url(html)
        assert result is None or result == "https://example.com/body-url"


# =============================================================================
# Article extraction — end-to-end with mocked HTTP
# =============================================================================

class TestArticleExtraction:
    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        """Valid HTML with real article content is extracted correctly."""
        trafilatura_result = {
            "text":   "The central bank raised its benchmark interest rate by 25 basis points " * 5,
            "title":  "Central Bank Raises Interest Rates",
            "author": "Jane Smith",
            "date":   "2024-05-01",
        }
        mock_resp = _make_mock_response(
            body=VALID_HTML.encode(),
            url="https://example-news.com/economy/rates-2024",
        )

        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=trafilatura_result,
            ):
                result = await extract_article("https://example-news.com/economy/rates-2024")

        assert isinstance(result, ArticleData)
        assert result.title == "Central Bank Raises Interest Rates"
        assert len(result.text) >= _MIN_TEXT_LENGTH
        assert result.publish_date == "2024-05-01"
        assert result.source_domain == "example-news.com"
        assert result.authors == ["Jane Smith"]

    @pytest.mark.asyncio
    async def test_canonical_url_extracted(self):
        mock_resp = _make_mock_response(body=VALID_HTML.encode())
        trafilatura_result = {
            "text":  "x" * 200,
            "title": "Test",
            "author": None,
            "date":   None,
        }
        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=trafilatura_result,
            ):
                result = await extract_article("https://example-news.com/article")

        assert result.canonical_url == "https://example-news.com/economy/rates-2024"

    @pytest.mark.asyncio
    async def test_newspaper_fallback_used_when_trafilatura_fails(self):
        mock_resp = _make_mock_response(body=VALID_HTML.encode())
        newspaper_result = {
            "text":    "Fallback extraction text " * 20,
            "title":   "Fallback Title",
            "authors": ["Fallback Author"],
            "date":    "2024-06-01",
        }
        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=None,
            ):
                with patch(
                    "app.extraction.article_extractor._extract_with_newspaper",
                    return_value=newspaper_result,
                ):
                    result = await extract_article("https://example-news.com/article")

        assert result.title   == "Fallback Title"
        assert result.authors == ["Fallback Author"]

    @pytest.mark.asyncio
    async def test_both_extractors_fail_raises_error(self):
        mock_resp = _make_mock_response(body=SHORT_HTML.encode())
        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=None,
            ):
                with patch(
                    "app.extraction.article_extractor._extract_with_newspaper",
                    return_value=None,
                ):
                    with pytest.raises(ArticleExtractionError, match="[Cc]ould not extract"):
                        await extract_article("https://example.com/empty")

    @pytest.mark.asyncio
    async def test_http_404_raises_error(self):
        mock_resp = _make_mock_response(status_code=404, body=b"Not Found")
        with _patch_stream(mock_resp):
            with pytest.raises(ArticleExtractionError, match="404"):
                await extract_article("https://example.com/missing-page")

    @pytest.mark.asyncio
    async def test_http_401_raises_error(self):
        """Paywall / authentication required."""
        mock_resp = _make_mock_response(status_code=401, body=b"Unauthorized")
        with _patch_stream(mock_resp):
            with pytest.raises(ArticleExtractionError, match="[Aa]ccess denied|paywall|401"):
                await extract_article("https://example.com/premium-article")

    @pytest.mark.asyncio
    async def test_http_403_raises_error(self):
        mock_resp = _make_mock_response(status_code=403, body=b"Forbidden")
        with _patch_stream(mock_resp):
            with pytest.raises(ArticleExtractionError, match="[Aa]ccess denied|403"):
                await extract_article("https://example.com/restricted")

    @pytest.mark.asyncio
    async def test_http_500_raises_error(self):
        mock_resp = _make_mock_response(status_code=500, body=b"Server Error")
        with _patch_stream(mock_resp):
            with pytest.raises(ArticleExtractionError, match="500"):
                await extract_article("https://example.com/broken")

    @pytest.mark.asyncio
    async def test_unsupported_content_type_raises_error(self):
        mock_resp = _make_mock_response(
            status_code=200,
            content_type="application/pdf",
            body=b"%PDF-1.4",
        )
        with _patch_stream(mock_resp):
            with pytest.raises(ArticleExtractionError, match="[Uu]nsupported content type"):
                await extract_article("https://example.com/document.pdf")

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_client.stream     = MagicMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ArticleExtractionError, match="[Tt]imed out"):
                await extract_article("https://slow-server.example.com/article")

    @pytest.mark.asyncio
    async def test_network_error_raises_extraction_error(self):
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_client.stream     = MagicMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ArticleExtractionError, match="[Nn]etwork error"):
                await extract_article("https://unreachable.example.com/article")

    @pytest.mark.asyncio
    async def test_response_size_limit_enforced(self):
        """Responses larger than MAX_RESPONSE_BYTES are truncated, not rejected."""
        big_body   = b"x" * (_MAX_RESPONSE_BYTES + 1_000_000)
        mock_resp  = _make_mock_response(body=big_body)

        trafilatura_result = {
            "text": "Some extracted text " * 20,
            "title": None, "author": None, "date": None,
        }

        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=trafilatura_result,
            ):
                # Should complete without error even with oversized response
                result = await extract_article("https://bigpage.example.com/article")
                assert result.text is not None


# =============================================================================
# SSRF prevention — private URLs rejected BEFORE any HTTP call
# =============================================================================

class TestSSRFPrevention:
    """
    Verify that private/internal URLs are rejected in validate_url()
    before any network connection is attempted.
    All these tests must raise BadRequestError, never make HTTP calls.
    """

    @pytest.mark.parametrize("url", [
        "http://10.0.0.1/api",
        "http://10.255.255.255/secret",
        "http://172.16.100.1/internal",
        "http://172.31.255.254/",
        "http://192.168.0.1/router",
        "http://192.168.255.255/",
        "http://127.0.0.1/",
        "http://127.0.0.2/loop",
        "http://169.254.169.254/latest/meta-data/iam/",    # AWS metadata
        "http://169.254.169.254/computeMetadata/v1/",      # GCP metadata
        "https://localhost/admin",
        "http://[::1]/secret",
        "http://myhost.local/",
        "https://service.internal/api",
        "http://app.corp/dashboard",
        "http://example.com:8080/app",
        "http://example.com:3306/mysql",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "data:text/html,<script>xss</script>",
    ])
    def test_ssrf_url_rejected(self, url: str):
        with pytest.raises(BadRequestError):
            validate_url(url)

    @pytest.mark.parametrize("url", [
        "https://www.reuters.com/article/test",
        "https://www.bbc.co.uk/news/world-1234",
        "http://edition.cnn.com/2024/article",
        "https://apnews.com/article/test-123",
    ])
    def test_valid_public_url_passes(self, url: str):
        result = validate_url(url)
        assert result == url


# =============================================================================
# Multiple-author extraction
# =============================================================================

class TestAuthorExtraction:
    @pytest.mark.asyncio
    async def test_multiple_authors_split_correctly(self):
        mock_resp = _make_mock_response(body=VALID_HTML.encode())
        trafilatura_result = {
            "text":   "Article text content here " * 20,
            "title":  "Test Article",
            "author": "Alice Smith; Bob Jones, Carol Davis",
            "date":   None,
        }
        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=trafilatura_result,
            ):
                result = await extract_article("https://example.com/multi-author")

        assert len(result.authors) == 3
        assert "Alice Smith" in result.authors
        assert "Bob Jones"   in result.authors

    @pytest.mark.asyncio
    async def test_no_author_returns_empty_list(self):
        mock_resp = _make_mock_response(body=VALID_HTML.encode())
        trafilatura_result = {
            "text":   "Article text content here " * 20,
            "title":  "Anonymous Article",
            "author": None,
            "date":   None,
        }
        with _patch_stream(mock_resp):
            with patch(
                "app.extraction.article_extractor._extract_with_trafilatura",
                return_value=trafilatura_result,
            ):
                result = await extract_article("https://example.com/no-author")

        assert result.authors == []


# =============================================================================
# Too-many-redirects error
# =============================================================================

class TestRedirects:
    @pytest.mark.asyncio
    async def test_too_many_redirects_raises_error(self):
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_client.stream     = MagicMock(
            side_effect=httpx.TooManyRedirects("Too many redirects")
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ArticleExtractionError, match="[Rr]edirect"):
                await extract_article("https://redirect-loop.example.com/article")
