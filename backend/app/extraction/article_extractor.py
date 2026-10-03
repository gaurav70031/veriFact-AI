"""
Article text extractor.

Given a public URL, fetch the page and extract clean article text.

Strategy (two-library cascade)
-------------------------------
1. Try trafilatura first — it is faster, handles most modern news sites,
   and returns clean boilerplate-free text.
2. Fall back to newspaper3k if trafilatura returns nothing useful.
3. If both fail, raise ArticleExtractionError so the API returns HTTP 422.

Neither library is required at import time — both are optional dependencies.
If only one is installed the cascade still works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.core.errors import ArticleExtractionError, BadRequestError
from app.core.security import validate_url

logger = logging.getLogger(__name__)

# ── HTTP client settings ──────────────────────────────────────────────────────
_TIMEOUT      = httpx.Timeout(15.0, connect=5.0)
_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB
_USER_AGENT   = (
    "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; "
    "+https://github.com/example/fake-news-detection)"
)


@dataclass
class ArticleData:
    url:     str
    title:   Optional[str]
    text:    str
    authors: list[str]
    publish_date: Optional[str]
    source_domain: Optional[str]


# ── Extractor implementations ─────────────────────────────────────────────────

def _extract_with_trafilatura(html: str, url: str) -> Optional[str]:
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        return text if text and len(text.strip()) > 100 else None
    except ImportError:
        logger.debug("trafilatura not installed — skipping.")
        return None
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)
        return None


def _extract_with_newspaper(url: str, html: str) -> Optional[tuple[str, Optional[str]]]:
    """Returns (text, title) or None."""
    try:
        from newspaper import Article as NewsArticle
        article = NewsArticle(url)
        article.download(input_html=html)
        article.parse()
        text  = article.text.strip()
        title = article.title.strip() if article.title else None
        return (text, title) if len(text) > 100 else None
    except ImportError:
        logger.debug("newspaper3k not installed — skipping.")
        return None
    except Exception as exc:
        logger.debug("newspaper3k extraction failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def extract_article(url: str) -> ArticleData:
    """
    Fetch a public URL and extract clean article text.

    Parameters
    ----------
    url : Public HTTPS URL of a news article.

    Returns
    -------
    ArticleData with title, text, authors, publish_date, source_domain.

    Raises
    ------
    BadRequestError           : URL fails security validation.
    ArticleExtractionError    : Page could not be fetched or text extracted.
    """
    validated_url = validate_url(url)

    # ── Fetch HTML ────────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(validated_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ArticleExtractionError(
            f"HTTP {exc.response.status_code} when fetching {validated_url}.",
            detail={"status_code": exc.response.status_code},
        )
    except httpx.TimeoutException:
        raise ArticleExtractionError(
            f"Request timed out while fetching {validated_url}.",
        )
    except httpx.RequestError as exc:
        raise ArticleExtractionError(
            f"Network error while fetching {validated_url}: {exc}",
        )

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ArticleExtractionError(
            f"URL returned unsupported content type: {content_type}. "
            "Only HTML pages are supported."
        )

    html = response.text

    # ── Extract text ──────────────────────────────────────────────────────────
    text:  Optional[str] = None
    title: Optional[str] = None

    # Strategy 1: trafilatura
    trafilatura_text = _extract_with_trafilatura(html, validated_url)
    if trafilatura_text:
        text  = trafilatura_text
        logger.debug("Article extracted via trafilatura (%d chars).", len(text))

    # Strategy 2: newspaper3k fallback
    if not text:
        newspaper_result = _extract_with_newspaper(validated_url, html)
        if newspaper_result:
            text, title = newspaper_result
            logger.debug("Article extracted via newspaper3k (%d chars).", len(text))

    if not text:
        raise ArticleExtractionError(
            "Could not extract article text from the provided URL. "
            "The page may be behind a paywall, require JavaScript, "
            "or contain no extractable article content."
        )

    # ── Parse domain ──────────────────────────────────────────────────────────
    from urllib.parse import urlparse
    parsed = urlparse(validated_url)
    domain = parsed.netloc.lstrip("www.")

    return ArticleData(
        url=validated_url,
        title=title,
        text=text,
        authors=[],
        publish_date=None,
        source_domain=domain,
    )


def extract_article_sync(url: str) -> ArticleData:
    """
    Synchronous wrapper — use only in non-async contexts or tests.
    In FastAPI route handlers, always use the async version.
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(extract_article(url))
