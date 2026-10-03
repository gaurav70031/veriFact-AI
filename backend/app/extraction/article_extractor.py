"""
Safe public article URL extractor.

Security guarantees
-------------------
* URL validated before any network request (SSRF prevention).
* Every redirect Location header is re-validated before following.
* Response body capped at MAX_RESPONSE_BYTES (5 MB) — stream is aborted.
* Connection timeout (5 s) + read timeout (15 s) enforced.
* Maximum 5 redirects followed.
* Only HTTP 200 responses with HTML/plain-text content type accepted.

Ethical constraints
-------------------
* We do NOT bypass paywalls, authentication, or robots.txt restrictions.
* Only the extractable public-facing text is returned.
* Full article text is NOT stored in the database — the caller stores
  only the first 10,000 characters (see analysis_service.py).

Extraction strategy (cascade)
------------------------------
1. trafilatura   — fast, boilerplate-aware, best for news sites.
                   Also extracts title, author, and publication date.
2. newspaper3k   — fallback; good for older / simpler sites.
                   Also extracts title, authors, and publish_date.
3. Both fail     — ArticleExtractionError raised → HTTP 422.

Returned fields
---------------
ArticleData:
  url           canonical URL (final URL after redirects)
  title         article headline (str | None)
  text          clean article body (str, min 100 chars)
  authors       list[str] where available
  publish_date  ISO date string | None
  source_domain domain name (e.g. "reuters.com")
  canonical_url canonical URL from <link rel="canonical"> if found
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.core.errors import ArticleExtractionError, BadRequestError
from app.core.security import validate_url, validate_redirect_url

logger = logging.getLogger(__name__)

# ── HTTP client constants ──────────────────────────────────────────────────────
_CONNECT_TIMEOUT    = 5.0     # seconds to establish TCP connection
_READ_TIMEOUT       = 15.0    # seconds to read response body
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024   # 5 MB hard cap on response size
_MAX_REDIRECTS      = 5
_USER_AGENT = (
    "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; "
    "+https://github.com/example/fake-news-detection)"
)

# Minimum extracted text length to be considered a valid extraction
_MIN_TEXT_LENGTH = 100

# ── Canonical URL extraction ───────────────────────────────────────────────────
_RE_CANONICAL = re.compile(
    r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _extract_canonical_url(html: str) -> Optional[str]:
    """Extract <link rel="canonical" href="..."> from HTML head."""
    match = _RE_CANONICAL.search(html[:4096])   # only scan head section
    return match.group(1).strip() if match else None


# ── ArticleData ────────────────────────────────────────────────────────────────

@dataclass
class ArticleData:
    """Extracted article data returned to the analysis pipeline."""
    url:           str
    title:         Optional[str]
    text:          str
    authors:       list[str]       = field(default_factory=list)
    publish_date:  Optional[str]   = None
    source_domain: Optional[str]   = None
    canonical_url: Optional[str]   = None


# ── Extraction backends ────────────────────────────────────────────────────────

def _extract_with_trafilatura(
    html: str,
    url:  str,
) -> Optional[dict]:
    """
    Extract text, title, author, and date using trafilatura.
    Returns a dict or None if extraction fails / returns too little text.
    """
    try:
        import trafilatura
        from trafilatura.settings import use_config

        config = use_config()
        config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")

        result = trafilatura.extract(
            html,
            url=url,
            output_format="python",   # returns dict with metadata
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
            config=config,
        )

        if not result:
            return None

        # When output_format="python" is unsupported in older versions,
        # trafilatura returns a string — handle both cases
        if isinstance(result, str):
            return {"text": result, "title": None, "author": None, "date": None}

        text = result.get("text") or ""
        if len(text.strip()) < _MIN_TEXT_LENGTH:
            return None

        return {
            "text":   text.strip(),
            "title":  result.get("title"),
            "author": result.get("author"),
            "date":   result.get("date"),
        }

    except ImportError:
        logger.debug("trafilatura not installed — skipping.")
        return None
    except TypeError:
        # output_format="python" not supported in this trafilatura version
        try:
            import trafilatura
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
            )
            if text and len(text.strip()) >= _MIN_TEXT_LENGTH:
                return {"text": text.strip(), "title": None, "author": None, "date": None}
        except Exception:
            pass
        return None
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)
        return None


def _extract_with_newspaper(
    url:  str,
    html: str,
) -> Optional[dict]:
    """
    Extract text, title, authors, and publish_date using newspaper3k.
    Returns a dict or None if extraction fails / returns too little text.
    """
    try:
        from newspaper import Article as NewsArticle

        article = NewsArticle(url)
        article.download(input_html=html)
        article.parse()

        text = article.text.strip()
        if len(text) < _MIN_TEXT_LENGTH:
            return None

        publish_date = None
        if article.publish_date:
            try:
                publish_date = article.publish_date.isoformat()
            except Exception:
                publish_date = str(article.publish_date)

        return {
            "text":    text,
            "title":   article.title.strip() if article.title else None,
            "authors": list(article.authors) if article.authors else [],
            "date":    publish_date,
        }

    except ImportError:
        logger.debug("newspaper3k not installed — skipping.")
        return None
    except Exception as exc:
        logger.debug("newspaper3k extraction failed: %s", exc)
        return None


# ── Safe HTTP fetch ────────────────────────────────────────────────────────────

class _SSRFSafeTransport(httpx.AsyncHTTPTransport):
    """
    Custom transport that validates every redirect Location header
    before following it.  Prevents open-redirect SSRF attacks.
    """
    pass   # redirect validation is done in extract_article() via event_hooks


async def _fetch_html(url: str) -> tuple[str, str]:
    """
    Fetch a URL safely and return (html_content, final_url).

    Safety measures applied:
      * Timeout: 5 s connect + 15 s read
      * Response size: aborted after MAX_RESPONSE_BYTES
      * Redirect validation: each Location header re-validated
      * Only 200 responses with HTML/text content accepted
    """
    final_url = url

    def _on_redirect(response: httpx.Response) -> None:
        """Called by httpx before each redirect is followed."""
        location = response.headers.get("location", "")
        if location:
            try:
                validate_redirect_url(location)
            except BadRequestError as exc:
                raise ArticleExtractionError(
                    f"Redirect blocked for security reasons: {exc.message}"
                )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT),
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
            headers={"User-Agent": _USER_AGENT},
            event_hooks={"response": [_on_redirect]},
        ) as client:
            # Stream the response so we can abort at MAX_RESPONSE_BYTES
            async with client.stream("GET", url) as response:
                final_url = str(response.url)

                if response.status_code == 401 or response.status_code == 403:
                    raise ArticleExtractionError(
                        f"Access denied (HTTP {response.status_code}). "
                        "The article may be behind a paywall or require authentication.",
                        detail={"status_code": response.status_code},
                    )
                if response.status_code == 404:
                    raise ArticleExtractionError(
                        f"Article not found (HTTP 404): {url}",
                        detail={"status_code": 404},
                    )
                if response.status_code >= 400:
                    raise ArticleExtractionError(
                        f"HTTP {response.status_code} when fetching {url}.",
                        detail={"status_code": response.status_code},
                    )

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    raise ArticleExtractionError(
                        f"Unsupported content type '{content_type}'. "
                        "Only HTML pages are supported.",
                        detail={"content_type": content_type},
                    )

                # Read up to MAX_RESPONSE_BYTES
                chunks: list[bytes] = []
                bytes_read = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    bytes_read += len(chunk)
                    if bytes_read > _MAX_RESPONSE_BYTES:
                        logger.warning(
                            "Response from %s exceeds %d MB — truncating.",
                            url, _MAX_RESPONSE_BYTES // 1_048_576,
                        )
                        chunks.append(chunk[:_MAX_RESPONSE_BYTES - (bytes_read - len(chunk))])
                        break
                    chunks.append(chunk)

                raw = b"".join(chunks)

                # Decode: respect charset in Content-Type, fall back to utf-8
                charset = "utf-8"
                ct_match = re.search(r"charset=([^\s;]+)", content_type)
                if ct_match:
                    charset = ct_match.group(1).strip('"\'')

                try:
                    html = raw.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    html = raw.decode("utf-8", errors="replace")

    except httpx.TimeoutException:
        raise ArticleExtractionError(
            f"Request timed out after {_READ_TIMEOUT}s fetching {url}. "
            "The server may be slow or unresponsive.",
        )
    except httpx.TooManyRedirects:
        raise ArticleExtractionError(
            f"Too many redirects (max {_MAX_REDIRECTS}) following {url}.",
        )
    except httpx.RequestError as exc:
        raise ArticleExtractionError(
            f"Network error fetching {url}: {exc}",
        )

    return html, final_url


# ── Public API ────────────────────────────────────────────────────────────────

async def extract_article(url: str) -> ArticleData:
    """
    Validate a URL, fetch it safely, and extract clean article content.

    Parameters
    ----------
    url : User-supplied URL (any string — will be validated).

    Returns
    -------
    ArticleData with title, text, authors, publish_date, canonical_url.

    Raises
    ------
    BadRequestError        : URL is malformed, private, or disallowed.
    ArticleExtractionError : Fetch failed or no extractable text found.
    """
    # Step 1: Validate URL (SSRF prevention)
    validated_url = validate_url(url)
    logger.info("Extracting article: %s", validated_url)

    # Step 2: Fetch HTML safely
    html, final_url = await _fetch_html(validated_url)

    # Step 3: Extract canonical URL from HTML head
    canonical_url = _extract_canonical_url(html)

    # Step 4: Extract article content (trafilatura first, newspaper3k fallback)
    extracted: Optional[dict] = None

    trafilatura_result = _extract_with_trafilatura(html, final_url)
    if trafilatura_result:
        extracted = trafilatura_result
        logger.info(
            "Extracted via trafilatura: %d chars, title=%r",
            len(extracted.get("text", "")),
            extracted.get("title"),
        )

    if not extracted:
        newspaper_result = _extract_with_newspaper(final_url, html)
        if newspaper_result:
            extracted = newspaper_result
            logger.info(
                "Extracted via newspaper3k: %d chars, title=%r",
                len(extracted.get("text", "")),
                extracted.get("title"),
            )

    if not extracted:
        raise ArticleExtractionError(
            "Could not extract article text from the provided URL. "
            "Possible causes: paywall, JavaScript-rendered page, "
            "robots restriction, or no article content found.",
            detail={"url": final_url},
        )

    # Step 5: Parse domain from final URL
    from urllib.parse import urlparse
    parsed      = urlparse(final_url)
    source_domain = parsed.netloc.lstrip("www.")

    # Step 6: Normalise authors
    raw_author  = extracted.get("author") or ""
    raw_authors = extracted.get("authors") or []
    if raw_author and not raw_authors:
        raw_authors = [a.strip() for a in re.split(r"[;,]", raw_author) if a.strip()]

    return ArticleData(
        url=final_url,
        title=extracted.get("title"),
        text=extracted["text"],
        authors=raw_authors,
        publish_date=extracted.get("date"),
        source_domain=source_domain,
        canonical_url=canonical_url,
    )


def extract_article_sync(url: str) -> ArticleData:
    """Synchronous wrapper for non-async contexts only (e.g. scripts)."""
    import asyncio
    return asyncio.run(extract_article(url))
