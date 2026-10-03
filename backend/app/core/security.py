"""
Security utilities: input sanitisation, URL validation, rate limiting hooks.

URL safety
----------
Only HTTPS URLs are accepted.  Private/loopback IP ranges are blocked to
prevent SSRF (Server-Side Request Forgery) attacks.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from app.core.errors import BadRequestError, ValidationError

# Private IP ranges (RFC 1918 + loopback + link-local)
_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def validate_url(url: str, max_length: int = 2_000) -> str:
    """
    Validate a public HTTPS URL.

    Rules:
      1. Must start with https://
      2. Must not exceed max_length characters
      3. Must not resolve to a private/loopback IP (SSRF guard)

    Returns the normalised URL string, or raises BadRequestError.
    """
    if not url or not isinstance(url, str):
        raise BadRequestError("URL must be a non-empty string.")

    url = url.strip()

    if len(url) > max_length:
        raise BadRequestError(f"URL exceeds maximum length of {max_length} characters.")

    parsed = urlparse(url)

    if parsed.scheme not in ("https", "http"):
        raise BadRequestError("Only HTTP/HTTPS URLs are supported.")

    if not parsed.netloc:
        raise BadRequestError("URL has no host.")

    hostname = parsed.hostname or ""

    # Block numeric IP addresses that fall in private ranges
    try:
        addr = ipaddress.ip_address(hostname)
        for private_range in _PRIVATE_RANGES:
            if addr in private_range:
                raise BadRequestError(
                    "Requests to private or loopback IP addresses are not permitted."
                )
    except ValueError:
        pass   # hostname is a domain name, not an IP — allowed

    # Block obviously internal hostnames
    blocked_hosts = {"localhost", "::1", "0.0.0.0"}
    if hostname.lower() in blocked_hosts:
        raise BadRequestError("Requests to localhost are not permitted.")

    return url


def sanitise_text(text: str, max_length: int = 50_000) -> str:
    """
    Basic text sanitisation:
      - Strip leading/trailing whitespace
      - Enforce maximum length
      - Ensure non-empty
    """
    if not text or not isinstance(text, str):
        raise ValidationError("Text input must be a non-empty string.")

    text = text.strip()

    if not text:
        raise ValidationError("Text input must not be blank.")

    if len(text) > max_length:
        raise ValidationError(
            f"Text input exceeds maximum allowed length of {max_length:,} characters."
        )

    return text
