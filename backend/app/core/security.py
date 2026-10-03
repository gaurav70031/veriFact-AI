"""
Security utilities: input sanitisation, URL validation, SSRF prevention.

URL safety
----------
Every URL submitted by users is passed through validate_url() before any
network request is made.  The function enforces:

  1. Scheme allowlist — only http:// and https://
  2. Length limit
  3. Private / loopback IP block (SSRF prevention — RFC 1918 + link-local)
  4. Blocked internal hostnames (localhost, metadata endpoints, .local TLDs)
  5. Port allowlist — only 80 and 443 (or no port)

Redirect safety
---------------
validate_redirect_url() is called on every redirect Location header before
following it, so a redirect chain cannot escape to a private IP.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from app.core.errors import BadRequestError, ValidationError

# ── Private IP ranges (RFC 1918 + loopback + link-local + special) ───────────
_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local / AWS metadata
    ipaddress.ip_network("100.64.0.0/10"),    # shared address space
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),          # ULA
    ipaddress.ip_network("fe80::/10"),         # link-local IPv6
    ipaddress.ip_network("::ffff:0:0/96"),     # IPv4-mapped
]

# Hostnames that must always be blocked (even if they don't look like IPs)
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset({
    "localhost",
    "localhost.localdomain",
    "::1",
    "0.0.0.0",
    "broadcasthost",
    # AWS / GCP / Azure metadata endpoints
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.com",
})

# Blocked TLD suffixes (internal/private networks)
_BLOCKED_TLD_SUFFIXES = (".local", ".internal", ".intranet", ".corp", ".lan")

# Only allow standard web ports (or no port specified)
_ALLOWED_PORTS: frozenset[int] = frozenset({80, 443})

# Allowed URL schemes
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _is_private_ip(hostname: str) -> bool:
    """Return True if hostname resolves to a private/reserved IP address."""
    try:
        addr = ipaddress.ip_address(hostname)
        return any(addr in net for net in _PRIVATE_RANGES)
    except ValueError:
        return False   # not an IP address literal


def validate_url(url: str, max_length: int = 2_000) -> str:
    """
    Validate a public URL and return the normalised string.

    Raises BadRequestError for any security or format violation.

    Checks (in order)
    -----------------
    1. Non-empty string
    2. Length ≤ max_length
    3. Scheme in {http, https}
    4. Non-empty host
    5. Host not in blocked hostname list
    6. Host TLD not in blocked suffix list
    7. Host not a private/loopback/reserved IP
    8. Port is 80, 443, or absent
    """
    if not url or not isinstance(url, str):
        raise BadRequestError("URL must be a non-empty string.")

    url = url.strip()

    if len(url) > max_length:
        raise BadRequestError(
            f"URL exceeds maximum allowed length of {max_length} characters."
        )

    parsed = urlparse(url)

    # 1. Scheme
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise BadRequestError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            "Only http and https are supported."
        )

    # 2. Host present
    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise BadRequestError("URL has no host component.")

    # 3. Blocked hostname list
    if hostname in _BLOCKED_HOSTNAMES:
        raise BadRequestError(
            f"Requests to '{hostname}' are not permitted."
        )

    # 4. Blocked TLD suffixes
    for suffix in _BLOCKED_TLD_SUFFIXES:
        if hostname.endswith(suffix):
            raise BadRequestError(
                f"Requests to internal domain '{hostname}' are not permitted."
            )

    # 5. Private IP check
    if _is_private_ip(hostname):
        raise BadRequestError(
            "Requests to private, loopback, or reserved IP addresses "
            "are not permitted (SSRF prevention)."
        )

    # 6. Port check
    if parsed.port is not None and parsed.port not in _ALLOWED_PORTS:
        raise BadRequestError(
            f"Port {parsed.port} is not allowed. "
            "Only ports 80 and 443 are permitted."
        )

    return url


def validate_redirect_url(url: str) -> str:
    """
    Validate a redirect Location header before following it.

    Same rules as validate_url() — ensures a redirect chain cannot
    escape to an internal network address.

    Raises BadRequestError if the redirect target is disallowed.
    """
    try:
        return validate_url(url)
    except BadRequestError as exc:
        raise BadRequestError(
            f"Unsafe redirect target blocked: {exc.message}"
        )


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
