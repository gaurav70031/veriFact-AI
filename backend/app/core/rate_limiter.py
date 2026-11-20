"""
Rate limiting middleware.

Uses an in-memory sliding window rate limiter.
For production with multiple workers, replace with Redis-backed limiter
(e.g. slowapi or fastapi-limiter).

Configuration (from env vars):
  RATE_LIMIT_ENABLED    : "true" | "false" (default: "true" in production, "false" in dev)
  RATE_LIMIT_PER_MINUTE : max requests per minute per IP (default: 60)
  RATE_LIMIT_BURST      : burst allowance (default: 10)

Endpoints exempt from rate limiting:
  /api/v1/health
  /docs, /redoc, /openapi.json
  / (root)
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

_ENABLED: Optional[bool] = None
_PER_MINUTE: int = 60
_BURST: int = 10

# Sliding window: store timestamps of requests per IP
# {ip: deque([timestamp, ...])}
_windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))


def _get_config() -> tuple[bool, int, int]:
    global _ENABLED, _PER_MINUTE, _BURST
    if _ENABLED is None:
        env = os.getenv("RATE_LIMIT_ENABLED", "").lower()
        app_env = os.getenv("APP_ENV", "development").lower()
        _ENABLED = env == "true" if env else app_env == "production"
        _PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        _BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))
    return _ENABLED, _PER_MINUTE, _BURST


# ── Exempt paths ──────────────────────────────────────────────────────────────

_EXEMPT_PREFIXES = (
    "/api/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
)


def _is_exempt(path: str) -> bool:
    for prefix in _EXEMPT_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _get_client_ip(request: Request) -> str:
    """Get client IP, respecting X-Forwarded-For behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Middleware ────────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Allows up to RATE_LIMIT_PER_MINUTE requests per minute per IP,
    with a burst of RATE_LIMIT_BURST requests in the first second.

    When the limit is exceeded, returns HTTP 429 with a Retry-After header.
    """

    async def dispatch(self, request: Request, call_next):
        enabled, per_minute, burst = _get_config()

        if not enabled:
            return await call_next(request)

        path = request.url.path

        # Skip exempt paths
        if _is_exempt(path):
            return await call_next(request)

        # Skip OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        ip = _get_client_ip(request)
        now = time.time()

        # Clean old entries (older than 60 seconds)
        window = _windows[ip]
        while window and window[0] < now - 60:
            window.popleft()

        # Check limit
        if len(window) >= per_minute:
            retry_after = int(60 - (now - window[0])) + 1
            logger.warning(
                "Rate limit exceeded for %s on %s (%d requests in 60s)",
                ip, path, len(window),
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": (
                        f"Too many requests. Limit: {per_minute} per minute. "
                        f"Please retry after {retry_after} seconds."
                    ),
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record this request
        window.append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, per_minute - len(window))
        response.headers["X-RateLimit-Limit"] = str(per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


def reset_rate_limiter() -> None:
    """Clear all rate limit windows (for testing)."""
    _windows.clear()
