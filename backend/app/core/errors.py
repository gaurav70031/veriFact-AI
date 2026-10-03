"""
Application exception hierarchy and FastAPI exception handlers.

All HTTP errors are raised as AppError subclasses.
Global handlers convert them to structured JSON responses.

Response body shape:
{
    "error":   "SHORT_CODE",
    "message": "Human-readable description.",
    "detail":  { ... }   // optional, present only when debug=True or relevant
}
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


# ── Base exception ────────────────────────────────────────────────────────────

class AppError(Exception):
    """Base class for all application-level errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code:  str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.detail  = detail


# ── HTTP 400 ──────────────────────────────────────────────────────────────────

class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code  = "VALIDATION_ERROR"


class BadRequestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code  = "BAD_REQUEST"


# ── HTTP 404 ──────────────────────────────────────────────────────────────────

class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code  = "NOT_FOUND"


# ── HTTP 503 ──────────────────────────────────────────────────────────────────

class ModelUnavailableError(AppError):
    """Raised when a trained model artefact has not been found on disk."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code  = "MODEL_UNAVAILABLE"


class ExternalServiceError(AppError):
    """Raised when a third-party API (NewsAPI, SerpAPI, etc.) is unreachable."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code  = "EXTERNAL_SERVICE_ERROR"


# ── HTTP 422 ─────────────────────────────────────────────────────────────────

class ArticleExtractionError(AppError):
    """Raised when URL article extraction fails."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code  = "ARTICLE_EXTRACTION_FAILED"


# ── Exception handlers (register with FastAPI app) ────────────────────────────

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    body: dict = {"error": exc.error_code, "message": exc.message}
    if exc.detail is not None:
        body["detail"] = exc.detail
    return JSONResponse(status_code=exc.status_code, content=body)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error":   "INTERNAL_ERROR",
            "message": "An unexpected error occurred.",
        },
    )
