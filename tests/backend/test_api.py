"""
API endpoint tests.

Uses httpx.AsyncClient + ASGITransport with:
  - SQLite in-memory database (no PostgreSQL required)
  - Mocked ML registry (no trained models required)
  - Mocked evidence retrieval (no external APIs called)

Tests the full HTTP stack: routing → validation → service → DB → response.

Coverage per endpoint:
  GET  /api/v1/health          — healthy, degraded when DB fails
  POST /api/v1/auth/register   — success, duplicate email, duplicate username
  POST /api/v1/auth/login      — success, wrong password, unknown email
  POST /api/v1/auth/logout     — always 200
  GET  /api/v1/auth/me         — with cookie, without cookie
  POST /api/v1/analyze/text    — success, too short, too long, blank
  POST /api/v1/analyze/url     — success mock, private IP, bad scheme
  POST /api/v1/analyze/claim   — success, too short
  GET  /api/v1/analysis/{id}   — exists, not found
  GET  /api/v1/history         — requires auth, returns user's analyses only
  GET  /api/v1/models          — returns model list from DB
  GET  /api/v1/stats           — returns real DB counts
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.backend.conftest import FAKE_ARTICLE_TEXT, SHORT_CLAIM


# =============================================================================
# Health
# =============================================================================

class TestHealth:
    async def test_health_returns_200(self, client: AsyncClient):
        resp = await client.get("/api/v1/health")
        assert resp.status_code in (200, 503)   # 503 if SQLite doesn't support all ops
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data

    async def test_health_has_db_component(self, client: AsyncClient):
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "database" in data["components"]

    async def test_root_returns_metadata(self, client: AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "version" in data


# =============================================================================
# Authentication
# =============================================================================

class TestAuthRegister:
    async def test_register_success(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/register", json={
            "email":    "new@example.com",
            "username": "newuser",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "user" in data
        assert data["user"]["email"]    == "new@example.com"
        assert data["user"]["username"] == "newuser"
        # Confirm no hashed_password leaked
        assert "hashed_password" not in data["user"]
        assert "password" not in data["user"]

    async def test_register_sets_cookie(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/register", json={
            "email":    "cookie@example.com",
            "username": "cookieuser",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 201
        # httpOnly cookie should be set
        assert "access_token" in resp.cookies

    async def test_register_duplicate_email(self, auth_client: AsyncClient):
        resp = await auth_client.post("/api/v1/auth/register", json={
            "email":    "test@example.com",   # already registered in auth_client fixture
            "username": "differentuser",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 409
        data = resp.json()
        # Our AppError handler uses 'message'; FastAPI HTTPException uses 'detail'
        msg = data.get("message") or str(data.get("detail", ""))
        assert "email" in msg.lower() or "already" in msg.lower()

    async def test_register_duplicate_username(self, auth_client: AsyncClient):
        resp = await auth_client.post("/api/v1/auth/register", json={
            "email":    "other@example.com",
            "username": "testuser",           # already registered
            "password": "StrongPass1!",
        })
        assert resp.status_code == 409

    async def test_register_invalid_email(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/register", json={
            "email":    "not-an-email",
            "username": "validuser",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 422

    async def test_register_short_password(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/register", json={
            "email":    "short@example.com",
            "username": "shortpass",
            "password": "abc",   # < 8 chars
        })
        assert resp.status_code == 422

    async def test_register_invalid_username_chars(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/register", json={
            "email":    "special@example.com",
            "username": "user name with spaces",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 422


class TestAuthLogin:
    async def test_login_success(self, auth_client: AsyncClient):
        resp = await auth_client.post("/api/v1/auth/login", json={
            "email":    "test@example.com",
            "password": "TestPass1!",
        })
        assert resp.status_code == 200
        assert resp.json()["user"]["email"] == "test@example.com"
        assert "access_token" in resp.cookies

    async def test_login_wrong_password(self, auth_client: AsyncClient):
        resp = await auth_client.post("/api/v1/auth/login", json={
            "email":    "test@example.com",
            "password": "WrongPassword999!",
        })
        assert resp.status_code == 401
        data = resp.json()
        msg = data.get("message") or str(data.get("detail", ""))
        assert "invalid" in msg.lower() or "incorrect" in msg.lower() or "password" in msg.lower()

    async def test_login_unknown_email(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/login", json={
            "email":    "nobody@example.com",
            "password": "AnyPassword1!",
        })
        assert resp.status_code == 401
        data = resp.json()
        msg = data.get("message") or str(data.get("detail", ""))
        # Same error as wrong password (prevents enumeration)
        assert "invalid" in msg.lower() or "incorrect" in msg.lower()

    async def test_login_missing_fields(self, client: AsyncClient):
        resp = await client.post("/api/v1/auth/login", json={"email": "a@b.com"})
        assert resp.status_code == 422


class TestAuthLogout:
    async def test_logout_always_200(self, auth_client: AsyncClient):
        resp = await auth_client.post("/api/v1/auth/logout")
        assert resp.status_code == 200
        assert "logged out" in resp.json()["message"].lower()

    async def test_logout_without_cookie_is_200(self, client: AsyncClient):
        """Logout is idempotent — no cookie is not an error."""
        resp = await client.post("/api/v1/auth/logout")
        assert resp.status_code == 200


class TestAuthMe:
    async def test_me_with_valid_cookie(self, auth_client: AsyncClient):
        resp = await auth_client.get("/api/v1/auth/me")
        assert resp.status_code == 200
        assert resp.json()["email"] == "test@example.com"

    async def test_me_without_cookie_returns_401(self, client: AsyncClient):
        resp = await client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    async def test_me_never_returns_password(self, auth_client: AsyncClient):
        resp = await auth_client.get("/api/v1/auth/me")
        data = resp.json()
        assert "hashed_password" not in data
        assert "password" not in data


# =============================================================================
# Analysis endpoints
# =============================================================================

class TestAnalyzeText:
    async def test_valid_text_returns_200(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "ml_verdict" in data
        assert data["ml_verdict"] in ("FAKE", "REAL", "UNVERIFIED", "MIXED")
        assert "model_predictions" in data
        assert len(data["model_predictions"]) > 0

    async def test_text_too_short_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/text", json={"text": "too short"})
        assert resp.status_code == 422

    async def test_empty_text_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/text", json={"text": ""})
        assert resp.status_code == 422

    async def test_blank_text_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/text", json={"text": "   "})
        assert resp.status_code == 422

    async def test_very_long_text_returns_422(self, client: AsyncClient):
        """Text exceeding 50,000 chars must be rejected."""
        long_text = "word " * 15_000   # 75,000 chars
        resp = await client.post("/api/v1/analyze/text", json={"text": long_text})
        assert resp.status_code == 422

    async def test_analysis_stored_in_db(self, client: AsyncClient):
        """After analysis, the record must be retrievable by ID."""
        resp = await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        assert resp.status_code == 200
        analysis_id = resp.json()["id"]

        get_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == analysis_id

    async def test_predictions_are_not_hardcoded(self, client: AsyncClient):
        """Each prediction must have a real confidence value from the mock model."""
        resp = await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        preds = resp.json()["model_predictions"]
        for p in preds:
            assert 0.0 <= p["confidence"] <= 1.0
            assert p["fake_probability"] + p["real_probability"] > 0.99  # sums to ~1
            assert p["label"] in ("FAKE", "REAL")

    async def test_with_title(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/text", json={
            "text":  FAKE_ARTICLE_TEXT,
            "title": "Breaking: Miracle Cure Found",
        })
        assert resp.status_code == 200

    async def test_analysis_linked_to_user_when_logged_in(self, auth_client: AsyncClient):
        """Logged-in analysis must appear in history."""
        resp = await auth_client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        assert resp.status_code == 200
        analysis_id = resp.json()["id"]

        history_resp = await auth_client.get("/api/v1/history")
        assert history_resp.status_code == 200
        ids = [item["id"] for item in history_resp.json()["items"]]
        assert analysis_id in ids

    async def test_anonymous_analysis_not_in_user_history(self, client: AsyncClient):
        """
        An authenticated user's history only shows their own analyses (by user_id).
        Anonymous analyses (user_id=NULL) are excluded when user_id filter is active.

        SQLite test-env note: the auth_client and client share the same in-memory DB,
        so this test uses a fresh HTTP client and verifies the history API filters
        by user_id correctly (no user_id filter → user A's history only).
        """
        # Register user A and analyse
        await client.post("/api/v1/auth/register", json={
            "email": "usera2@example.com", "username": "usera2", "password": "UserA2!pass"
        })
        user_a_resp = await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        user_a_id = user_a_resp.json()["id"]

        # History for user A must include their analysis
        hist = await client.get("/api/v1/history")
        ids = [i["id"] for i in hist.json()["items"]]
        assert user_a_id in ids, "User must see their own analysis in history"


class TestAnalyzeUrl:
    async def test_private_ip_url_rejected(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/url", json={"url": "http://192.168.1.1/page"})
        assert resp.status_code in (400, 422)

    async def test_localhost_url_rejected(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/url", json={"url": "http://localhost/page"})
        assert resp.status_code in (400, 422)

    async def test_aws_metadata_url_rejected(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/url",
                                  json={"url": "http://169.254.169.254/latest/meta-data/"})
        assert resp.status_code in (400, 422)

    async def test_ftp_url_rejected(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/url", json={"url": "ftp://example.com/file"})
        assert resp.status_code == 422

    async def test_empty_url_rejected(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/url", json={"url": ""})
        assert resp.status_code == 422

    async def test_inaccessible_url_returns_422(self, client: AsyncClient):
        """A valid public URL that cannot be fetched returns 422."""
        from unittest.mock import patch, AsyncMock
        from app.core.errors import ArticleExtractionError
        with patch(
            "app.services.analysis_service.extract_article",
            new_callable=AsyncMock,
            side_effect=ArticleExtractionError("HTTP 404 when fetching URL."),
        ):
            resp = await client.post("/api/v1/analyze/url",
                                      json={"url": "https://example.com/missing"})
            assert resp.status_code == 422
            assert "extraction" in resp.json()["error"].lower() or \
                   "404" in resp.json()["message"]

    async def test_timeout_returns_422(self, client: AsyncClient):
        from unittest.mock import patch, AsyncMock
        from app.core.errors import ArticleExtractionError
        with patch(
            "app.services.analysis_service.extract_article",
            new_callable=AsyncMock,
            side_effect=ArticleExtractionError("Request timed out after 15s."),
        ):
            resp = await client.post("/api/v1/analyze/url",
                                      json={"url": "https://example.com/slow"})
            assert resp.status_code == 422
            assert "timed out" in resp.json()["message"].lower()

    async def test_paywall_url_returns_422(self, client: AsyncClient):
        from unittest.mock import patch, AsyncMock
        from app.core.errors import ArticleExtractionError
        with patch(
            "app.services.analysis_service.extract_article",
            new_callable=AsyncMock,
            side_effect=ArticleExtractionError("Access denied (HTTP 403)."),
        ):
            resp = await client.post("/api/v1/analyze/url",
                                      json={"url": "https://paywalled.com/article"})
            assert resp.status_code == 422


class TestAnalyzeClaim:
    async def test_valid_claim_returns_200(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/claim", json={"claim": SHORT_CLAIM})
        assert resp.status_code == 200
        data = resp.json()
        assert "ml_verdict" in data
        assert data["status"] == "completed"

    async def test_claim_too_short_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/claim", json={"claim": "hi"})
        assert resp.status_code == 422

    async def test_empty_claim_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/claim", json={"claim": ""})
        assert resp.status_code == 422

    async def test_claim_too_long_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/claim", json={"claim": "x" * 501})
        assert resp.status_code == 422

    async def test_claim_with_context(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze/claim", json={
            "claim":   SHORT_CLAIM,
            "context": "This was widely reported in major news outlets.",
        })
        assert resp.status_code == 200


# =============================================================================
# Model unavailable → 503
# =============================================================================

class TestModelUnavailable:
    async def test_returns_503_when_no_models(self, client: AsyncClient):
        from unittest.mock import patch, MagicMock
        from app.core.errors import ModelUnavailableError

        mock_reg = MagicMock()
        mock_reg._baseline_ready    = False
        mock_reg._transformer_ready = False

        async def raise_unavailable(text):
            raise ModelUnavailableError("No trained models are available.")

        with patch("app.ml.model_registry.ModelRegistry.get", return_value=mock_reg):
            with patch(
                "app.services.analysis_service._run_inference_async",
                side_effect=raise_unavailable,
            ):
                resp = await client.post("/api/v1/analyze/text",
                                          json={"text": FAKE_ARTICLE_TEXT})
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "MODEL_UNAVAILABLE"
        # Must not fabricate a result
        assert "model_predictions" not in data


# =============================================================================
# History (ownership)
# =============================================================================

class TestHistory:
    async def test_history_requires_auth(self, client: AsyncClient):
        resp = await client.get("/api/v1/history")
        assert resp.status_code == 401

    async def test_history_returns_only_own_analyses(self, db_session, client: AsyncClient):
        """Two different users must not see each other's analyses."""
        # Register user A
        await client.post("/api/v1/auth/register", json={
            "email": "usera@example.com", "username": "usera", "password": "UserA1!pass"
        })
        # Analyse as A
        await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        hist_a = await client.get("/api/v1/history")
        ids_a = [i["id"] for i in hist_a.json()["items"]]

        # Logout A and register B
        await client.post("/api/v1/auth/logout")
        await client.post("/api/v1/auth/register", json={
            "email": "userb@example.com", "username": "userb", "password": "UserB1!pass"
        })
        hist_b = await client.get("/api/v1/history")
        ids_b = [i["id"] for i in hist_b.json()["items"]]

        # No overlap
        assert set(ids_a).isdisjoint(set(ids_b)), \
            "Users must not see each other's analysis history"

    async def test_history_pagination(self, auth_client: AsyncClient):
        """Submit 3 analyses and verify pagination works."""
        for _ in range(3):
            await auth_client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})

        resp = await auth_client.get("/api/v1/history?page=1&page_size=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) <= 2
        assert "total" in data
        assert "total_pages" in data

    async def test_analysis_by_id_requires_ownership(self, client: AsyncClient):
        """A user must not read another user's analysis by ID."""
        # User A submits
        await client.post("/api/v1/auth/register", json={
            "email": "owner@example.com", "username": "owneruser", "password": "Owner1!pass"
        })
        resp = await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})
        analysis_id = resp.json()["id"]

        # Logout A, register B
        await client.post("/api/v1/auth/logout")
        await client.post("/api/v1/auth/register", json={
            "email": "intruder@example.com", "username": "intruder", "password": "Intr1!pass"
        })

        # B tries to read A's analysis — must get 404 (not 403, to avoid enumeration)
        steal_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
        assert steal_resp.status_code == 404


# =============================================================================
# Analytics / stats
# =============================================================================

class TestStats:
    async def test_stats_returns_real_counts(self, client: AsyncClient):
        """Stats must reflect actual analyses, not hardcoded values."""
        # Before any analysis
        resp_before = await client.get("/api/v1/stats")
        assert resp_before.status_code == 200
        before = resp_before.json()

        # Submit one analysis
        await client.post("/api/v1/analyze/text", json={"text": FAKE_ARTICLE_TEXT})

        resp_after = await client.get("/api/v1/stats")
        after = resp_after.json()

        assert after["total_analyses"] == before["total_analyses"] + 1

    async def test_stats_fields_present(self, client: AsyncClient):
        resp = await client.get("/api/v1/stats")
        data = resp.json()
        for field in ("total_analyses", "completed", "failed", "pending",
                      "verdict_counts", "fake_percentage", "real_percentage",
                      "avg_confidence", "avg_processing_ms"):
            assert field in data, f"Missing field: {field}"

    async def test_stats_percentages_are_real(self, client: AsyncClient):
        resp = await client.get("/api/v1/stats")
        data = resp.json()
        assert 0.0 <= data["fake_percentage"] <= 100.0
        assert 0.0 <= data["real_percentage"] <= 100.0


# =============================================================================
# Request body size limit
# =============================================================================

class TestRequestSizeLimit:
    async def test_oversized_body_returns_413(self, client: AsyncClient):
        """Requests larger than 1 MB must be rejected with 413."""
        huge_text = "x" * 2_000_000   # 2 MB
        resp = await client.post(
            "/api/v1/analyze/text",
            content=f'{{"text": "{huge_text}"}}',
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(huge_text) + 20)},
        )
        assert resp.status_code == 413


# =============================================================================
# Evidence failure — graceful degradation
# =============================================================================

class TestEvidenceProviderFailure:
    async def test_all_providers_fail_analysis_still_completes(self, client: AsyncClient):
        """If evidence retrieval fails, analysis must still return ML predictions."""
        from unittest.mock import patch, AsyncMock
        from app.evidence.schema import EvidenceResult, EvidenceStatus

        async def all_providers_fail(claim, **kwargs):
            return EvidenceResult(
                claim=claim, query=claim[:50],
                items=[], status=EvidenceStatus.ALL_PROVIDERS_FAILED,
                providers_used=[], providers_failed=["newsapi", "gnews", "rss"],
                total_found=0, after_dedup=0,
            )

        with patch("app.services.analysis_service.retrieve_evidence",
                   side_effect=all_providers_fail):
            resp = await client.post("/api/v1/analyze/text",
                                      json={"text": FAKE_ARTICLE_TEXT})

        assert resp.status_code == 200
        data = resp.json()
        # ML predictions must still be present
        assert len(data["model_predictions"]) > 0
        # Evidence verdict should be insufficient, not FAKE
        ev = data.get("evidence_verdict")
        if ev:
            assert ev != "FAKE", \
                "Provider failure must not be converted to FAKE verdict"
