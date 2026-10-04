"""
Unit tests for backend core layer.

Coverage:
  - AppError hierarchy and HTTP status codes
  - URL validation / SSRF prevention
  - Input sanitisation
  - JWT encode/decode round-trip
  - Password hashing (bcrypt)
  - Bad JWT tokens raise JWTError
"""
from __future__ import annotations

import pytest
from jose import JWTError


# =============================================================================
# Error hierarchy
# =============================================================================

class TestErrorHierarchy:
    def test_app_error_status_codes(self):
        from app.core.errors import (
            AppError, BadRequestError, NotFoundError,
            ModelUnavailableError, ArticleExtractionError, ValidationError,
        )
        assert AppError("x").status_code == 500
        assert BadRequestError("x").status_code == 400
        assert NotFoundError("x").status_code == 404
        assert ModelUnavailableError("x").status_code == 503
        assert ArticleExtractionError("x").status_code == 422
        assert ValidationError("x").status_code == 422

    def test_error_codes_are_strings(self):
        from app.core.errors import ModelUnavailableError, NotFoundError
        assert ModelUnavailableError.error_code == "MODEL_UNAVAILABLE"
        assert NotFoundError.error_code         == "NOT_FOUND"

    def test_detail_attached(self):
        from app.core.errors import BadRequestError
        err = BadRequestError("bad", detail={"field": "url"})
        assert err.detail == {"field": "url"}
        assert err.message == "bad"


# =============================================================================
# URL validation and SSRF prevention
# =============================================================================

class TestURLValidation:
    """validate_url must block private/internal IPs and accept public ones."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from app.core.security import validate_url, validate_redirect_url
        self.validate_url         = validate_url
        self.validate_redirect_url = validate_redirect_url

    # ── Valid ─────────────────────────────────────────────────────────────────
    def test_valid_https(self):
        url = self.validate_url("https://www.reuters.com/article/test")
        assert url == "https://www.reuters.com/article/test"

    def test_valid_http(self):
        url = self.validate_url("http://example.com/news")
        assert "example.com" in url

    def test_strips_whitespace(self):
        url = self.validate_url("  https://bbc.co.uk/news  ")
        assert url.startswith("https://")

    # ── Scheme rejection ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("bad_url", [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "data:text/html,<script>xss</script>",
        "javascript:alert(1)",
        "example.com/no-scheme",
    ])
    def test_bad_scheme_rejected(self, bad_url):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_url(bad_url)

    # ── SSRF — private IPs ────────────────────────────────────────────────────
    @pytest.mark.parametrize("private", [
        "http://10.0.0.1/admin",
        "http://172.16.0.1/secret",
        "http://192.168.1.1/router",
        "http://127.0.0.1/local",
        "http://127.0.0.2/loop",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "https://localhost/admin",
        "http://[::1]/internal",
    ])
    def test_private_ip_blocked(self, private):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_url(private)

    # ── Internal TLDs ─────────────────────────────────────────────────────────
    @pytest.mark.parametrize("internal", [
        "http://myserver.local/",
        "https://api.internal/v1",
        "http://app.corp/dashboard",
    ])
    def test_internal_tld_blocked(self, internal):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_url(internal)

    # ── Port restriction ──────────────────────────────────────────────────────
    def test_port_8080_rejected(self):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_url("http://example.com:8080/app")

    def test_port_443_allowed(self):
        url = self.validate_url("https://example.com:443/page")
        assert "example.com" in url

    # ── Length limit ──────────────────────────────────────────────────────────
    def test_too_long_rejected(self):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_url("https://example.com/" + "a" * 2000)

    def test_empty_rejected(self):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_url("")

    # ── Redirect validation ───────────────────────────────────────────────────
    def test_redirect_to_private_blocked(self):
        from app.core.errors import BadRequestError
        with pytest.raises(BadRequestError):
            self.validate_redirect_url("http://192.168.1.1/internal")

    def test_redirect_valid_passes(self):
        url = self.validate_redirect_url("https://www.bbc.co.uk/news/redirect")
        assert "bbc.co.uk" in url


# =============================================================================
# Text sanitisation
# =============================================================================

class TestSanitiseText:
    @pytest.fixture(autouse=True)
    def _import(self):
        from app.core.security import sanitise_text
        self.sanitise = sanitise_text

    def test_strips_whitespace(self):
        result = self.sanitise("  hello world  ")
        assert result == "hello world"

    def test_blank_raises(self):
        from app.core.errors import ValidationError
        with pytest.raises(ValidationError):
            self.sanitise("   ")

    def test_empty_raises(self):
        from app.core.errors import ValidationError
        with pytest.raises(ValidationError):
            self.sanitise("")

    def test_too_long_raises(self):
        from app.core.errors import ValidationError
        with pytest.raises(ValidationError):
            self.sanitise("x" * 100, max_length=50)

    def test_exact_max_length_passes(self):
        text = "a" * 50
        assert self.sanitise(text, max_length=50) == text


# =============================================================================
# Auth utilities — JWT + bcrypt
# =============================================================================

class TestAuthUtils:
    @pytest.fixture(autouse=True)
    def _import(self):
        import warnings
        warnings.filterwarnings("ignore")
        from app.core.auth_utils import (
            hash_password, verify_password,
            create_access_token, decode_access_token,
        )
        self.hash    = hash_password
        self.verify  = verify_password
        self.create  = create_access_token
        self.decode  = decode_access_token

    def test_hash_is_not_plaintext(self):
        h = self.hash("plaintext123")
        assert h != "plaintext123"
        assert h.startswith("$2b$")

    def test_verify_correct_password(self):
        h = self.hash("MyPass1!")
        assert self.verify("MyPass1!", h) is True

    def test_verify_wrong_password(self):
        h = self.hash("correct")
        assert self.verify("wrong", h) is False

    def test_different_hashes_for_same_password(self):
        """bcrypt uses random salt — same password → different hash each time."""
        h1 = self.hash("password")
        h2 = self.hash("password")
        assert h1 != h2

    def test_jwt_create_and_decode_round_trip(self):
        token = self.create(user_id=42, role="user")
        payload = self.decode(token)
        assert payload["sub"]  == "42"
        assert payload["role"] == "user"
        assert payload["type"] == "access"

    def test_jwt_tampered_raises(self):
        token = self.create(user_id=1, role="user")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(JWTError):
            self.decode(tampered)

    def test_jwt_empty_string_raises(self):
        with pytest.raises(JWTError):
            self.decode("")

    def test_jwt_contains_user_id(self):
        token = self.create(user_id=99, role="admin")
        payload = self.decode(token)
        assert int(payload["sub"]) == 99

    def test_jwt_role_preserved(self):
        token = self.create(user_id=1, role="analyst")
        payload = self.decode(token)
        assert payload["role"] == "analyst"
