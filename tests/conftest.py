"""
Root conftest.py — shared fixtures for all tests.

asyncio_mode = auto  set in pytest.ini so every async test runs without
@pytest.mark.asyncio decoration.
"""
import os
import sys
import warnings
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
ML_ROOT      = REPO_ROOT / "ml"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Silence bcrypt / passlib version-check stderr spam ───────────────────────
warnings.filterwarnings("ignore", message=".*bcrypt.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Test environment env vars ─────────────────────────────────────────────────
os.environ.setdefault("SENTENCE_TRANSFORMERS_ENABLED", "false")
os.environ.setdefault("APP_ENV",              "development")
os.environ.setdefault("SECRET_KEY",           "test-secret-key-for-tests-only-32c")
os.environ.setdefault("POSTGRES_HOST",        "localhost")
os.environ.setdefault("POSTGRES_DB",          "fakenews_test")
os.environ.setdefault("POSTGRES_USER",        "fakenews_user")
os.environ.setdefault("POSTGRES_PASSWORD",    "password")
os.environ.setdefault("COOKIE_SECURE",        "false")
os.environ.setdefault("NEWSAPI_KEY",          "")
os.environ.setdefault("GNEWS_API_KEY",        "")
os.environ.setdefault("SERPAPI_KEY",          "")


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "integration: mark as integration test (requires DB)")
    config.addinivalue_line("markers", "slow: mark as slow test")
