"""
Backend test fixtures.

Database strategy
-----------------
Tests use SQLite in-memory via aiosqlite, wired through a fresh
async engine and session per test.  No PostgreSQL required.

FastAPI strategy
----------------
httpx.AsyncClient + ASGITransport runs the full FastAPI app in-process.
ML model calls are mocked so tests run without trained artifacts.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# ── Path setup (backend must be importable) ───────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

warnings.filterwarnings("ignore")

from app.db.base import Base
import app.models  # noqa: registers all ORM models on Base.metadata

# ── In-memory SQLite engine ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_engine():
    """Create a fresh in-memory SQLite engine for each test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session; roll back after each test (no persistent data)."""
    session_factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


# ── FastAPI test client ───────────────────────────────────────────────────────

def _make_mock_registry():
    """Return a ModelRegistry mock that simulates baseline models available."""
    registry = MagicMock()
    registry._baseline_ready    = True
    registry._transformer_ready = False
    registry._baseline_ids      = ["logistic_regression", "linear_svm", "naive_bayes"]

    fake_pred = {
        "label":            "FAKE",
        "is_fake":          True,
        "confidence":       0.82,
        "fake_probability": 0.82,
        "real_probability": 0.18,
        "model_id":         "logistic_regression",
        "inference_time_ms": 12.4,
    }
    all_preds = {
        "logistic_regression": {**fake_pred, "model_id": "logistic_regression"},
        "linear_svm":          {**fake_pred, "model_id": "linear_svm",  "confidence": 0.79},
        "naive_bayes":         {**fake_pred, "model_id": "naive_bayes", "confidence": 0.71},
    }
    registry.predict_all_baseline = MagicMock(return_value=all_preds)
    registry.predict_transformer  = MagicMock(side_effect=Exception("not loaded"))
    registry.status               = MagicMock(return_value={
        "baseline_ready":    True,
        "transformer_ready": False,
        "available_models":  ["logistic_regression", "linear_svm", "naive_bayes"],
    })
    return registry


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Full FastAPI test client with:
      - in-memory SQLite DB (via db_session override)
      - mocked ML registry (no trained models needed)
      - mocked evidence retrieval (no real HTTP calls)
    """
    from app.main import create_app
    from app.db.session import get_db
    from app.ml.model_registry import get_registry
    from app.evidence.evidence_service import retrieve_evidence
    from app.evidence.schema import EvidenceResult, EvidenceStatus

    mock_registry = _make_mock_registry()

    # Evidence mock — returns empty result by default
    async def mock_retrieve_evidence(claim: str, **kwargs):
        return EvidenceResult(
            claim=claim,
            query=claim[:50],
            items=[],
            status=EvidenceStatus.INSUFFICIENT,
            providers_used=[],
            providers_failed=[],
            total_found=0,
            after_dedup=0,
        )

    app = create_app()

    # Override DB dependency to use SQLite session
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("app.ml.model_registry.ModelRegistry.get", return_value=mock_registry),
        patch("app.services.analysis_service.retrieve_evidence", side_effect=mock_retrieve_evidence),
        patch("app.services.analysis_service._generate_explanations", new_callable=AsyncMock, return_value={}),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac


@pytest.fixture
async def auth_client(client: AsyncClient) -> AsyncClient:
    """Test client with a registered + logged-in user."""
    resp = await client.post("/api/v1/auth/register", json={
        "email":    "test@example.com",
        "username": "testuser",
        "password": "TestPass1!",
    })
    assert resp.status_code == 201, f"Register failed: {resp.text}"
    return client


# ── Helper data ───────────────────────────────────────────────────────────────

FAKE_ARTICLE_TEXT = (
    "Scientists discover miracle cure that doctors are suppressing "
    "from the public to protect pharmaceutical profits and government conspiracy. "
    "The shocking truth has been hidden for decades according to unnamed sources. "
    "This revolutionary treatment cures all diseases with no side effects. "
    "The medical establishment is hiding this from you to protect drug company profits."
)

REAL_ARTICLE_TEXT = (
    "The European Central Bank raised its benchmark interest rate by 25 basis points "
    "on Thursday, citing persistent inflationary pressures across the eurozone. "
    "ECB President Christine Lagarde stated the decision was unanimous among board members. "
    "Markets reacted positively, with bond yields falling slightly after the announcement. "
    "Analysts expect the bank to pause rate increases in the coming months."
)

SHORT_CLAIM = "The WHO declared COVID-19 a pandemic in March 2020."
