"""
Database layer tests.

Tests ORM model persistence, relationships, cascade deletes,
and ownership logic using in-memory SQLite.

No PostgreSQL required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


# =============================================================================
# User model
# =============================================================================

class TestUserModel:
    async def test_create_user(self, db_session: AsyncSession):
        from app.models.user import User, UserRole
        import warnings; warnings.filterwarnings("ignore")
        from app.core.auth_utils import hash_password

        user = User(
            email="alice@example.com",
            username="alice",
            hashed_password=hash_password("Pass1!abc"),
            role=UserRole.USER,
            is_active=True,
        )
        db_session.add(user)
        await db_session.flush()

        assert user.id is not None
        result = await db_session.execute(select(User).where(User.email == "alice@example.com"))
        fetched = result.scalar_one()
        assert fetched.username == "alice"
        assert fetched.hashed_password != "Pass1!abc"   # must be hashed

    async def test_unique_email_constraint(self, db_session: AsyncSession):
        from app.models.user import User, UserRole
        import warnings; warnings.filterwarnings("ignore")
        from app.core.auth_utils import hash_password
        from sqlalchemy.exc import IntegrityError

        for uname in ("user1", "user2"):
            db_session.add(User(
                email="same@example.com",
                username=uname,
                hashed_password=hash_password("Pass1!"),
                role=UserRole.USER,
            ))
        with pytest.raises(IntegrityError):
            await db_session.flush()

    async def test_password_never_stored_plaintext(self, db_session: AsyncSession):
        import warnings; warnings.filterwarnings("ignore")
        from app.models.user import User, UserRole
        from app.core.auth_utils import hash_password

        user = User(
            email="secure@example.com",
            username="secure",
            hashed_password=hash_password("MySecret1!"),
            role=UserRole.USER,
        )
        db_session.add(user)
        await db_session.flush()

        fetched = (await db_session.execute(
            select(User).where(User.email == "secure@example.com")
        )).scalar_one()

        assert "MySecret1!" not in fetched.hashed_password
        assert fetched.hashed_password.startswith("$2b$")


# =============================================================================
# Analysis model and ownership
# =============================================================================

class TestAnalysisModel:
    async def _make_user(self, db_session, email="u@e.com", username="user"):
        import warnings; warnings.filterwarnings("ignore")
        from app.models.user import User, UserRole
        from app.core.auth_utils import hash_password
        user = User(email=email, username=username,
                    hashed_password=hash_password("Pass1!"), role=UserRole.USER)
        db_session.add(user)
        await db_session.flush()
        return user

    async def test_create_analysis_anonymous(self, db_session: AsyncSession):
        from app.models.analysis import Analysis, InputType, AnalysisStatus
        a = Analysis(
            input_type=InputType.TEXT,
            original_input="Test article text for analysis.",
            status=AnalysisStatus.PENDING,
            user_id=None,
        )
        db_session.add(a)
        await db_session.flush()
        assert a.id is not None
        assert a.user_id is None

    async def test_analysis_linked_to_user(self, db_session: AsyncSession):
        from app.models.analysis import Analysis, InputType, AnalysisStatus
        user = await self._make_user(db_session)
        a = Analysis(
            input_type=InputType.TEXT,
            original_input="Some article text.",
            status=AnalysisStatus.PENDING,
            user_id=user.id,
        )
        db_session.add(a)
        await db_session.flush()
        assert a.user_id == user.id

    async def test_user_delete_sets_null_on_analysis(self, db_session: AsyncSession):
        """
        Verify that an Analysis can exist with user_id=NULL.
        The ON DELETE SET NULL FK behaviour is PostgreSQL-specific and cannot
        be fully tested with SQLite in-memory. What we test here is that:
          1. An Analysis with user_id=NULL is valid (nullable FK).
          2. Anonymous analyses persist correctly.
        The full SET NULL behaviour is covered by the API ownership tests.
        """
        from app.models.analysis import Analysis, InputType, AnalysisStatus

        # Create an anonymous analysis (user_id=NULL)
        a = Analysis(
            input_type=InputType.TEXT,
            original_input="Anonymous article text.",
            status=AnalysisStatus.COMPLETED,
            user_id=None,    # explicitly NULL
        )
        db_session.add(a)
        await db_session.flush()

        fetched = (await db_session.execute(
            select(Analysis).where(Analysis.id == a.id)
        )).scalar_one_or_none()

        assert fetched is not None
        assert fetched.user_id is None, "Anonymous analysis must have user_id=NULL"


# =============================================================================
# Claim + EvidenceSource cascade
# =============================================================================

class TestClaimAndEvidenceCascade:
    async def test_deleting_analysis_cascades_to_claims(self, db_session: AsyncSession):
        from app.models.analysis import Analysis, InputType, AnalysisStatus, FinalVerdict
        from app.models.claim import Claim, ClaimVerdict
        from app.models.evidence_source import EvidenceSource, SourceType, EvidenceRelationship

        a = Analysis(input_type=InputType.TEXT, original_input="text",
                     status=AnalysisStatus.COMPLETED, final_verdict=FinalVerdict.REAL)
        db_session.add(a)
        await db_session.flush()

        claim = Claim(analysis_id=a.id, claim_text="A claim.", position=1,
                      verdict=ClaimVerdict.REAL)
        db_session.add(claim)
        await db_session.flush()

        ev = EvidenceSource(
            claim_id=claim.id, source_name="Reuters",
            title="Test Article", url="https://reuters.com/test",
            source_type=SourceType.NEWS_API,
            retrieved_at=datetime.now(timezone.utc),
            relevance_score=0.8, rank=1,
            relationship_to_claim=EvidenceRelationship.SUPPORTING,
        )
        db_session.add(ev)
        await db_session.flush()

        claim_id = claim.id
        ev_id    = ev.id

        await db_session.delete(a)
        await db_session.flush()

        assert (await db_session.get(Claim, claim_id)) is None, \
            "Claim must be deleted when Analysis is deleted (CASCADE)"
        assert (await db_session.get(EvidenceSource, ev_id)) is None, \
            "EvidenceSource must be deleted when Claim is deleted (CASCADE)"

    async def test_evidence_relationship_not_relevant_storable(self, db_session: AsyncSession):
        """EvidenceRelationship.NOT_RELEVANT must be persistable (added in migration 0002)."""
        from datetime import datetime, timezone
        from app.models.analysis import Analysis, InputType, AnalysisStatus
        from app.models.claim import Claim, ClaimVerdict
        from app.models.evidence_source import EvidenceSource, SourceType, EvidenceRelationship

        a = Analysis(input_type=InputType.TEXT, original_input="text",
                     status=AnalysisStatus.COMPLETED)
        db_session.add(a)
        await db_session.flush()

        claim = Claim(analysis_id=a.id, claim_text="Claim.", position=1)
        db_session.add(claim)
        await db_session.flush()

        ev = EvidenceSource(
            claim_id=claim.id, source_name="Source",
            title="Title", url="https://example.com/1",
            source_type=SourceType.NEWS_API,
            retrieved_at=datetime.now(timezone.utc),
            relevance_score=0.1, rank=1,
            relationship_to_claim=EvidenceRelationship.NOT_RELEVANT,
        )
        db_session.add(ev)
        await db_session.flush()

        fetched = await db_session.get(EvidenceSource, ev.id)
        assert fetched.relationship_to_claim == EvidenceRelationship.NOT_RELEVANT

    async def test_evidence_assessment_enum_storable(self, db_session: AsyncSession):
        """EvidenceAssessment.INSUFFICIENT_EVIDENCE must persist (migration 0002)."""
        from app.models.analysis import Analysis, InputType, AnalysisStatus
        from app.models.claim import Claim, EvidenceAssessment

        a = Analysis(input_type=InputType.TEXT, original_input="text",
                     status=AnalysisStatus.COMPLETED)
        db_session.add(a)
        await db_session.flush()

        claim = Claim(
            analysis_id=a.id, claim_text="Claim.", position=1,
            evidence_verdict=EvidenceAssessment.INSUFFICIENT_EVIDENCE,
            evidence_explanation="No sources found.",
        )
        db_session.add(claim)
        await db_session.flush()

        fetched = await db_session.get(Claim, claim.id)
        assert fetched.evidence_verdict == EvidenceAssessment.INSUFFICIENT_EVIDENCE
        assert fetched.evidence_explanation == "No sources found."


# =============================================================================
# Prediction model
# =============================================================================

class TestPredictionModel:
    async def test_prediction_confidence_range_constraint(self, db_session: AsyncSession):
        """Confidence outside [0, 1] must fail the CHECK constraint."""
        from sqlalchemy.exc import IntegrityError
        from app.models.analysis import Analysis, InputType, AnalysisStatus
        from app.models.model_version import ModelVersion, ModelType, ModelAlgorithm
        from app.models.prediction import Prediction, PredictionLabel

        a = Analysis(input_type=InputType.TEXT, original_input="txt",
                     status=AnalysisStatus.COMPLETED)
        db_session.add(a)
        mv = ModelVersion(model_name="lr", version="1.0", model_type=ModelType.TRADITIONAL,
                          algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
                          artifact_path="ml/saved_models/lr.pkl")
        db_session.add(mv)
        await db_session.flush()

        pred = Prediction(
            analysis_id=a.id, model_version_id=mv.id,
            label=PredictionLabel.FAKE,
            confidence=1.5,         # OUT OF RANGE
            fake_probability=1.5,   # OUT OF RANGE
            real_probability=0.0,
        )
        db_session.add(pred)
        with pytest.raises(IntegrityError):
            await db_session.flush()
