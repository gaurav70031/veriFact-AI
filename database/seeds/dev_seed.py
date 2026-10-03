"""
DEV/TEST SEED DATA — NOT FOR PRODUCTION
========================================
This script populates the database with realistic-looking but entirely
fictional development data so the frontend and analytics can be developed
and tested without needing real user submissions or trained models.

WARNING: Never run this against a production database.
         All data here is fabricated for development purposes only.

Usage (from the backend/ directory with .env loaded):
    python ../database/seeds/dev_seed.py

Or with explicit env:
    DATABASE_URL=postgresql://... python dev_seed.py

Requirements:
    pip install sqlalchemy psycopg2-binary passlib[bcrypt] python-dotenv
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from the database/seeds/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / "backend" / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from app.core.config import get_settings
from app.db.base import Base
from app.models import (
    User, UserRole,
    Analysis, InputType, AnalysisStatus, FinalVerdict,
    Claim, ClaimVerdict,
    ModelVersion, ModelType, ModelAlgorithm,
    Prediction, PredictionLabel,
    EvidenceSource, SourceType, EvidenceRelationship,
    ModelRun, RunStatus,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
settings = get_settings()
SYNC_URL = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

NOW = datetime.now(timezone.utc)


def ts(days_ago: float = 0, hours_ago: float = 0) -> datetime:
    """Return a timezone-aware timestamp offset from now."""
    return NOW - timedelta(days=days_ago, hours=hours_ago)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def seed_users(session: Session) -> list[User]:
    """
    [DEV DATA] Three fictional user accounts for testing different roles.
    Passwords are bcrypt-hashed — not stored in plaintext.
    """
    users = [
        User(
            email="dev.admin@example.test",
            username="dev_admin",
            hashed_password=pwd_ctx.hash("DevAdmin@1234"),
            full_name="Dev Admin",
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
            created_at=ts(days_ago=30),
            updated_at=ts(days_ago=30),
        ),
        User(
            email="analyst.test@example.test",
            username="test_analyst",
            hashed_password=pwd_ctx.hash("Analyst@5678"),
            full_name="Test Analyst",
            role=UserRole.ANALYST,
            is_active=True,
            is_verified=True,
            created_at=ts(days_ago=20),
            updated_at=ts(days_ago=20),
        ),
        User(
            email="regular.user@example.test",
            username="regular_user",
            hashed_password=pwd_ctx.hash("User@9999"),
            full_name="Regular User",
            role=UserRole.USER,
            is_active=True,
            is_verified=False,
            created_at=ts(days_ago=5),
            updated_at=ts(days_ago=5),
        ),
    ]
    session.add_all(users)
    session.flush()
    print(f"  ✓ Seeded {len(users)} users")
    return users


def seed_model_versions(session: Session) -> list[ModelVersion]:
    """
    [DEV DATA] Four model version records matching the planned ML artifacts.
    Metrics are representative of what a well-trained model should achieve —
    they are placeholders until real training runs are completed.
    """
    versions = [
        ModelVersion(
            model_name="tfidf_logistic_regression",
            version="1.0.0",
            model_type=ModelType.TRADITIONAL,
            algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
            artifact_path="ml/models/saved/lr_model.pkl",
            vectorizer_path="ml/models/saved/tfidf_vectorizer.pkl",
            dataset_name="ISOT Fake News Dataset",
            training_samples=35918,
            test_samples=8980,
            accuracy=0.9842,
            precision=0.9851,
            recall=0.9833,
            f1_score=0.9842,
            roc_auc=0.9971,
            is_active=True,
            description="TF-IDF vectorizer (max 50k features, 1-2 ngrams) + "
                        "Logistic Regression (C=1.0, max_iter=1000).",
            created_at=ts(days_ago=14),
            updated_at=ts(days_ago=14),
        ),
        ModelVersion(
            model_name="tfidf_linear_svm",
            version="1.0.0",
            model_type=ModelType.TRADITIONAL,
            algorithm=ModelAlgorithm.LINEAR_SVM,
            artifact_path="ml/models/saved/svm_model.pkl",
            vectorizer_path="ml/models/saved/tfidf_vectorizer.pkl",
            dataset_name="ISOT Fake News Dataset",
            training_samples=35918,
            test_samples=8980,
            accuracy=0.9901,
            precision=0.9907,
            recall=0.9895,
            f1_score=0.9901,
            roc_auc=0.9988,
            is_active=True,
            description="TF-IDF vectorizer (max 50k features, 1-2 ngrams) + "
                        "LinearSVC (C=1.0) wrapped in CalibratedClassifierCV.",
            created_at=ts(days_ago=14),
            updated_at=ts(days_ago=14),
        ),
        ModelVersion(
            model_name="tfidf_naive_bayes",
            version="1.0.0",
            model_type=ModelType.TRADITIONAL,
            algorithm=ModelAlgorithm.NAIVE_BAYES,
            artifact_path="ml/models/saved/nb_model.pkl",
            vectorizer_path="ml/models/saved/tfidf_vectorizer.pkl",
            dataset_name="ISOT Fake News Dataset",
            training_samples=35918,
            test_samples=8980,
            accuracy=0.9423,
            precision=0.9441,
            recall=0.9402,
            f1_score=0.9421,
            roc_auc=0.9812,
            is_active=True,
            description="TF-IDF vectorizer + Multinomial Naive Bayes (alpha=0.1).",
            created_at=ts(days_ago=14),
            updated_at=ts(days_ago=14),
        ),
        ModelVersion(
            model_name="distilbert_fakenews",
            version="1.0.0",
            model_type=ModelType.TRANSFORMER,
            algorithm=ModelAlgorithm.DISTILBERT,
            artifact_path="ml/models/saved/distilbert",
            vectorizer_path=None,
            dataset_name="ISOT Fake News Dataset",
            training_samples=35918,
            test_samples=8980,
            accuracy=0.9934,
            precision=0.9938,
            recall=0.9930,
            f1_score=0.9934,
            roc_auc=0.9995,
            is_active=True,
            description="DistilBERT-base-uncased fine-tuned for 3 epochs on ISOT "
                        "(batch=16, lr=2e-5, max_len=512).",
            created_at=ts(days_ago=7),
            updated_at=ts(days_ago=7),
        ),
        ModelVersion(
            model_name="ensemble_v1",
            version="1.0.0",
            model_type=ModelType.ENSEMBLE,
            algorithm=ModelAlgorithm.ENSEMBLE,
            artifact_path="ml/models/saved/ensemble_config.json",
            vectorizer_path=None,
            dataset_name="ISOT Fake News Dataset",
            training_samples=35918,
            test_samples=8980,
            accuracy=0.9951,
            precision=0.9954,
            recall=0.9948,
            f1_score=0.9951,
            roc_auc=0.9997,
            is_active=True,
            description="Weighted average ensemble: LR(0.2) + SVM(0.3) + NB(0.1) + "
                        "DistilBERT(0.4).",
            created_at=ts(days_ago=7),
            updated_at=ts(days_ago=7),
        ),
    ]
    session.add_all(versions)
    session.flush()
    print(f"  ✓ Seeded {len(versions)} model versions")
    return versions


def seed_analyses(
    session: Session,
    users: list[User],
    model_versions: list[ModelVersion],
) -> None:
    """
    [DEV DATA] Six fictional analyses covering all input types and verdicts.
    These are used to populate the history page and analytics charts.
    """
    lr, svm, nb, distilbert, ensemble = model_versions

    # ── Analysis 1: FAKE news article via URL ────────────────────────────────
    a1 = Analysis(
        user_id=users[0].id,
        input_type=InputType.URL,
        original_input="https://example-fake-news.test/miracle-cure-article",
        source_url="https://example-fake-news.test/miracle-cure-article",
        article_title="Scientists Discover Miracle Cure That Doctors Don't Want You to Know",
        article_text=(
            "Researchers at an unnamed university have allegedly discovered a simple "
            "household remedy that cures all known diseases. The medical establishment "
            "is suppressing this information to protect pharmaceutical profits..."
        ),
        status=AnalysisStatus.COMPLETED,
        final_verdict=FinalVerdict.FAKE,
        final_confidence=0.9712,
        summary=(
            "High likelihood of misinformation. Article uses sensationalist language, "
            "cites unnamed sources, and makes unverifiable medical claims. "
            "No corroborating evidence found in reputable outlets."
        ),
        processing_time_ms=1842,
        created_at=ts(days_ago=10),
        updated_at=ts(days_ago=10),
    )
    session.add(a1)
    session.flush()

    c1 = Claim(
        analysis_id=a1.id,
        claim_text=(
            "Scientists discovered a miracle cure that doctors are suppressing "
            "to protect pharmaceutical profits."
        ),
        position=1,
        verdict=ClaimVerdict.FAKE,
        confidence=0.9712,
        explanation=(
            "No peer-reviewed study supports this claim. "
            "Language pattern ('doctors don't want you to know') is a known "
            "misinformation marker. No named institution or researcher cited."
        ),
        top_tokens_json=(
            '[{"token": "miracle", "weight": 0.312}, '
            '{"token": "suppressing", "weight": 0.287}, '
            '{"token": "unnamed", "weight": 0.198}, '
            '{"token": "profits", "weight": 0.156}, '
            '{"token": "allegedly", "weight": 0.134}]'
        ),
        created_at=ts(days_ago=10),
        updated_at=ts(days_ago=10),
    )
    session.add(c1)
    session.flush()

    session.add_all([
        EvidenceSource(
            claim_id=c1.id,
            source_name="Reuters",
            title="No evidence for suppressed miracle cure claims, say health experts",
            url="https://www.reuters.com/fact-check/example-debunk",
            snippet="Health experts have repeatedly debunked claims of suppressed cures, "
                    "noting such narratives exploit distrust in medical institutions.",
            source_type=SourceType.FACT_CHECK,
            published_at=ts(days_ago=15),
            retrieved_at=ts(days_ago=10),
            relevance_score=0.891,
            rank=1,
            relationship_to_claim=EvidenceRelationship.CONTRADICTING,
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
        EvidenceSource(
            claim_id=c1.id,
            source_name="WHO",
            title="Infodemic management — identifying misinformation in health claims",
            url="https://www.who.int/teams/risk-communication/infodemic-management",
            snippet="The WHO notes that sensationalist medical claims without peer-review "
                    "citations are a common form of health misinformation.",
            source_type=SourceType.OFFICIAL,
            published_at=ts(days_ago=180),
            retrieved_at=ts(days_ago=10),
            relevance_score=0.743,
            rank=2,
            relationship_to_claim=EvidenceRelationship.CONTRADICTING,
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
    ])

    session.add_all([
        Prediction(
            analysis_id=a1.id,
            model_version_id=lr.id,
            label=PredictionLabel.FAKE,
            confidence=0.9634,
            fake_probability=0.9634,
            real_probability=0.0366,
            explanation_json=(
                '[{"token": "miracle", "weight": 0.31}, '
                '{"token": "suppressing", "weight": 0.29}]'
            ),
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
        Prediction(
            analysis_id=a1.id,
            model_version_id=svm.id,
            label=PredictionLabel.FAKE,
            confidence=0.9801,
            fake_probability=0.9801,
            real_probability=0.0199,
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
        Prediction(
            analysis_id=a1.id,
            model_version_id=nb.id,
            label=PredictionLabel.FAKE,
            confidence=0.9312,
            fake_probability=0.9312,
            real_probability=0.0688,
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
        Prediction(
            analysis_id=a1.id,
            model_version_id=distilbert.id,
            label=PredictionLabel.FAKE,
            confidence=0.9878,
            fake_probability=0.9878,
            real_probability=0.0122,
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
        Prediction(
            analysis_id=a1.id,
            model_version_id=ensemble.id,
            label=PredictionLabel.FAKE,
            confidence=0.9712,
            fake_probability=0.9712,
            real_probability=0.0288,
            created_at=ts(days_ago=10),
            updated_at=ts(days_ago=10),
        ),
    ])

    session.add_all([
        ModelRun(analysis_id=a1.id, model_version_id=lr.id,
                 status=RunStatus.SUCCESS, inference_time_ms=12,
                 input_length=312, succeeded=True,
                 created_at=ts(days_ago=10), updated_at=ts(days_ago=10)),
        ModelRun(analysis_id=a1.id, model_version_id=svm.id,
                 status=RunStatus.SUCCESS, inference_time_ms=9,
                 input_length=312, succeeded=True,
                 created_at=ts(days_ago=10), updated_at=ts(days_ago=10)),
        ModelRun(analysis_id=a1.id, model_version_id=nb.id,
                 status=RunStatus.SUCCESS, inference_time_ms=7,
                 input_length=312, succeeded=True,
                 created_at=ts(days_ago=10), updated_at=ts(days_ago=10)),
        ModelRun(analysis_id=a1.id, model_version_id=distilbert.id,
                 status=RunStatus.SUCCESS, inference_time_ms=1621,
                 input_length=312, succeeded=True,
                 created_at=ts(days_ago=10), updated_at=ts(days_ago=10)),
    ])

    # ── Analysis 2: REAL news article via text input ─────────────────────────
    a2 = Analysis(
        user_id=users[1].id,
        input_type=InputType.TEXT,
        original_input=(
            "The European Central Bank raised its key interest rate by 25 basis "
            "points on Thursday, citing persistent inflationary pressures across "
            "the eurozone. ECB President Christine Lagarde stated the decision was "
            "unanimous and in line with the bank's 2% inflation target mandate."
        ),
        article_title=None,
        status=AnalysisStatus.COMPLETED,
        final_verdict=FinalVerdict.REAL,
        final_confidence=0.9543,
        summary=(
            "Content consistent with factual financial reporting. "
            "Claim is corroborated by multiple independent news sources and "
            "official ECB communications."
        ),
        processing_time_ms=2103,
        created_at=ts(days_ago=7, hours_ago=3),
        updated_at=ts(days_ago=7, hours_ago=3),
    )
    session.add(a2)
    session.flush()

    c2 = Claim(
        analysis_id=a2.id,
        claim_text=(
            "The ECB raised its key interest rate by 25 basis points, "
            "citing persistent inflation in the eurozone."
        ),
        position=1,
        verdict=ClaimVerdict.REAL,
        confidence=0.9543,
        explanation=(
            "Claim matches official ECB press release language and is corroborated "
            "by Reuters, Bloomberg, and the ECB website directly."
        ),
        top_tokens_json=(
            '[{"token": "ECB", "weight": 0.341}, '
            '{"token": "interest rate", "weight": 0.298}, '
            '{"token": "inflation", "weight": 0.201}, '
            '{"token": "eurozone", "weight": 0.167}]'
        ),
        created_at=ts(days_ago=7, hours_ago=3),
        updated_at=ts(days_ago=7, hours_ago=3),
    )
    session.add(c2)
    session.flush()

    session.add_all([
        EvidenceSource(
            claim_id=c2.id,
            source_name="ECB",
            title="Monetary policy decisions — ECB raises rates by 25bps",
            url="https://www.ecb.europa.eu/press/pr/date/2024/html/example.en.html",
            snippet="The Governing Council decided to raise the three key ECB interest "
                    "rates by 25 basis points.",
            source_type=SourceType.OFFICIAL,
            published_at=ts(days_ago=8),
            retrieved_at=ts(days_ago=7, hours_ago=3),
            relevance_score=0.964,
            rank=1,
            relationship_to_claim=EvidenceRelationship.SUPPORTING,
            created_at=ts(days_ago=7, hours_ago=3),
            updated_at=ts(days_ago=7, hours_ago=3),
        ),
        EvidenceSource(
            claim_id=c2.id,
            source_name="Reuters",
            title="ECB hikes rates, Lagarde signals more if needed",
            url="https://www.reuters.com/markets/europe/ecb-example-article",
            snippet="The European Central Bank raised borrowing costs again on Thursday "
                    "as it battles to bring inflation back to its 2% target.",
            source_type=SourceType.NEWS_API,
            published_at=ts(days_ago=8),
            retrieved_at=ts(days_ago=7, hours_ago=3),
            relevance_score=0.921,
            rank=2,
            relationship_to_claim=EvidenceRelationship.SUPPORTING,
            created_at=ts(days_ago=7, hours_ago=3),
            updated_at=ts(days_ago=7, hours_ago=3),
        ),
    ])

    session.add_all([
        Prediction(analysis_id=a2.id, model_version_id=lr.id,
                   label=PredictionLabel.REAL, confidence=0.9421,
                   fake_probability=0.0579, real_probability=0.9421,
                   created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
        Prediction(analysis_id=a2.id, model_version_id=svm.id,
                   label=PredictionLabel.REAL, confidence=0.9601,
                   fake_probability=0.0399, real_probability=0.9601,
                   created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
        Prediction(analysis_id=a2.id, model_version_id=nb.id,
                   label=PredictionLabel.REAL, confidence=0.9102,
                   fake_probability=0.0898, real_probability=0.9102,
                   created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
        Prediction(analysis_id=a2.id, model_version_id=distilbert.id,
                   label=PredictionLabel.REAL, confidence=0.9712,
                   fake_probability=0.0288, real_probability=0.9712,
                   created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
        Prediction(analysis_id=a2.id, model_version_id=ensemble.id,
                   label=PredictionLabel.REAL, confidence=0.9543,
                   fake_probability=0.0457, real_probability=0.9543,
                   created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
    ])

    session.add_all([
        ModelRun(analysis_id=a2.id, model_version_id=lr.id,
                 status=RunStatus.SUCCESS, inference_time_ms=14,
                 input_length=421, succeeded=True,
                 created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
        ModelRun(analysis_id=a2.id, model_version_id=svm.id,
                 status=RunStatus.SUCCESS, inference_time_ms=11,
                 input_length=421, succeeded=True,
                 created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
        ModelRun(analysis_id=a2.id, model_version_id=distilbert.id,
                 status=RunStatus.SUCCESS, inference_time_ms=1834,
                 input_length=421, succeeded=True,
                 created_at=ts(days_ago=7), updated_at=ts(days_ago=7)),
    ])

    # ── Analysis 3: FAKE — search query submission ────────────────────────────
    a3 = Analysis(
        user_id=users[2].id,
        input_type=InputType.QUERY,
        original_input="5G towers cause COVID-19",
        status=AnalysisStatus.COMPLETED,
        final_verdict=FinalVerdict.FAKE,
        final_confidence=0.9987,
        summary=(
            "This is a widely debunked conspiracy theory. No scientific basis exists "
            "for a connection between 5G radio waves and viral disease. "
            "Multiple health organisations have explicitly refuted this claim."
        ),
        processing_time_ms=3241,
        created_at=ts(days_ago=4),
        updated_at=ts(days_ago=4),
    )
    session.add(a3)
    session.flush()

    c3 = Claim(
        analysis_id=a3.id,
        claim_text="5G towers cause or spread COVID-19.",
        position=1,
        verdict=ClaimVerdict.FAKE,
        confidence=0.9987,
        explanation=(
            "COVID-19 is caused by the SARS-CoV-2 virus, a biological pathogen. "
            "Radio waves cannot carry or transmit viruses. WHO, CDC, and ICNIRP "
            "have all explicitly debunked this claim."
        ),
        top_tokens_json=(
            '[{"token": "5G", "weight": 0.421}, '
            '{"token": "cause", "weight": 0.187}, '
            '{"token": "towers", "weight": 0.143}]'
        ),
        created_at=ts(days_ago=4),
        updated_at=ts(days_ago=4),
    )
    session.add(c3)
    session.flush()

    session.add_all([
        EvidenceSource(
            claim_id=c3.id,
            source_name="WHO",
            title="5G mobile networks and health — WHO fact sheet",
            url="https://www.who.int/news-room/questions-and-answers/item/radiation-5g-networks",
            snippet="5G technology does not spread COVID-19. Viruses cannot travel on "
                    "radio waves or mobile networks.",
            source_type=SourceType.OFFICIAL,
            published_at=ts(days_ago=400),
            retrieved_at=ts(days_ago=4),
            relevance_score=0.981,
            rank=1,
            relationship_to_claim=EvidenceRelationship.CONTRADICTING,
            created_at=ts(days_ago=4),
            updated_at=ts(days_ago=4),
        ),
        EvidenceSource(
            claim_id=c3.id,
            source_name="Full Fact",
            title="5G and coronavirus: the facts",
            url="https://fullfact.org/online/5g-coronavirus-theory/",
            snippet="There is no connection between 5G and COVID-19. This theory has "
                    "been repeatedly investigated and found to have no factual basis.",
            source_type=SourceType.FACT_CHECK,
            published_at=ts(days_ago=500),
            retrieved_at=ts(days_ago=4),
            relevance_score=0.974,
            rank=2,
            relationship_to_claim=EvidenceRelationship.CONTRADICTING,
            created_at=ts(days_ago=4),
            updated_at=ts(days_ago=4),
        ),
    ])

    session.add_all([
        Prediction(analysis_id=a3.id, model_version_id=lr.id,
                   label=PredictionLabel.FAKE, confidence=0.9971,
                   fake_probability=0.9971, real_probability=0.0029,
                   created_at=ts(days_ago=4), updated_at=ts(days_ago=4)),
        Prediction(analysis_id=a3.id, model_version_id=distilbert.id,
                   label=PredictionLabel.FAKE, confidence=0.9994,
                   fake_probability=0.9994, real_probability=0.0006,
                   created_at=ts(days_ago=4), updated_at=ts(days_ago=4)),
        Prediction(analysis_id=a3.id, model_version_id=ensemble.id,
                   label=PredictionLabel.FAKE, confidence=0.9987,
                   fake_probability=0.9987, real_probability=0.0013,
                   created_at=ts(days_ago=4), updated_at=ts(days_ago=4)),
    ])

    # ── Analysis 4: UNVERIFIED — ambiguous claim ──────────────────────────────
    a4 = Analysis(
        user_id=users[1].id,
        input_type=InputType.TEXT,
        original_input=(
            "The government plans to introduce a new digital currency "
            "to replace physical cash by 2026."
        ),
        status=AnalysisStatus.COMPLETED,
        final_verdict=FinalVerdict.UNVERIFIED,
        final_confidence=0.5812,
        summary=(
            "Claim references real ongoing discussions about Central Bank Digital "
            "Currencies (CBDCs) but the specific timeline and scope are not confirmed "
            "by official sources. Insufficient evidence to verify or refute."
        ),
        processing_time_ms=2891,
        created_at=ts(days_ago=2),
        updated_at=ts(days_ago=2),
    )
    session.add(a4)
    session.flush()

    c4 = Claim(
        analysis_id=a4.id,
        claim_text="The government will replace physical cash with a digital currency by 2026.",
        position=1,
        verdict=ClaimVerdict.UNVERIFIED,
        confidence=0.5812,
        explanation=(
            "CBDC research is ongoing in many countries but no confirmed rollout "
            "timeline or cash replacement policy has been officially announced."
        ),
        created_at=ts(days_ago=2),
        updated_at=ts(days_ago=2),
    )
    session.add(c4)
    session.flush()

    session.add(EvidenceSource(
        claim_id=c4.id,
        source_name="Bank for International Settlements",
        title="CBDCs: an opportunity for the monetary system",
        url="https://www.bis.org/publ/arpdf/ar2021e3.htm",
        snippet="Central banks worldwide are exploring digital currencies, but concrete "
                "timelines for replacing cash vary significantly by jurisdiction.",
        source_type=SourceType.OFFICIAL,
        published_at=ts(days_ago=365),
        retrieved_at=ts(days_ago=2),
        relevance_score=0.712,
        rank=1,
        relationship_to_claim=EvidenceRelationship.INCONCLUSIVE,
        created_at=ts(days_ago=2),
        updated_at=ts(days_ago=2),
    ))

    session.add_all([
        Prediction(analysis_id=a4.id, model_version_id=lr.id,
                   label=PredictionLabel.FAKE, confidence=0.5912,
                   fake_probability=0.5912, real_probability=0.4088,
                   created_at=ts(days_ago=2), updated_at=ts(days_ago=2)),
        Prediction(analysis_id=a4.id, model_version_id=distilbert.id,
                   label=PredictionLabel.REAL, confidence=0.5341,
                   fake_probability=0.4659, real_probability=0.5341,
                   created_at=ts(days_ago=2), updated_at=ts(days_ago=2)),
    ])

    # ── Analysis 5: FAILED analysis (for error-state testing) ────────────────
    a5 = Analysis(
        user_id=users[0].id,
        input_type=InputType.URL,
        original_input="https://paywalled-journal.test/restricted-article-123",
        source_url="https://paywalled-journal.test/restricted-article-123",
        status=AnalysisStatus.FAILED,
        error_message=(
            "Unable to extract article content: HTTP 403 Forbidden. "
            "The target URL is behind a paywall or access restriction."
        ),
        processing_time_ms=412,
        created_at=ts(days_ago=1),
        updated_at=ts(days_ago=1),
    )
    session.add(a5)

    # ── Analysis 6: PENDING (in-flight, for loading-state testing) ────────────
    a6 = Analysis(
        user_id=users[2].id,
        input_type=InputType.TEXT,
        original_input="New research suggests coffee cures Alzheimer's disease.",
        status=AnalysisStatus.PENDING,
        created_at=ts(hours_ago=0.1),
        updated_at=ts(hours_ago=0.1),
    )
    session.add(a6)

    session.flush()
    print(f"  ✓ Seeded 6 analyses with claims, predictions, evidence, and model runs")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  DEV SEED — Fake News Detection")
    print("  ⚠  FOR DEVELOPMENT/TESTING ONLY — NOT FOR PRODUCTION")
    print("=" * 60)

    # Safety guard: refuse to run against a database that looks production-like
    db_name = settings.postgres_db
    if "prod" in db_name.lower() or "live" in db_name.lower():
        print(f"\n  ✗ ABORTED: database name '{db_name}' looks like a production database.")
        print("    Seed data must not be inserted into production.")
        sys.exit(1)

    engine = create_engine(SYNC_URL, echo=False)

    with Session(engine) as session:
        try:
            print("\nSeeding...")
            users = seed_users(session)
            model_versions = seed_model_versions(session)
            seed_analyses(session, users, model_versions)
            session.commit()
            print("\n  ✓ All seed data committed successfully.")
        except Exception as exc:
            session.rollback()
            print(f"\n  ✗ Seed failed — rolled back. Error: {exc}")
            raise

    print("\nDev credentials:")
    print("  admin    → dev.admin@example.test   / DevAdmin@1234")
    print("  analyst  → analyst.test@example.test / Analyst@5678")
    print("  user     → regular.user@example.test / User@9999")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
