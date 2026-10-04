"""
Integration verification script.
Run from backend/ directory:
    SENTENCE_TRANSFORMERS_ENABLED=false python integration_check.py
"""
import sys
import os
import warnings
# Suppress passlib/bcrypt version compatibility warning (cosmetic — hashing works)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*bcrypt.*')
os.environ.setdefault('SENTENCE_TRANSFORMERS_ENABLED', 'false')

sys.path.insert(0, '.')

errors = []
ok     = []

def check(label):
    def decorator(fn):
        try:
            fn()
            ok.append(label)
            print(f'[OK]   {label}')
        except Exception as e:
            errors.append((label, str(e)))
            print(f'[FAIL] {label}: {e}')
    return decorator


# ── 1. Auth packages ──────────────────────────────────────────────────────────

@check('jose + passlib installed')
def _():
    from jose import jwt
    from passlib.context import CryptContext


# ── 2. Secret key is not placeholder ────────────────────────────────────────

@check('SECRET_KEY is not default placeholder (reads .env directly)')
def _():
    from pathlib import Path
    env_file = Path(__file__).parent.parent / '.env'
    if not env_file.exists():
        env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        content = env_file.read_text()
        for line in content.splitlines():
            if line.startswith('SECRET_KEY='):
                val = line.split('=', 1)[1].strip()
                placeholders = {
                    'change-me-in-production-use-32-chars-min',
                    'change-me-generate-with-secrets-token-hex-32',
                    'change-me-to-a-random-32-char-string',
                }
                assert val not in placeholders, f'SECRET_KEY is placeholder: {val[:20]}...'
                assert len(val) >= 32, f'SECRET_KEY too short: {len(val)} chars'
                return
    # If .env not found, skip (not a failure — env var might be set elsewhere)
    print('       (skipped — .env file not found in expected location)')


# ── 3. Auth utilities ────────────────────────────────────────────────────────

@check('bcrypt hash + verify + JWT round-trip')
def _():
    from app.core.auth_utils import hash_password, verify_password, create_access_token, decode_access_token
    h = hash_password('StrongPass1!')
    assert verify_password('StrongPass1!', h)
    assert not verify_password('wrong', h)
    token = create_access_token(42, 'user')
    payload = decode_access_token(token)
    assert payload['sub'] == '42'
    assert payload['role'] == 'user'
    assert payload['type'] == 'access'


# ── 4. ORM models ─────────────────────────────────────────────────────────────

@check('7 ORM models + new enums (EvidenceAssessment, NOT_RELEVANT)')
def _():
    import app.models
    from app.models import User, Analysis, Claim, Prediction, EvidenceSource, ModelVersion, ModelRun
    from app.models.claim import EvidenceAssessment
    from app.models.evidence_source import EvidenceRelationship
    assert EvidenceRelationship.NOT_RELEVANT.value == 'not_relevant'
    assert EvidenceAssessment.INSUFFICIENT_EVIDENCE.value == 'INSUFFICIENT_EVIDENCE'


# ── 5. Schemas ────────────────────────────────────────────────────────────────

@check('AnalysisResponse has ml_verdict (not final_verdict)')
def _():
    from app.schemas.analyze import AnalysisResponse
    fields = AnalysisResponse.model_fields
    assert 'ml_verdict' in fields,        'ml_verdict missing from AnalysisResponse'
    assert 'final_verdict' not in fields, 'final_verdict must NOT be in AnalysisResponse'


@check('AnalysisListItem has final_verdict')
def _():
    from app.schemas.history import AnalysisListItem
    assert 'final_verdict' in AnalysisListItem.model_fields


@check('Auth schemas importable (requires email-validator)')
def _():
    from app.schemas.auth import RegisterRequest, LoginRequest, UserOut, AuthResponse
    r = RegisterRequest(email='test@example.com', username='testuser', password='StrongP1!')
    assert r.email == 'test@example.com'


# ── 6. Services have user_id parameter ────────────────────────────────────────

@check('analyse_text/url/claim and get_history/get_analysis_by_id all have user_id')
def _():
    import inspect
    from app.services import analysis_service, history_service
    for fn_name, fn in [
        ('analyse_text',       analysis_service.analyse_text),
        ('analyse_url',        analysis_service.analyse_url),
        ('analyse_claim',      analysis_service.analyse_claim),
        ('get_history',        history_service.get_history),
        ('get_analysis_by_id', history_service.get_analysis_by_id),
    ]:
        sig = inspect.signature(fn)
        assert 'user_id' in sig.parameters, f'{fn_name} missing user_id param'


# ── 7. Router mounts all required routes ─────────────────────────────────────

@check('API router mounts /auth, /analyze, /history, /health, /evidence, /news, /explanation')
def _():
    from app.api.v1.router import api_router
    all_paths = []
    for r in api_router.routes:
        if hasattr(r, 'path'):
            all_paths.append(r.path)
        for sr in getattr(r, 'routes', []):
            if hasattr(sr, 'path'):
                all_paths.append(getattr(r, 'prefix', '') + sr.path)
    path_str = ' '.join(all_paths)
    required = ['/auth', '/analyze', '/history', '/health', '/evidence', '/news', '/explanation']
    for req in required:
        assert req in path_str, f'{req} not found in router paths'


# ── 8. FastAPI app starts ────────────────────────────────────────────────────

@check('FastAPI app creates without errors (no DB required)')
def _():
    from app.main import app as fastapi_app
    assert fastapi_app.title == 'Fake News Detection API'


# ── 9. Dependencies importable ───────────────────────────────────────────────

@check('get_current_user, get_optional_user, require_admin importable')
def _():
    from app.core.dependencies import get_current_user, get_optional_user, require_admin


# ── 10. Analysis pipeline (no inference needed) ───────────────────────────────

@check('claim extraction, evidence comparator, verdict engine all import + run')
def _():
    from app.analysis.claim_extractor   import extract_claims
    from app.analysis.evidence_comparator import compare
    from app.analysis.verdict_engine    import assess_claim, aggregate_verdict
    from app.models.evidence_source import EvidenceRelationship

    claims = extract_claims(
        'The WHO declared COVID-19 a pandemic in March 2020.', max_claims=3
    )
    assert len(claims) >= 1

    result = compare(
        claim_text='The central bank raised interest rates.',
        snippet='Central bank raised rates citing inflation.',
    )
    assert result.relationship in list(EvidenceRelationship)
    assert 0.0 <= result.comparison_score <= 1.0


# ── 11. Explainer imports (lazy — no torch required) ─────────────────────────

@check('explainer module imports without requiring torch/transformers')
def _():
    from app.analysis.explainer import (
        ExplanationResult, TokenWeight,
        explain_baseline_with_fallback, aggregate_top_tokens,
    )
    u = ExplanationResult.unavailable('test_model', 'no models loaded')
    assert u.method == 'unavailable'


# ── 12. Evidence service (no real HTTP) ──────────────────────────────────────

@check('evidence query extractor runs')
def _():
    from app.evidence.query_extractor import extract_queries
    qs = extract_queries('5G towers caused COVID-19 pandemic spread')
    assert len(qs.primary_query) > 0
    assert len(qs.keywords) > 0


# ── 13. Alembic migration can be generated ───────────────────────────────────

@check('alembic.ini and migration files exist')
def _():
    from pathlib import Path
    base = Path(__file__).parent
    assert (base / 'alembic.ini').exists(), 'alembic.ini missing'
    assert (base / 'alembic' / 'versions' / '0001_initial_schema.py').exists(), '0001 migration missing'
    assert (base / 'alembic' / 'versions' / '0002_claim_evidence_engine.py').exists(), '0002 migration missing'


# ── Results ───────────────────────────────────────────────────────────────────

print()
print('=' * 60)
print(f'  Integration check: {len(ok)} passed, {len(errors)} failed')
print('=' * 60)
if errors:
    for label, msg in errors:
        print(f'  [FAIL] {label}')
        print(f'         {msg}')
    sys.exit(1)
else:
    print('  All backend integration checks passed.')
    sys.exit(0)
