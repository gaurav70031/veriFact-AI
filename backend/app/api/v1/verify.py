"""
Standalone verification endpoint.

POST /api/v1/verify
  Verify a single claim, article text, headline, or URL without running
  the full ML pipeline. Returns an evidence-based assessment only.

POST /api/v1/verify/providers
  List all configured evidence providers and their status.

This endpoint is intentionally separate from POST /analyze/* because:
  1. It is faster — no ML inference, only evidence retrieval + comparison.
  2. It is usable without trained model weights.
  3. It makes the ML/evidence separation explicit in the API.

Pipeline
--------
  input (text | url | claim)
    → [url] article extraction via trafilatura
    → claim extraction
    → concurrent evidence retrieval (all providers)
    → evidence comparison (lexical + optional semantic)
    → verdict aggregation
    → response

Critical distinctions enforced
-------------------------------
* ML predictions are NOT part of this response.
* INSUFFICIENT_EVIDENCE ≠ false.
* No sources are fabricated — only real retrieval results are returned.
* Provider failures are surfaced, never silently swallowed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.analysis.claim_extractor import extract_claims
from app.analysis.evidence_comparator import compare as compare_evidence
from app.analysis.verdict_engine import (
    assess_claim, aggregate_verdict,
    DEFAULT_CONFIG, ClaimVerdictResult,
)
from app.evidence.evidence_service import retrieve_evidence
from app.evidence.query_extractor import extract_queries
from app.models.evidence_source import EvidenceRelationship

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/verify", tags=["Verification"])


# ── Request schema ────────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """
    Verification request — one of text/url/claim must be provided.
    """
    text:    Optional[str] = Field(
        None,
        min_length=20,
        max_length=50_000,
        description="Full article text or headline to verify.",
    )
    url:     Optional[str] = Field(
        None,
        max_length=2_000,
        description="Public article URL to extract and verify.",
    )
    claim:   Optional[str] = Field(
        None,
        min_length=10,
        max_length=500,
        description="Single factual claim to verify.",
    )
    max_results: int = Field(
        default=10, ge=1, le=30,
        description="Max evidence items per claim.",
    )
    from_days:   int = Field(
        default=90, ge=1, le=365,
        description="Restrict evidence to last N days.",
    )
    max_claims:  int = Field(
        default=3, ge=1, le=10,
        description="Max claims to extract from text.",
    )

    @field_validator("url")
    @classmethod
    def url_scheme(cls, v: Optional[str]) -> Optional[str]:
        if v and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    def has_input(self) -> bool:
        return bool(self.text or self.url or self.claim)


# ── Response schemas ──────────────────────────────────────────────────────────

class EvidenceSourceOut(BaseModel):
    source_name:           str
    title:                 str
    url:                   str
    snippet:               Optional[str]
    source_type:           str
    provider_name:         str
    published_at:          Optional[str]
    relevance_score:       float
    relationship_to_claim: str
    comparison_score:      float
    trust_tier:            int             # 0=unknown, 1=official, 2=research, 3-4=news


class ClaimVerificationOut(BaseModel):
    claim_text:            str
    position:              int
    evidence_verdict:      str
    evidence_explanation:  str
    supporting_count:      int
    contradicting_count:   int
    inconclusive_count:    int
    not_relevant_count:    int
    total_evidence:        int
    evidence_sources:      list[EvidenceSourceOut]
    evidence_limitations:  list[str]


class VerificationResponse(BaseModel):
    input_type:            str             # "text" | "url" | "claim"
    article_title:         Optional[str]
    source_url:            Optional[str]
    overall_verdict:       str
    overall_explanation:   str
    claims_checked:        list[ClaimVerificationOut]
    total_evidence:        int
    supporting:            int
    contradicting:         int
    inconclusive:          int
    providers_used:        list[str]
    providers_failed:      list[str]
    processing_time_ms:    int
    limitations:           list[str]


class ProviderStatusOut(BaseModel):
    name:          str
    configured:    bool
    description:   str
    requires_key:  bool
    env_var:       Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trust_tier_from_provider(provider_name: str) -> int:
    for tier in range(1, 5):
        if f":tier{tier}" in provider_name:
            return tier
    return 0


async def _run_claim_verification(
    claim_text:  str,
    position:    int,
    max_results: int,
    from_days:   int,
) -> tuple[ClaimVerdictResult, list[tuple]]:
    """
    Retrieve evidence and compare it for a single claim.
    Returns (ClaimVerdictResult, [(ev_item, ComparisonResult), ...])
    """
    ev_result = await _safe_retrieve(claim_text, max_results, from_days)
    qs = extract_queries(claim_text)
    keywords = qs.keywords

    relationship_counts: dict[EvidenceRelationship, int] = {}
    ev_pairs = []

    if ev_result and ev_result.items:
        for ev_item in ev_result.items:
            snippet = " ".join(filter(None, [ev_item.title, ev_item.description or ""]))
            comp = compare_evidence(
                claim_text=claim_text,
                snippet=snippet,
                claim_keywords=keywords,
            )
            relationship_counts[comp.relationship] = (
                relationship_counts.get(comp.relationship, 0) + 1
            )
            ev_pairs.append((ev_item, comp))

    total_ev = len(ev_result.items) if ev_result else 0
    claim_result = assess_claim(
        claim_text=claim_text,
        position=position,
        relationship_counts=relationship_counts,
        total_evidence=total_ev,
        ml_label=None,   # no ML in verification-only mode
        config=DEFAULT_CONFIG,
    )
    return claim_result, ev_pairs


async def _safe_retrieve(claim: str, max_results: int, from_days: int):
    try:
        return await retrieve_evidence(claim, max_results=max_results, from_days=from_days)
    except Exception as exc:
        logger.warning("Evidence retrieval failed for claim %r: %s", claim[:60], exc)
        return None


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify a claim or article",
    description=(
        "Verify a factual claim, article text, headline, or URL using "
        "real external evidence sources.\n\n"
        "**This endpoint does not use ML models.** "
        "It only retrieves and compares evidence.\n\n"
        "**Verdict values:**\n"
        "- `LIKELY_CREDIBLE` — multiple sources support the claim\n"
        "- `CONTRADICTED` — multiple sources contradict the claim\n"
        "- `LIKELY_MISLEADING` — contradicting evidence found\n"
        "- `UNVERIFIED` — mixed or inconclusive evidence\n"
        "- `INSUFFICIENT_EVIDENCE` — not enough sources found\n\n"
        "`INSUFFICIENT_EVIDENCE` does **not** mean the claim is false. "
        "Absence of evidence is not evidence of absence.\n\n"
        "No sources are fabricated. All evidence comes from real retrieval."
    ),
)
async def verify(body: VerifyRequest) -> VerificationResponse:
    if not body.has_input():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one of: text, url, claim.",
        )

    t0 = time.perf_counter()

    input_type    = "claim" if body.claim else ("url" if body.url else "text")
    article_title: Optional[str] = None
    source_url:    Optional[str] = body.url
    full_text:     str           = ""

    # ── URL extraction ────────────────────────────────────────────────────────
    if body.url:
        try:
            from app.extraction.article_extractor import extract_article
            article = await extract_article(body.url)
            full_text     = f"{article.title or ''} {article.text}".strip()
            article_title = article.title
            source_url    = body.url
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not extract article from URL: {exc}",
            )
    elif body.claim:
        full_text = body.claim.strip()
    else:
        full_text = body.text.strip()

    # ── Claim extraction ──────────────────────────────────────────────────────
    if body.claim:
        from app.analysis.claim_extractor import ExtractedClaim
        extracted = [ExtractedClaim(text=body.claim.strip(), position=1, score=5.0)]
    else:
        extracted = extract_claims(full_text, max_claims=body.max_claims)

    if not extracted:
        extracted_fallback_text = full_text[:300]
        from app.analysis.claim_extractor import ExtractedClaim
        extracted = [ExtractedClaim(text=extracted_fallback_text, position=1, score=1.0)]

    # ── Concurrent evidence retrieval + comparison ────────────────────────────
    tasks = [
        _run_claim_verification(
            ec.text, ec.position, body.max_results, body.from_days
        )
        for ec in extracted
    ]
    results = await asyncio.gather(*tasks)

    claim_results      = [r[0] for r in results]
    ev_pairs_per_claim = [r[1] for r in results]

    # ── Aggregate providers across all claims ─────────────────────────────────
    all_providers_used:   list[str] = []
    all_providers_failed: list[str] = []

    # Re-run provider metadata extraction from results
    for ev_pairs in ev_pairs_per_claim:
        for ev_item, _ in ev_pairs:
            if ev_item.provider_name not in all_providers_used:
                all_providers_used.append(ev_item.provider_name)

    # ── Aggregate verdict ─────────────────────────────────────────────────────
    verdict = aggregate_verdict(
        claim_results=claim_results,
        ml_ensemble_label=None,
        ml_ensemble_confidence=None,
        providers_used=all_providers_used,
        providers_failed=all_providers_failed,
    )

    # ── Build response ────────────────────────────────────────────────────────
    claims_out: list[ClaimVerificationOut] = []
    for ec, cvr, ev_pairs in zip(extracted, claim_results, ev_pairs_per_claim):
        sources_out: list[EvidenceSourceOut] = []
        for ev_item, comp in sorted(ev_pairs, key=lambda x: x[0].relevance_score, reverse=True):
            sources_out.append(EvidenceSourceOut(
                source_name=ev_item.source_name,
                title=ev_item.title,
                url=ev_item.url,
                snippet=ev_item.description,
                source_type=ev_item.source_type.value,
                provider_name=ev_item.provider_name,
                published_at=(
                    ev_item.published_at.isoformat() if ev_item.published_at else None
                ),
                relevance_score=ev_item.relevance_score,
                relationship_to_claim=comp.relationship.value,
                comparison_score=comp.comparison_score,
                trust_tier=_trust_tier_from_provider(ev_item.provider_name),
            ))

        claims_out.append(ClaimVerificationOut(
            claim_text=ec.text,
            position=ec.position,
            evidence_verdict=cvr.evidence_assessment.value,
            evidence_explanation=cvr.evidence_explanation,
            supporting_count=cvr.supporting_count,
            contradicting_count=cvr.contradicting_count,
            inconclusive_count=cvr.inconclusive_count,
            not_relevant_count=cvr.not_relevant_count,
            total_evidence=cvr.total_evidence,
            evidence_sources=sources_out,
            evidence_limitations=cvr.evidence_limitations,
        ))

    processing_ms = int((time.perf_counter() - t0) * 1000)

    limitations = list(dict.fromkeys(verdict.evidence_limitations + [
        "Evidence-only verification does not use ML models.",
        "Absence of evidence is not proof of falsehood.",
        "Source relevance scores are computed from text overlap, not editorial review.",
        "Only publicly accessible articles are retrieved.",
    ]))

    return VerificationResponse(
        input_type=input_type,
        article_title=article_title,
        source_url=source_url,
        overall_verdict=verdict.overall_evidence_assessment.value,
        overall_explanation=verdict.overall_explanation,
        claims_checked=claims_out,
        total_evidence=verdict.total_evidence,
        supporting=verdict.total_supporting,
        contradicting=verdict.total_contradicting,
        inconclusive=verdict.total_inconclusive,
        providers_used=all_providers_used,
        providers_failed=all_providers_failed,
        processing_time_ms=processing_ms,
        limitations=limitations,
    )


# ── Provider status endpoint ──────────────────────────────────────────────────

@router.get(
    "/providers",
    response_model=list[ProviderStatusOut],
    summary="Evidence provider configuration status",
    description=(
        "Returns the list of all evidence providers and whether they are "
        "configured (API key present in environment).\n\n"
        "Useful for diagnosing why evidence retrieval returns no results.\n\n"
        "**API keys are never returned** — only a boolean `configured` flag."
    ),
)
async def provider_status() -> list[ProviderStatusOut]:
    from app.evidence.evidence_service import _build_default_providers

    providers = _build_default_providers()
    return [
        ProviderStatusOut(
            name=p.name,
            configured=p.is_configured,
            description=_provider_description(p.name),
            requires_key=_provider_requires_key(p.name),
            env_var=_provider_env_var(p.name),
        )
        for p in providers
    ]


def _provider_description(name: str) -> str:
    return {
        "tavily":     "Tavily AI Search — extracts content, tier-aware source scoring",
        "newsapi":    "NewsAPI — 70,000+ news sources, relevancy-sorted",
        "gnews":      "GNews — global news aggregation, 60,000+ sources",
        "rss_feed":   "RSS feeds — real-time curated feeds, no API key required",
        "web_search": "Web Search — SerpAPI / Brave Search fallback",
    }.get(name, "Evidence provider")


def _provider_requires_key(name: str) -> bool:
    return name != "rss_feed"


def _provider_env_var(name: str) -> Optional[str]:
    return {
        "tavily":     "TAVILY_API_KEY",
        "newsapi":    "NEWSAPI_KEY",
        "gnews":      "GNEWS_API_KEY",
        "web_search": "SERPAPI_KEY or BRAVE_SEARCH_KEY",
    }.get(name)
