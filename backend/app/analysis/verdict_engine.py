"""
Verdict engine.

Aggregates per-claim evidence relationships into a final EvidenceAssessment.

Design principles
-----------------
* Transparent: every decision maps to a documented rule.
* Conservative: biases toward UNVERIFIED / INSUFFICIENT_EVIDENCE rather
  than making strong claims when evidence is weak.
* Configurable: all thresholds live in VerdictConfig and can be overridden
  per-deployment without changing logic.
* Separation of concerns: NEVER treats model confidence as factual truth.
  ML predictions and evidence assessments are produced independently and
  returned together so the caller can display both.

Assessment rules (applied in order, first match wins)
------------------------------------------------------
1. INSUFFICIENT_EVIDENCE
   — no evidence found at all, OR
   — all items classified as NOT_RELEVANT

2. CONTRADICTED
   — ≥ CONTRADICT_MIN contradicting items AND
   — contradicting count ≥ supporting count

3. LIKELY_MISLEADING
   — ≥ MISLEAD_CONTRADICT_MIN contradicting items AND
   — ML ensemble says FAKE with confidence ≥ MISLEAD_ML_THRESHOLD

4. LIKELY_CREDIBLE
   — ≥ SUPPORT_MIN supporting items AND
   — contradicting count = 0

5. UNVERIFIED
   — evidence exists but no clear direction
   — catch-all for everything that doesn't match rules 1–4

These rules are intentionally conservative.  A claim needs multiple
independent supporting sources to reach LIKELY_CREDIBLE.  A single
contradicting article is not enough to reach CONTRADICTED.

Evidence limitations
--------------------
The engine always reports evidence limitations so users understand the
boundaries of the assessment:
  * How many providers were searched
  * How many items were found
  * Whether any providers failed
  * Whether the assessment is based on only one source
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.models.claim import EvidenceAssessment
from app.models.evidence_source import EvidenceRelationship


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class VerdictConfig:
    """
    Configurable thresholds for the verdict engine.

    All counts refer to the number of EvidenceSource items with the
    given relationship classification for a single claim.
    """

    # Minimum supporting items needed to reach LIKELY_CREDIBLE
    support_min:              int   = 2

    # Minimum contradicting items needed to reach CONTRADICTED
    contradict_min:           int   = 2

    # Minimum contradicting items + ML confidence needed for LIKELY_MISLEADING
    mislead_contradict_min:   int   = 1
    mislead_ml_threshold:     float = 0.65   # model fake_probability

    # If total meaningful evidence (not NOT_RELEVANT) is below this, use
    # INSUFFICIENT_EVIDENCE even if some items exist
    min_meaningful_evidence:  int   = 1


# Default config — used when none is passed explicitly
DEFAULT_CONFIG = VerdictConfig()


# ── Per-claim result ──────────────────────────────────────────────────────────

@dataclass
class ClaimVerdictResult:
    """
    Evidence-based assessment for one claim.

    Intentionally keeps ML prediction separate from evidence assessment.
    """
    # The claim text
    claim_text: str
    position:   int

    # Evidence counts (from comparator output)
    supporting_count:    int = 0
    contradicting_count: int = 0
    inconclusive_count:  int = 0
    not_relevant_count:  int = 0
    total_evidence:      int = 0

    # Evidence assessment (evidence-only — does NOT incorporate ML)
    evidence_assessment: EvidenceAssessment = EvidenceAssessment.INSUFFICIENT_EVIDENCE
    evidence_explanation: str = ""
    evidence_limitations: list[str] = field(default_factory=list)

    # ML prediction (stored separately for transparency)
    ml_label:       Optional[str]   = None   # "FAKE" | "REAL"
    ml_confidence:  Optional[float] = None
    ml_fake_prob:   Optional[float] = None
    ml_real_prob:   Optional[float] = None

    # Whether evidence and ML predictions agree
    assessment_conflict: bool = False


# ── Final analysis result ─────────────────────────────────────────────────────

@dataclass
class AnalysisVerdict:
    """
    Complete verdict for the full analysis (one or more claims).

    Contains both ML-derived and evidence-derived assessments so the
    frontend can display them separately and explain the difference.
    """
    # Per-claim results
    claim_results: list[ClaimVerdictResult] = field(default_factory=list)

    # Aggregate ML ensemble prediction (unchanged from existing pipeline)
    ml_ensemble_label:      Optional[str]   = None
    ml_ensemble_confidence: Optional[float] = None

    # Overall evidence-based assessment (aggregated from claim_results)
    overall_evidence_assessment: EvidenceAssessment = EvidenceAssessment.INSUFFICIENT_EVIDENCE
    overall_explanation:         str = ""

    # Evidence statistics
    total_supporting:    int = 0
    total_contradicting: int = 0
    total_inconclusive:  int = 0
    total_not_relevant:  int = 0
    total_evidence:      int = 0
    providers_used:      list[str] = field(default_factory=list)
    providers_failed:    list[str] = field(default_factory=list)

    # Evidence limitations — always reported for transparency
    evidence_limitations: list[str] = field(default_factory=list)


# ── Per-claim verdict function ────────────────────────────────────────────────

def assess_claim(
    claim_text:          str,
    position:            int,
    relationship_counts: dict[EvidenceRelationship, int],
    total_evidence:      int,
    ml_label:            Optional[str]   = None,
    ml_confidence:       Optional[float] = None,
    ml_fake_prob:        Optional[float] = None,
    ml_real_prob:        Optional[float] = None,
    config:              VerdictConfig   = DEFAULT_CONFIG,
) -> ClaimVerdictResult:
    """
    Produce an EvidenceAssessment for one claim given evidence relationship counts.

    Parameters
    ----------
    claim_text          : The claim text.
    position            : Ordinal position (1-based).
    relationship_counts : {EvidenceRelationship: count} from comparator.
    total_evidence      : Total evidence items found (before classification).
    ml_label            : ML ensemble label ("FAKE"/"REAL"/None).
    ml_confidence       : ML ensemble confidence (0–1).
    ml_fake_prob        : ML fake probability.
    ml_real_prob        : ML real probability.
    config              : VerdictConfig thresholds.

    Returns
    -------
    ClaimVerdictResult with assessment and explanation.
    """
    sup   = relationship_counts.get(EvidenceRelationship.SUPPORTING,    0)
    con   = relationship_counts.get(EvidenceRelationship.CONTRADICTING,  0)
    inc   = relationship_counts.get(EvidenceRelationship.INCONCLUSIVE,   0)
    norel = relationship_counts.get(EvidenceRelationship.NOT_RELEVANT,   0)
    meaningful = sup + con + inc

    result = ClaimVerdictResult(
        claim_text=claim_text,
        position=position,
        supporting_count=sup,
        contradicting_count=con,
        inconclusive_count=inc,
        not_relevant_count=norel,
        total_evidence=total_evidence,
        ml_label=ml_label,
        ml_confidence=ml_confidence,
        ml_fake_prob=ml_fake_prob,
        ml_real_prob=ml_real_prob,
    )

    # Build limitations list
    limitations: list[str] = []
    if total_evidence == 0:
        limitations.append("No evidence sources were retrieved for this claim.")
    elif meaningful == 0:
        limitations.append(
            f"All {norel} retrieved source(s) had insufficient topic overlap."
        )
    if sup == 1:
        limitations.append("Only one supporting source found — limited corroboration.")
    if con == 1 and sup > con:
        limitations.append("One contradicting source found — may be outlier.")
    result.evidence_limitations = limitations

    # ── Rule 1: INSUFFICIENT_EVIDENCE ────────────────────────────────────────
    if meaningful < config.min_meaningful_evidence:
        result.evidence_assessment = EvidenceAssessment.INSUFFICIENT_EVIDENCE
        if total_evidence == 0:
            result.evidence_explanation = (
                "No evidence was found for this claim. "
                "This does NOT mean the claim is false — the absence of "
                "search results cannot be interpreted as disproof."
            )
        else:
            result.evidence_explanation = (
                f"{total_evidence} source(s) retrieved but none had sufficient "
                "topic overlap to draw a conclusion."
            )
        _check_conflict(result)
        return result

    # ── Rule 2: CONTRADICTED ─────────────────────────────────────────────────
    if con >= config.contradict_min and con >= sup:
        result.evidence_assessment = EvidenceAssessment.CONTRADICTED
        result.evidence_explanation = (
            f"{con} source(s) contradict this claim vs {sup} supporting. "
            "Multiple independent sources dispute the claim's accuracy."
        )
        _check_conflict(result)
        return result

    # ── Rule 3: LIKELY_MISLEADING ─────────────────────────────────────────────
    if (
        con >= config.mislead_contradict_min
        and ml_label == "FAKE"
        and (ml_fake_prob or 0.0) >= config.mislead_ml_threshold
    ):
        result.evidence_assessment = EvidenceAssessment.LIKELY_MISLEADING
        result.evidence_explanation = (
            f"ML model predicts FAKE ({(ml_fake_prob or 0):.0%} probability) "
            f"and {con} evidence source(s) contradict the claim. "
            f"Combined signal suggests the claim may be misleading. "
            f"({sup} supporting, {inc} inconclusive sources also found.)"
        )
        _check_conflict(result)
        return result

    # ── Rule 4: LIKELY_CREDIBLE ───────────────────────────────────────────────
    if sup >= config.support_min and con == 0:
        result.evidence_assessment = EvidenceAssessment.LIKELY_CREDIBLE
        result.evidence_explanation = (
            f"{sup} source(s) support this claim with no contradicting sources. "
            "The claim appears consistent with available evidence."
        )
        _check_conflict(result)
        return result

    # ── Rule 5: UNVERIFIED (catch-all) ────────────────────────────────────────
    result.evidence_assessment = EvidenceAssessment.UNVERIFIED
    parts = []
    if sup:
        parts.append(f"{sup} supporting")
    if con:
        parts.append(f"{con} contradicting")
    if inc:
        parts.append(f"{inc} inconclusive")
    result.evidence_explanation = (
        f"Evidence found ({', '.join(parts)}) but no clear consensus. "
        "Cannot reach a confident assessment from available sources."
    )
    _check_conflict(result)
    return result


def _check_conflict(result: ClaimVerdictResult) -> None:
    """Flag when ML prediction and evidence assessment disagree significantly."""
    if result.ml_label is None:
        return
    ev = result.evidence_assessment
    ml = result.ml_label
    if ml == "FAKE" and ev == EvidenceAssessment.LIKELY_CREDIBLE:
        result.assessment_conflict = True
    elif ml == "REAL" and ev in (
        EvidenceAssessment.CONTRADICTED,
        EvidenceAssessment.LIKELY_MISLEADING,
    ):
        result.assessment_conflict = True


# ── Overall aggregation ────────────────────────────────────────────────────────

def aggregate_verdict(
    claim_results:          list[ClaimVerdictResult],
    ml_ensemble_label:      Optional[str]   = None,
    ml_ensemble_confidence: Optional[float] = None,
    providers_used:         Optional[list[str]] = None,
    providers_failed:       Optional[list[str]] = None,
) -> AnalysisVerdict:
    """
    Aggregate individual claim verdicts into a single AnalysisVerdict.

    Aggregation strategy
    --------------------
    The "worst" evidence assessment wins (most concerning result bubbles up):
      CONTRADICTED > LIKELY_MISLEADING > INSUFFICIENT_EVIDENCE >
      UNVERIFIED > LIKELY_CREDIBLE

    This is intentionally conservative — if any claim is CONTRADICTED,
    the overall result is CONTRADICTED.

    Parameters
    ----------
    claim_results           : List of per-claim results from assess_claim().
    ml_ensemble_label       : Overall ML ensemble label.
    ml_ensemble_confidence  : ML ensemble confidence.
    providers_used          : Names of providers that returned results.
    providers_failed        : Names of providers that errored.

    Returns
    -------
    AnalysisVerdict with overall assessment and full details.
    """
    # Assessment severity order (lower index = more severe)
    _SEVERITY: dict[EvidenceAssessment, int] = {
        EvidenceAssessment.CONTRADICTED:          0,
        EvidenceAssessment.LIKELY_MISLEADING:     1,
        EvidenceAssessment.INSUFFICIENT_EVIDENCE: 2,
        EvidenceAssessment.UNVERIFIED:            3,
        EvidenceAssessment.LIKELY_CREDIBLE:       4,
    }

    verdict = AnalysisVerdict(
        claim_results=claim_results,
        ml_ensemble_label=ml_ensemble_label,
        ml_ensemble_confidence=ml_ensemble_confidence,
        providers_used=providers_used or [],
        providers_failed=providers_failed or [],
    )

    if not claim_results:
        verdict.overall_evidence_assessment = EvidenceAssessment.INSUFFICIENT_EVIDENCE
        verdict.overall_explanation = "No claims were extracted or assessed."
        return verdict

    # Sum evidence counts across all claims
    for r in claim_results:
        verdict.total_supporting    += r.supporting_count
        verdict.total_contradicting += r.contradicting_count
        verdict.total_inconclusive  += r.inconclusive_count
        verdict.total_not_relevant  += r.not_relevant_count
        verdict.total_evidence      += r.total_evidence
        verdict.evidence_limitations.extend(r.evidence_limitations)

    # Pick most severe assessment
    most_severe = min(
        claim_results,
        key=lambda r: _SEVERITY.get(r.evidence_assessment, 99),
    )
    verdict.overall_evidence_assessment = most_severe.evidence_assessment

    # Build overall explanation
    n = len(claim_results)
    verdict.overall_explanation = _build_overall_explanation(verdict, n)

    return verdict


def _build_overall_explanation(v: AnalysisVerdict, n_claims: int) -> str:
    """Compose a human-readable overall explanation."""
    assessment = v.overall_evidence_assessment
    parts: list[str] = []

    parts.append(f"Analysed {n_claims} claim(s) from the submitted text.")

    if v.total_evidence == 0:
        parts.append(
            "No evidence was found. This does not indicate the content is false "
            "— evidence absence is not proof of falsehood."
        )
        return " ".join(parts)

    parts.append(
        f"Evidence: {v.total_supporting} supporting, "
        f"{v.total_contradicting} contradicting, "
        f"{v.total_inconclusive} inconclusive, "
        f"{v.total_not_relevant} not relevant "
        f"(from {len(v.providers_used)} provider(s))."
    )

    if assessment == EvidenceAssessment.CONTRADICTED:
        parts.append(
            "Multiple sources contradict one or more claims. "
            "The content may contain inaccurate assertions."
        )
    elif assessment == EvidenceAssessment.LIKELY_MISLEADING:
        parts.append(
            "ML models predict misinformation and contradicting evidence was found. "
            "The content shows indicators of being misleading."
        )
    elif assessment == EvidenceAssessment.LIKELY_CREDIBLE:
        parts.append(
            "Multiple supporting sources found with no contradictions. "
            "Claims appear consistent with available evidence."
        )
    elif assessment == EvidenceAssessment.UNVERIFIED:
        parts.append(
            "Evidence was found but results are mixed or inconclusive. "
            "A definitive assessment cannot be made from available sources."
        )
    else:  # INSUFFICIENT_EVIDENCE
        parts.append(
            "Insufficient evidence to make an assessment. "
            "This does not imply the content is false."
        )

    if v.providers_failed:
        parts.append(
            f"Note: {len(v.providers_failed)} evidence provider(s) were unavailable."
        )

    return " ".join(parts)
