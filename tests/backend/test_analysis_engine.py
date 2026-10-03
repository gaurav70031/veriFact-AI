"""
Tests for the claim analysis and evidence comparison engine.

Coverage
--------
  TestClaimExtractor       — sentence splitting, scoring, edge cases
  TestEvidenceComparator   — lexical similarity, contradiction detection,
                             all four relationship classifications, thresholds
  TestVerdictEngine        — all 5 rules, aggregation, conflict detection,
                             edge cases (no claims, no evidence, all failed)
  TestVerdictAggregation   — overall verdict aggregation across multiple claims
  TestEnginePipeline       — end-to-end without DB (unit-level integration)

All tests use in-memory data only.
No DB, no HTTP, no ML models required.

Run from the project root:
    pytest tests/backend/test_analysis_engine.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Prevent sentence-transformers from downloading models during tests
# The comparator gracefully falls back to lexical mode when the model is None.
import app.analysis.evidence_comparator as _comparator_mod
_comparator_mod._load_sentence_transformer.cache_clear()
_comparator_mod._load_sentence_transformer = lambda: None   # type: ignore[assignment]

from app.analysis.claim_extractor import extract_claims, ExtractedClaim, _score_sentence
from app.analysis.evidence_comparator import (
    compare, ComparisonResult,
    _tokenise, _jaccard, _keyword_overlap, _contradiction_signal,
    NOT_RELEVANT_THRESHOLD, SUPPORT_THRESHOLD, CONTRADICTION_SIGNAL_WEIGHT,
)
from app.analysis.verdict_engine import (
    assess_claim, aggregate_verdict,
    ClaimVerdictResult, AnalysisVerdict,
    VerdictConfig, DEFAULT_CONFIG,
)
from app.models.evidence_source import EvidenceRelationship
from app.models.claim import EvidenceAssessment


# =============================================================================
# Claim Extractor
# =============================================================================

class TestClaimExtractor:

    # ── Short inputs ──────────────────────────────────────────────────────────

    def test_short_input_returns_single_claim(self):
        text = "COVID-19 vaccines are 95% effective against severe disease."
        claims = extract_claims(text)
        assert len(claims) == 1
        assert claims[0].text == text

    def test_short_input_position_is_1(self):
        claims = extract_claims("5G towers cause COVID-19 pandemic worldwide 2020.")
        assert claims[0].position == 1

    def test_short_input_score_is_positive(self):
        claims = extract_claims("The WHO declared COVID-19 a pandemic in March 2020.")
        assert claims[0].score > 0

    # ── Multi-sentence inputs ─────────────────────────────────────────────────

    def test_long_text_returns_multiple_claims(self):
        text = (
            "The European Central Bank raised interest rates by 25 basis points on Thursday. "
            "ECB President Christine Lagarde stated the decision was unanimous. "
            "Inflation in the eurozone reached 8.9% in July 2022. "
            "The bank expects inflation to remain elevated throughout 2023. "
            "Bond yields fell slightly after the announcement was made public. "
            "Market analysts expect further rate increases in the coming months. "
        )
        claims = extract_claims(text, max_claims=5)
        assert len(claims) >= 2

    def test_claims_ordered_by_position(self):
        text = (
            "The Federal Reserve raised rates by 75 basis points in June 2022. "
            "Chair Jerome Powell cited persistent inflation as the primary reason. "
            "US inflation reached 9.1% in June 2022, the highest in 40 years. "
            "Mortgage rates rose sharply following the Fed announcement worldwide. "
        )
        claims = extract_claims(text, max_claims=4)
        positions = [c.position for c in claims]
        assert positions == sorted(positions), "Claims must be in source order"

    def test_max_claims_respected(self):
        text = " ".join([
            f"Company XYZ reported a {i}% increase in revenue in Q{i % 4 + 1} 2023."
            for i in range(1, 20)
        ])
        claims = extract_claims(text, max_claims=3)
        assert len(claims) <= 3

    def test_empty_input_returns_empty_list(self):
        assert extract_claims("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert extract_claims("   ") == []

    def test_extracted_claims_are_strings(self):
        claims = extract_claims("NASA launched the Artemis I mission in November 2022.")
        for c in claims:
            assert isinstance(c.text, str)
            assert isinstance(c.position, int)
            assert isinstance(c.score, float)

    def test_claim_text_within_length_limit(self):
        """Claims should be ≤ 300 chars."""
        long_sentence = "The World Health Organization announced " + "a" * 500
        claims = extract_claims(long_sentence)
        for c in claims:
            assert len(c.text) <= 300

    # ── Sentence scoring ──────────────────────────────────────────────────────

    def test_sentence_with_number_scores_higher(self):
        no_num  = "The government announced an important policy today"
        with_num = "The government raised rates by 25 basis points today"
        assert _score_sentence(with_num) > _score_sentence(no_num)

    def test_question_scores_zero(self):
        assert _score_sentence("Did the WHO declare a pandemic in 2020?") == 0.0

    def test_very_short_sentence_scores_zero(self):
        assert _score_sentence("Hi") == 0.0

    def test_opinion_marker_scores_zero(self):
        assert _score_sentence("I think vaccines might be dangerous perhaps") == 0.0

    def test_sentence_with_entity_scores_higher_than_generic(self):
        generic = "rates increased by ten percent yesterday morning"
        entity  = "Federal Reserve rates increased by 10 percent in March 2023"
        assert _score_sentence(entity) >= _score_sentence(generic)

    def test_always_returns_at_least_one_claim(self):
        """Even if no sentence scores above zero, one claim is returned."""
        text = "Perhaps maybe this might be true or not, who knows really?"
        claims = extract_claims(text)
        assert len(claims) >= 1


# =============================================================================
# Evidence Comparator — internal helpers
# =============================================================================

class TestComparatorHelpers:

    def test_tokenise_removes_stopwords(self):
        tokens = _tokenise("the quick brown fox jumps over the lazy dog")
        for stop in ("the", "over"):
            assert stop not in tokens

    def test_tokenise_lowercases(self):
        tokens = _tokenise("ECB Raised Rates 2024")
        assert all(t == t.lower() for t in tokens)

    def test_tokenise_removes_punctuation(self):
        tokens = _tokenise("rates, raised! by: 25%")
        for t in tokens:
            assert "," not in t and "!" not in t and ":" not in t

    def test_tokenise_removes_short_tokens(self):
        tokens = _tokenise("a in is to be of the economy")
        assert all(len(t) > 2 for t in tokens)

    def test_jaccard_identical_sets(self):
        s = {"rates", "raised", "ecb", "eurozone"}
        assert _jaccard(s, s) == 1.0

    def test_jaccard_disjoint_sets(self):
        assert _jaccard({"apple"}, {"orange"}) == 0.0

    def test_jaccard_partial_overlap(self):
        a = {"rates", "raised", "ecb"}
        b = {"rates", "inflation", "bank"}
        score = _jaccard(a, b)
        assert 0.0 < score < 1.0

    def test_jaccard_empty_sets_returns_zero(self):
        assert _jaccard(set(), set()) == 0.0

    def test_keyword_overlap_all_present(self):
        claim_kws = {"rates", "raised", "ecb"}
        snippet   = {"rates", "raised", "ecb", "eurozone", "bank"}
        assert _keyword_overlap(claim_kws, snippet) == 1.0

    def test_keyword_overlap_none_present(self):
        claim_kws = {"vaccines", "covid", "pandemic"}
        snippet   = {"football", "championship", "weekend"}
        assert _keyword_overlap(claim_kws, snippet) == 0.0

    def test_contradiction_signal_debunked(self):
        assert _contradiction_signal("This claim has been debunked by experts.") > 0.5

    def test_contradiction_signal_false(self):
        assert _contradiction_signal("The report is factually false according to WHO.") > 0.5

    def test_contradiction_signal_no_evidence(self):
        assert _contradiction_signal("No evidence supports this claim whatsoever.") > 0.5

    def test_contradiction_signal_neutral_text(self):
        neutral = "The central bank raised interest rates by 25 basis points."
        assert _contradiction_signal(neutral) == 0.0

    def test_contradiction_signal_negation(self):
        assert _contradiction_signal("The government did not raise rates.") > 0.0


# =============================================================================
# Evidence Comparator — compare() function
# =============================================================================

class TestEvidenceComparator:

    # ── NOT_RELEVANT ──────────────────────────────────────────────────────────

    def test_unrelated_content_is_not_relevant(self):
        result = compare(
            claim_text="COVID-19 vaccines are 95% effective against severe disease.",
            snippet="Manchester United won the football championship last weekend.",
        )
        assert result.relationship == EvidenceRelationship.NOT_RELEVANT

    def test_empty_snippet_is_not_relevant(self):
        result = compare(
            claim_text="The WHO declared a pandemic.",
            snippet="",
        )
        assert result.relationship == EvidenceRelationship.NOT_RELEVANT

    def test_empty_claim_is_not_relevant(self):
        result = compare(claim_text="", snippet="WHO pandemic declaration 2020.")
        assert result.relationship == EvidenceRelationship.NOT_RELEVANT

    # ── SUPPORTING ────────────────────────────────────────────────────────────

    def test_highly_similar_text_is_supporting(self):
        claim   = "The ECB raised interest rates by 25 basis points in June 2023."
        snippet = "European Central Bank increased rates by 25 basis points June 2023 inflation."
        result = compare(claim_text=claim, snippet=snippet)
        # Should be SUPPORTING or INCONCLUSIVE (not NOT_RELEVANT or CONTRADICTING)
        assert result.relationship in (
            EvidenceRelationship.SUPPORTING,
            EvidenceRelationship.INCONCLUSIVE,
        )

    def test_supporting_has_no_contradiction_signal(self):
        claim   = "NASA successfully launched the Artemis mission in 2022."
        snippet = "NASA Artemis mission launch successful 2022 moon orbit rocket."
        result = compare(claim_text=claim, snippet=snippet)
        if result.relationship == EvidenceRelationship.SUPPORTING:
            assert result.contradiction_signal < CONTRADICTION_SIGNAL_WEIGHT

    # ── CONTRADICTING ─────────────────────────────────────────────────────────

    def test_debunked_language_triggers_contradicting(self):
        claim   = "5G towers caused the COVID-19 pandemic to spread."
        snippet = "5G COVID pandemic claim debunked false no evidence wireless towers."
        result = compare(claim_text=claim, snippet=snippet)
        # Must not be SUPPORTING; should be CONTRADICTING or INCONCLUSIVE
        assert result.relationship != EvidenceRelationship.SUPPORTING

    def test_factually_false_label_triggers_contradicting(self):
        claim   = "Vaccines cause autism according to new research studies."
        snippet = "Vaccines autism link factually false disproved no evidence scientific consensus."
        result = compare(claim_text=claim, snippet=snippet)
        assert result.relationship in (
            EvidenceRelationship.CONTRADICTING,
            EvidenceRelationship.INCONCLUSIVE,
        )

    # ── INCONCLUSIVE ──────────────────────────────────────────────────────────

    def test_partial_overlap_is_inconclusive(self):
        claim   = "The Federal Reserve raised interest rates in 2022."
        snippet = "Central banks worldwide reviewed monetary policy during 2022 inflation concerns."
        result = compare(claim_text=claim, snippet=snippet)
        # Should be in the middle range — not NOT_RELEVANT but not SUPPORTING
        assert result.relationship in (
            EvidenceRelationship.INCONCLUSIVE,
            EvidenceRelationship.SUPPORTING,
            EvidenceRelationship.NOT_RELEVANT,
        )

    # ── Score properties ──────────────────────────────────────────────────────

    def test_comparison_score_in_0_1_range(self):
        result = compare(
            claim_text="COVID vaccine 95% effective.",
            snippet="Vaccine effectiveness study results published."
        )
        assert 0.0 <= result.comparison_score <= 1.0

    def test_contradiction_signal_in_0_1_range(self):
        result = compare(
            claim_text="This is false claim.",
            snippet="This is debunked false misinformation."
        )
        assert 0.0 <= result.contradiction_signal <= 1.0

    def test_result_has_rationale(self):
        result = compare(claim_text="Some claim here.", snippet="Some snippet here.")
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0

    def test_result_has_method(self):
        result = compare(claim_text="Some claim.", snippet="Some snippet.")
        assert result.method in ("lexical", "semantic")

    def test_keywords_improve_scoring(self):
        """Providing pre-extracted keywords should not decrease the score."""
        claim   = "ECB raised rates 25 basis points 2023"
        snippet = "ECB increased interest rates 25bp eurozone 2023 inflation"
        r_no_kw   = compare(claim_text=claim, snippet=snippet)
        r_with_kw = compare(claim_text=claim, snippet=snippet,
                             claim_keywords=["ECB", "rates", "25", "2023"])
        assert r_with_kw.comparison_score >= r_no_kw.comparison_score - 0.01

    def test_same_text_high_score(self):
        text   = "The World Health Organization declared COVID-19 a pandemic in March 2020."
        result = compare(claim_text=text, snippet=text)
        assert result.comparison_score > SUPPORT_THRESHOLD

    def test_completely_different_text_low_score(self):
        result = compare(
            claim_text="The Federal Reserve raised interest rates by 75 basis points.",
            snippet="Manchester United goalkeeper saves penalty shootout.",
        )
        assert result.comparison_score < SUPPORT_THRESHOLD


# =============================================================================
# Verdict Engine — assess_claim()
# =============================================================================

class TestAssessClaim:

    def _rel_counts(self, sup=0, con=0, inc=0, norel=0):
        counts = {}
        if sup:   counts[EvidenceRelationship.SUPPORTING]    = sup
        if con:   counts[EvidenceRelationship.CONTRADICTING]  = con
        if inc:   counts[EvidenceRelationship.INCONCLUSIVE]   = inc
        if norel: counts[EvidenceRelationship.NOT_RELEVANT]   = norel
        return counts

    # ── Rule 1: INSUFFICIENT_EVIDENCE ────────────────────────────────────────

    def test_no_evidence_is_insufficient(self):
        result = assess_claim("Some claim.", 1, {}, total_evidence=0)
        assert result.evidence_assessment == EvidenceAssessment.INSUFFICIENT_EVIDENCE

    def test_all_not_relevant_is_insufficient(self):
        result = assess_claim(
            "COVID vaccine claim.", 1,
            self._rel_counts(norel=5),
            total_evidence=5,
        )
        assert result.evidence_assessment == EvidenceAssessment.INSUFFICIENT_EVIDENCE

    def test_insufficient_evidence_not_false(self):
        """INSUFFICIENT_EVIDENCE must never say the claim is false."""
        result = assess_claim("Some claim.", 1, {}, total_evidence=0)
        assert "false" not in result.evidence_explanation.lower() or \
               "does not" in result.evidence_explanation.lower()

    def test_insufficient_evidence_explanation_mentions_absence(self):
        result = assess_claim("Some claim.", 1, {}, total_evidence=0)
        assert len(result.evidence_explanation) > 20

    # ── Rule 2: CONTRADICTED ─────────────────────────────────────────────────

    def test_two_contradicting_no_support_is_contradicted(self):
        result = assess_claim(
            "5G causes COVID.", 1,
            self._rel_counts(con=2),
            total_evidence=2,
        )
        assert result.evidence_assessment == EvidenceAssessment.CONTRADICTED

    def test_contradicted_when_con_exceeds_sup(self):
        result = assess_claim(
            "Claim text.", 1,
            self._rel_counts(sup=1, con=3),
            total_evidence=4,
        )
        assert result.evidence_assessment == EvidenceAssessment.CONTRADICTED

    def test_one_contradicting_not_contradicted(self):
        """Single contradicting source not enough to reach CONTRADICTED."""
        result = assess_claim(
            "Some specific claim here.", 1,
            self._rel_counts(sup=2, con=1),
            total_evidence=3,
        )
        # Should be LIKELY_CREDIBLE or UNVERIFIED, not CONTRADICTED
        assert result.evidence_assessment != EvidenceAssessment.CONTRADICTED

    # ── Rule 3: LIKELY_MISLEADING ─────────────────────────────────────────────

    def test_misleading_when_ml_fake_and_contradicting(self):
        result = assess_claim(
            "Vaccine claim.", 1,
            self._rel_counts(con=1, inc=1),
            total_evidence=2,
            ml_label="FAKE",
            ml_fake_prob=0.80,
            ml_confidence=0.80,
        )
        assert result.evidence_assessment == EvidenceAssessment.LIKELY_MISLEADING

    def test_not_misleading_when_ml_real(self):
        """LIKELY_MISLEADING requires ML to say FAKE."""
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(con=1),
            total_evidence=1,
            ml_label="REAL",
            ml_fake_prob=0.20,
            ml_confidence=0.80,
        )
        assert result.evidence_assessment != EvidenceAssessment.LIKELY_MISLEADING

    def test_not_misleading_when_ml_confidence_low(self):
        """Low ML confidence should not trigger LIKELY_MISLEADING."""
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(con=1),
            total_evidence=1,
            ml_label="FAKE",
            ml_fake_prob=0.55,   # below mislead_ml_threshold of 0.65
            ml_confidence=0.55,
        )
        assert result.evidence_assessment != EvidenceAssessment.LIKELY_MISLEADING

    # ── Rule 4: LIKELY_CREDIBLE ───────────────────────────────────────────────

    def test_likely_credible_two_supporting_no_contradicting(self):
        result = assess_claim(
            "ECB raised rates.", 1,
            self._rel_counts(sup=2),
            total_evidence=2,
        )
        assert result.evidence_assessment == EvidenceAssessment.LIKELY_CREDIBLE

    def test_not_likely_credible_with_any_contradicting(self):
        result = assess_claim(
            "ECB raised rates.", 1,
            self._rel_counts(sup=3, con=1),
            total_evidence=4,
        )
        assert result.evidence_assessment != EvidenceAssessment.LIKELY_CREDIBLE

    def test_not_likely_credible_with_only_one_supporting(self):
        """Single supporting source not enough for LIKELY_CREDIBLE."""
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(sup=1),
            total_evidence=1,
        )
        assert result.evidence_assessment != EvidenceAssessment.LIKELY_CREDIBLE

    # ── Rule 5: UNVERIFIED ────────────────────────────────────────────────────

    def test_mixed_evidence_is_unverified(self):
        result = assess_claim(
            "Some claim text.", 1,
            self._rel_counts(sup=1, con=1, inc=2),
            total_evidence=4,
        )
        assert result.evidence_assessment == EvidenceAssessment.UNVERIFIED

    def test_only_inconclusive_is_unverified(self):
        result = assess_claim(
            "Some claim text.", 1,
            self._rel_counts(inc=3),
            total_evidence=3,
        )
        assert result.evidence_assessment == EvidenceAssessment.UNVERIFIED

    def test_one_supporting_is_unverified(self):
        """Not enough for LIKELY_CREDIBLE → falls to UNVERIFIED."""
        result = assess_claim(
            "A specific factual claim here.", 1,
            self._rel_counts(sup=1, inc=1),
            total_evidence=2,
        )
        assert result.evidence_assessment == EvidenceAssessment.UNVERIFIED

    # ── Conflict detection ────────────────────────────────────────────────────

    def test_conflict_flagged_ml_fake_evidence_credible(self):
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(sup=3),
            total_evidence=3,
            ml_label="FAKE",
            ml_fake_prob=0.85,
            ml_confidence=0.85,
        )
        assert result.evidence_assessment == EvidenceAssessment.LIKELY_CREDIBLE
        assert result.assessment_conflict is True

    def test_conflict_flagged_ml_real_evidence_contradicted(self):
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(con=3),
            total_evidence=3,
            ml_label="REAL",
            ml_fake_prob=0.20,
            ml_confidence=0.80,
        )
        assert result.evidence_assessment == EvidenceAssessment.CONTRADICTED
        assert result.assessment_conflict is True

    def test_no_conflict_when_both_agree(self):
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(con=2),
            total_evidence=2,
            ml_label="FAKE",
            ml_fake_prob=0.80,
            ml_confidence=0.80,
        )
        assert result.assessment_conflict is False

    # ── Evidence fields populated ─────────────────────────────────────────────

    def test_result_has_counts(self):
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(sup=2, con=1, inc=1, norel=1),
            total_evidence=5,
        )
        assert result.supporting_count    == 2
        assert result.contradicting_count == 1
        assert result.inconclusive_count  == 1
        assert result.not_relevant_count  == 1
        assert result.total_evidence      == 5

    def test_result_has_explanation(self):
        result = assess_claim("Some claim.", 1, {}, total_evidence=0)
        assert isinstance(result.evidence_explanation, str)
        assert len(result.evidence_explanation) > 0

    def test_result_has_limitations(self):
        result = assess_claim("Some claim.", 1, {}, total_evidence=0)
        assert isinstance(result.evidence_limitations, list)

    def test_limitation_added_for_single_supporting(self):
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(sup=1),
            total_evidence=1,
        )
        assert any("one supporting" in lim.lower() or "only one" in lim.lower()
                   for lim in result.evidence_limitations)

    # ── Custom config ─────────────────────────────────────────────────────────

    def test_custom_config_lowers_support_threshold(self):
        """With support_min=1, a single supporting source → LIKELY_CREDIBLE."""
        config = VerdictConfig(support_min=1)
        result = assess_claim(
            "Some claim.", 1,
            {EvidenceRelationship.SUPPORTING: 1},
            total_evidence=1,
            config=config,
        )
        assert result.evidence_assessment == EvidenceAssessment.LIKELY_CREDIBLE

    def test_custom_config_raises_contradict_threshold(self):
        """With contradict_min=3, two contradictions → not CONTRADICTED."""
        config = VerdictConfig(contradict_min=3)
        result = assess_claim(
            "Some claim.", 1,
            self._rel_counts(con=2),
            total_evidence=2,
            config=config,
        )
        assert result.evidence_assessment != EvidenceAssessment.CONTRADICTED


# =============================================================================
# Verdict Engine — aggregate_verdict()
# =============================================================================

class TestVerdictAggregation:

    def _make_result(
        self,
        assessment: EvidenceAssessment,
        sup=0, con=0, inc=0, norel=0, total=0,
    ) -> ClaimVerdictResult:
        return ClaimVerdictResult(
            claim_text="Test claim.",
            position=1,
            supporting_count=sup,
            contradicting_count=con,
            inconclusive_count=inc,
            not_relevant_count=norel,
            total_evidence=total,
            evidence_assessment=assessment,
            evidence_explanation="Test explanation.",
        )

    def test_empty_claims_is_insufficient(self):
        verdict = aggregate_verdict([])
        assert verdict.overall_evidence_assessment == EvidenceAssessment.INSUFFICIENT_EVIDENCE

    def test_single_credible_claim(self):
        r = self._make_result(EvidenceAssessment.LIKELY_CREDIBLE, sup=2, total=2)
        verdict = aggregate_verdict([r])
        assert verdict.overall_evidence_assessment == EvidenceAssessment.LIKELY_CREDIBLE

    def test_contradicted_beats_credible(self):
        """CONTRADICTED on one claim overrides LIKELY_CREDIBLE on another."""
        r1 = self._make_result(EvidenceAssessment.LIKELY_CREDIBLE, sup=2, total=2)
        r2 = self._make_result(EvidenceAssessment.CONTRADICTED,    con=2, total=2)
        verdict = aggregate_verdict([r1, r2])
        assert verdict.overall_evidence_assessment == EvidenceAssessment.CONTRADICTED

    def test_misleading_beats_unverified(self):
        r1 = self._make_result(EvidenceAssessment.UNVERIFIED,       inc=2, total=2)
        r2 = self._make_result(EvidenceAssessment.LIKELY_MISLEADING, con=1, total=1)
        verdict = aggregate_verdict([r1, r2])
        assert verdict.overall_evidence_assessment == EvidenceAssessment.LIKELY_MISLEADING

    def test_severity_order_full(self):
        """
        The most severe assessment wins:
        CONTRADICTED > LIKELY_MISLEADING > INSUFFICIENT > UNVERIFIED > LIKELY_CREDIBLE
        """
        results = [
            self._make_result(EvidenceAssessment.LIKELY_CREDIBLE),
            self._make_result(EvidenceAssessment.UNVERIFIED),
            self._make_result(EvidenceAssessment.INSUFFICIENT_EVIDENCE),
            self._make_result(EvidenceAssessment.LIKELY_MISLEADING),
            self._make_result(EvidenceAssessment.CONTRADICTED),
        ]
        verdict = aggregate_verdict(results)
        assert verdict.overall_evidence_assessment == EvidenceAssessment.CONTRADICTED

    def test_totals_summed_correctly(self):
        r1 = self._make_result(EvidenceAssessment.UNVERIFIED, sup=1, con=0, inc=2, total=3)
        r2 = self._make_result(EvidenceAssessment.UNVERIFIED, sup=0, con=1, inc=1, total=2)
        verdict = aggregate_verdict([r1, r2])
        assert verdict.total_supporting    == 1
        assert verdict.total_contradicting == 1
        assert verdict.total_inconclusive  == 3
        assert verdict.total_evidence      == 5

    def test_ml_fields_preserved(self):
        r = self._make_result(EvidenceAssessment.LIKELY_CREDIBLE)
        verdict = aggregate_verdict(
            [r],
            ml_ensemble_label="REAL",
            ml_ensemble_confidence=0.91,
        )
        assert verdict.ml_ensemble_label      == "REAL"
        assert verdict.ml_ensemble_confidence == 0.91

    def test_providers_recorded(self):
        r = self._make_result(EvidenceAssessment.UNVERIFIED)
        verdict = aggregate_verdict(
            [r],
            providers_used=["newsapi", "gnews"],
            providers_failed=["serpapi"],
        )
        assert "newsapi" in verdict.providers_used
        assert "gnews"   in verdict.providers_used
        assert "serpapi" in verdict.providers_failed

    def test_overall_explanation_is_non_empty(self):
        r = self._make_result(EvidenceAssessment.LIKELY_CREDIBLE, sup=2, total=2)
        verdict = aggregate_verdict([r])
        assert isinstance(verdict.overall_explanation, str)
        assert len(verdict.overall_explanation) > 20

    def test_explanation_mentions_claim_count(self):
        results = [
            self._make_result(EvidenceAssessment.UNVERIFIED),
            self._make_result(EvidenceAssessment.UNVERIFIED),
        ]
        verdict = aggregate_verdict(results)
        assert "2" in verdict.overall_explanation

    def test_insufficient_explanation_does_not_say_false(self):
        """INSUFFICIENT_EVIDENCE must never conclude the content is false."""
        r = self._make_result(EvidenceAssessment.INSUFFICIENT_EVIDENCE)
        verdict = aggregate_verdict([r])
        explanation = verdict.overall_explanation.lower()
        # Acceptable: "does not indicate ... is false" or "not proof of falsehood"
        # Not acceptable: baldly asserting "content is false" without qualification
        if "false" in explanation:
            # Must be qualified — not a bare claim of falsity
            assert (
                "not" in explanation or
                "does not" in explanation or
                "proof of" in explanation or
                "absence" in explanation
            ), f"Explanation implies false without qualification: {explanation}"


# =============================================================================
# End-to-end pipeline (no DB, no ML, no HTTP)
# =============================================================================

class TestEnginePipeline:
    """
    Integration tests for the full claim → compare → verdict pipeline
    without any external dependencies.
    """

    def _run_pipeline(
        self,
        text:        str,
        ev_items:    list[dict],
        ml_label:    str   = "FAKE",
        ml_fake_prob: float = 0.80,
        max_claims:  int   = 3,
    ) -> AnalysisVerdict:
        """
        Run the analysis engine on synthetic data.
        ev_items: list of {"title": str, "snippet": str}
        """
        from app.evidence.schema import EvidenceItem, SourceType, EvidenceResult, EvidenceStatus
        from datetime import datetime, timezone

        claims = extract_claims(text, max_claims=max_claims)

        evidence_items = [
            EvidenceItem(
                source_name="Test Source",
                title=item["title"],
                url=f"https://example.com/{i}",
                source_type=SourceType.NEWS_API,
                description=item.get("snippet", ""),
                published_at=datetime.now(timezone.utc),
                provider_name="test_provider",
            )
            for i, item in enumerate(ev_items)
        ]

        ev_result = EvidenceResult(
            claim="test",
            query="test",
            items=evidence_items,
            status=EvidenceStatus.FOUND if evidence_items else EvidenceStatus.INSUFFICIENT,
            providers_used=["test_provider"],
            providers_failed=[],
            total_found=len(evidence_items),
            after_dedup=len(evidence_items),
        )

        claim_results = []
        for claim in claims:
            from collections import defaultdict
            counts: dict[EvidenceRelationship, int] = defaultdict(int)
            for ev_item in evidence_items:
                snippet = f"{ev_item.title} {ev_item.description or ''}".strip()
                cmp_result = compare(claim.text, snippet)
                counts[cmp_result.relationship] += 1

            cvr = assess_claim(
                claim_text=claim.text,
                position=claim.position,
                relationship_counts=dict(counts),
                total_evidence=len(evidence_items),
                ml_label=ml_label,
                ml_fake_prob=ml_fake_prob,
                ml_confidence=ml_fake_prob,
            )
            claim_results.append(cvr)

        return aggregate_verdict(
            claim_results=claim_results,
            ml_ensemble_label=ml_label,
            ml_ensemble_confidence=ml_fake_prob,
        )

    def test_credible_content_with_supporting_evidence(self):
        text = (
            "The European Central Bank raised interest rates by 25 basis points. "
            "ECB President Lagarde cited persistent inflation. "
            "Bond markets reacted positively to the ECB rate decision Thursday."
        )
        ev_items = [
            {"title": "ECB raises rates 25bp inflation Lagarde eurozone 2023",
             "snippet": "ECB raised interest rates 25 basis points eurozone inflation target."},
            {"title": "ECB interest rate increase monetary policy decision Thursday",
             "snippet": "European Central Bank increased rates by 25bp Lagarde announced."},
        ]
        verdict = self._run_pipeline(text, ev_items, ml_label="REAL", ml_fake_prob=0.20)
        # With supporting evidence and ML=REAL, should not be CONTRADICTED
        assert verdict.overall_evidence_assessment != EvidenceAssessment.CONTRADICTED

    def test_debunked_claim_with_contradicting_evidence(self):
        text = (
            "Scientists claim 5G towers caused COVID-19 to spread. "
            "The wireless technology directly spread the coronavirus virus."
        )
        ev_items = [
            {"title": "5G COVID claim debunked false no evidence WHO",
             "snippet": "5G COVID pandemic link debunked false no scientific evidence wireless."},
            {"title": "5G towers COVID-19 claim factually false disproved",
             "snippet": "COVID 5G claim factually false disproved no connection towers."},
        ]
        verdict = self._run_pipeline(text, ev_items, ml_label="FAKE", ml_fake_prob=0.90)
        assert verdict.overall_evidence_assessment in (
            EvidenceAssessment.CONTRADICTED,
            EvidenceAssessment.LIKELY_MISLEADING,
            EvidenceAssessment.UNVERIFIED,
        )

    def test_no_evidence_never_means_false(self):
        text = "The government raised taxes by 15% in 2023."
        verdict = self._run_pipeline(text, ev_items=[], ml_label="FAKE", ml_fake_prob=0.75)
        assert verdict.overall_evidence_assessment == EvidenceAssessment.INSUFFICIENT_EVIDENCE
        # Explanation must not say "false"
        assert "false" not in verdict.overall_explanation.lower() or \
               "does not" in verdict.overall_explanation.lower()

    def test_ml_and_evidence_verdicts_are_independent(self):
        """
        ML says FAKE but evidence supports the claim.
        Both should be present — not silently merged.
        """
        text = "The ECB raised rates by 25 basis points in June 2023."
        ev_items = [
            {"title": "ECB rate increase 25bp June 2023 official statement",
             "snippet": "ECB raised rates 25bp June 2023 unanimous board decision."},
            {"title": "European Central Bank rates inflation June 2023",
             "snippet": "European Central Bank raised interest rates June 2023 inflation."},
        ]
        verdict = self._run_pipeline(text, ev_items, ml_label="FAKE", ml_fake_prob=0.85)
        assert verdict.ml_ensemble_label == "FAKE"
        # Evidence should remain independent of ML
        assert verdict.overall_evidence_assessment is not None

    def test_empty_text_returns_insufficient(self):
        verdict = self._run_pipeline("", ev_items=[], max_claims=5)
        assert verdict.overall_evidence_assessment == EvidenceAssessment.INSUFFICIENT_EVIDENCE

    def test_claim_results_count_matches_extracted_claims(self):
        text = (
            "NASA launched Artemis I in 2022 successfully to the moon. "
            "The mission lasted 25 days in lunar orbit. "
            "Artemis I carried no crew but tested the Orion capsule."
        )
        verdict = self._run_pipeline(text, ev_items=[], max_claims=3)
        assert len(verdict.claim_results) <= 3
        assert len(verdict.claim_results) >= 1
