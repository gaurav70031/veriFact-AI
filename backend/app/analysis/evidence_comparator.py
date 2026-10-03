"""
Evidence comparator.

For each (claim_text, evidence_snippet) pair, decides whether the evidence:
  SUPPORTING    — the snippet corroborates the claim
  CONTRADICTING — the snippet contradicts the claim
  INCONCLUSIVE  — related but no clear direction
  NOT_RELEVANT  — too low overlap to be meaningful

Design principles
-----------------
* No fabricated credibility scores. Scores come only from actual text overlap.
* No external API calls. Everything runs locally in milliseconds.
* Transparent rules: every decision has a traceable rationale string.
* Configurable thresholds: adjust SUPPORT_THRESHOLD / CONTRADICT_THRESHOLD
  without changing logic.

Comparison algorithm
--------------------
1. Tokenise both texts (lowercase, strip punctuation, remove stop words).
2. Compute Jaccard similarity on token sets → `token_overlap` (0–1).
3. Compute keyword overlap: fraction of claim keywords that appear in
   the snippet → `keyword_overlap` (0–1).
4. Detect negation / contradiction signals in the snippet (regex patterns
   for "did not", "false", "debunked", "no evidence", etc.).
5. Combine into a `comparison_score` (weighted blend of overlaps).
6. Apply decision rules:
     score < NOT_RELEVANT_THRESHOLD                   → NOT_RELEVANT
     score ≥ SUPPORT_THRESHOLD  + no contradiction    → SUPPORTING
     score ≥ SUPPORT_THRESHOLD  + contradiction found → CONTRADICTING
     otherwise                                        → INCONCLUSIVE

Upgrade path
------------
If `sentence-transformers` is installed, the module uses
`all-MiniLM-L6-v2` cosine similarity instead of token Jaccard.
This gives significantly better semantic matching (e.g. "raised rates"
matching "increased interest rates").  The thresholds remain the same.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from app.models.evidence_source import EvidenceRelationship

# ── Configurable thresholds ───────────────────────────────────────────────────
# Tune these to trade off precision vs recall.

NOT_RELEVANT_THRESHOLD = 0.08   # below this → NOT_RELEVANT
SUPPORT_THRESHOLD      = 0.20   # at or above + no contradiction → SUPPORTING
                                 # at or above + contradiction    → CONTRADICTING
# Contradiction signal must have weight above this to count
CONTRADICTION_SIGNAL_WEIGHT = 0.5


# ── Stop words (inline — no NLTK needed) ─────────────────────────────────────
_STOP_WORDS = frozenset("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further get had hasn't have having he her here him his how i if in into is
it its itself just let me more most my no nor not of off on once only or our
out over own same she should so some such than that the their them then there
these they this those through to too under until up very was we were what
when where which while who will with would you your
""".split())

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenise(text: str) -> set[str]:
    """Lowercase, strip punctuation, remove stop words, return token set."""
    tokens = text.lower().translate(_PUNCT_TABLE).split()
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _keyword_overlap(claim_tokens: set[str], snippet_tokens: set[str]) -> float:
    """Fraction of claim tokens that appear in the snippet."""
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & snippet_tokens) / len(claim_tokens)


# ── Contradiction detection patterns ─────────────────────────────────────────
_CONTRADICTION_PATTERNS = [
    # Explicit debunking
    (r"\b(debunked|disproved|false|misinformation|fabricated|misleading)\b",  1.0),
    # Direct negation of a claim
    (r"\b(did\s+not|does\s+not|have\s+not|has\s+not|is\s+not|are\s+not|was\s+not|were\s+not)\b", 0.8),
    (r"\bno\s+(evidence|proof|support|link|connection|relation)\b",            0.9),
    (r"\b(contradict|refutes?|rebuts?|counters?|denies?|denied)\b",            0.9),
    (r"\b(never|neither|nor)\b",                                               0.6),
    (r"\b(factually\s+incorrect|factually\s+inaccurate)\b",                    1.0),
    (r"\b(experts?\s+(say|said|state|stated|confirm|confirmed)\s+.{0,30}(false|untrue|wrong|incorrect))",  0.9),
]

_COMPILED_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), weight)
    for pat, weight in _CONTRADICTION_PATTERNS
]


def _contradiction_signal(snippet: str) -> float:
    """
    Return the maximum contradiction weight found in the snippet.
    0.0 = no contradiction signal; 1.0 = strong contradiction.
    """
    max_weight = 0.0
    for pattern, weight in _COMPILED_PATTERNS:
        if pattern.search(snippet):
            max_weight = max(max_weight, weight)
    return max_weight


# ── Sentence-transformer upgrade ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_sentence_transformer():
    """Load the sentence transformer model (cached — loaded once per process)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def _semantic_similarity(text_a: str, text_b: str) -> Optional[float]:
    """
    Compute cosine similarity using sentence-transformers if available.
    Returns None when the library is not installed.
    """
    model = _load_sentence_transformer()
    if model is None:
        return None
    try:
        import numpy as np
        embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
        score = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))   # clamp to [0, 1]
    except Exception:
        return None


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """
    Result of comparing one evidence item against one claim.

    Fields
    ------
    relationship    : The classified relationship.
    comparison_score: Overall semantic similarity (0–1). Derived from
                      actual text overlap — NOT a fabricated credibility score.
    contradiction_signal: Weight of contradiction patterns found (0–1).
    rationale       : Human-readable explanation of the decision.
    method          : "semantic" | "lexical" — which algorithm was used.
    """
    relationship:         EvidenceRelationship
    comparison_score:     float
    contradiction_signal: float
    rationale:            str
    method:               str = "lexical"


def compare(
    claim_text:   str,
    snippet:      str,
    claim_keywords: Optional[list[str]] = None,
) -> ComparisonResult:
    """
    Classify the relationship between a claim and an evidence snippet.

    Parameters
    ----------
    claim_text      : The factual claim being checked.
    snippet         : The evidence snippet (title + description, ≤ 500 chars).
    claim_keywords  : Optional pre-extracted keywords from the claim (from
                      query_extractor). Used to boost keyword_overlap scoring.

    Returns
    -------
    ComparisonResult with relationship, score, and rationale.

    Notes
    -----
    * Never fabricates a score — every number is derived from text content.
    * Returns INCONCLUSIVE when uncertain rather than guessing.
    * Does NOT interpret absence of results as proof of falsehood.
    """
    if not claim_text.strip() or not snippet.strip():
        return ComparisonResult(
            relationship=EvidenceRelationship.NOT_RELEVANT,
            comparison_score=0.0,
            contradiction_signal=0.0,
            rationale="Empty claim or snippet — cannot compare.",
        )

    # ── Step 1: Try semantic similarity ──────────────────────────────────────
    method = "lexical"
    semantic_score = _semantic_similarity(claim_text, snippet)

    # ── Step 2: Lexical overlap (always computed as fallback/supplement) ──────
    claim_tokens   = _tokenise(claim_text)
    snippet_tokens = _tokenise(snippet)

    jaccard  = _jaccard(claim_tokens, snippet_tokens)
    kw_score = _keyword_overlap(
        set(claim_keywords) if claim_keywords else claim_tokens,
        snippet_tokens,
    )

    if semantic_score is not None:
        # Blend: 70% semantic + 30% keyword overlap
        comparison_score = 0.70 * semantic_score + 0.30 * kw_score
        method = "semantic"
    else:
        # Lexical only: 50% Jaccard + 50% keyword overlap
        comparison_score = 0.50 * jaccard + 0.50 * kw_score

    comparison_score = round(min(max(comparison_score, 0.0), 1.0), 4)

    # ── Step 3: Contradiction detection ──────────────────────────────────────
    contradiction_signal = _contradiction_signal(snippet)

    # ── Step 4: Decision rules ────────────────────────────────────────────────
    if comparison_score < NOT_RELEVANT_THRESHOLD:
        return ComparisonResult(
            relationship=EvidenceRelationship.NOT_RELEVANT,
            comparison_score=comparison_score,
            contradiction_signal=contradiction_signal,
            rationale=(
                f"Low content overlap (score={comparison_score:.3f}) — "
                "evidence is not relevant to this claim."
            ),
            method=method,
        )

    if comparison_score >= SUPPORT_THRESHOLD:
        if contradiction_signal >= CONTRADICTION_SIGNAL_WEIGHT:
            return ComparisonResult(
                relationship=EvidenceRelationship.CONTRADICTING,
                comparison_score=comparison_score,
                contradiction_signal=contradiction_signal,
                rationale=(
                    f"Related content (score={comparison_score:.3f}) with "
                    f"contradiction signals (strength={contradiction_signal:.2f}). "
                    "Evidence appears to contradict or dispute the claim."
                ),
                method=method,
            )
        return ComparisonResult(
            relationship=EvidenceRelationship.SUPPORTING,
            comparison_score=comparison_score,
            contradiction_signal=contradiction_signal,
            rationale=(
                f"Good content overlap (score={comparison_score:.3f}) with "
                "no contradicting signals. Evidence aligns with the claim."
            ),
            method=method,
        )

    # Between NOT_RELEVANT and SUPPORT thresholds — inconclusive
    return ComparisonResult(
        relationship=EvidenceRelationship.INCONCLUSIVE,
        comparison_score=comparison_score,
        contradiction_signal=contradiction_signal,
        rationale=(
            f"Partial content overlap (score={comparison_score:.3f}). "
            "Evidence is related to the topic but does not clearly "
            "support or contradict the claim."
        ),
        method=method,
    )
