"""
Claim extractor.

Splits article text into discrete, checkable factual claims.

Design philosophy
-----------------
A "claim" is a single declarative sentence that:
  * Asserts a fact (not a question, opinion marker, or filler)
  * Is specific enough to search for (contains a named entity, number,
    location, or concrete noun — not just "it was confirmed today")
  * Is concise enough to use as a search query (≤ 300 chars)

Strategy
--------
1. Split text into sentences using a simple sentence boundary detector
   (no heavy NLP library required at import time).
2. Score each sentence on several heuristics:
     - Contains a number or percentage → high signal
     - Contains a likely named entity (capitalised multi-word phrase) → high signal
     - Contains a year (4-digit) → medium signal
     - Avoids opinion markers ("I think", "perhaps", "allegedly") → not a claim
     - Length 20–300 chars → appropriate for searching
3. Return the top-N highest-scoring sentences as claims.

For very short inputs (≤ 300 chars) the entire text is returned as a
single claim, because the input is itself already claim-sized.

No external model is called — this runs synchronously and instantly.
A spaCy upgrade path is provided: if spaCy + en_core_web_sm are installed,
sentence segmentation is more accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Sentence boundary detection ───────────────────────────────────────────────
# Splits on ". " or "! " or "? " followed by an uppercase letter,
# but NOT on common abbreviations (Mr., Dr., etc.)
_ABBREV = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "inc", "ltd", "corp", "est", "approx", "govt", "dept", "fig",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "u.s", "u.k", "e.g", "i.e",
})

_RE_SENTENCE_END = re.compile(r"(?<!\b\w)(?<=[.!?])\s+(?=[A-Z\"])")


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Uses spaCy if available; falls back to regex-based splitter.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            nlp.add_pipe("sentencizer")
        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        doc = nlp(text[:20_000])
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except (ImportError, Exception):
        pass

    # Regex fallback
    # First, protect known abbreviations from being treated as sentence ends
    protected = re.sub(
        r"\b(" + "|".join(_ABBREV) + r")\.",
        lambda m: m.group(0).replace(".", "<!DOT!>"),
        text,
        flags=re.IGNORECASE,
    )
    raw_parts = _RE_SENTENCE_END.split(protected)
    sentences = []
    for part in raw_parts:
        s = part.replace("<!DOT!>", ".").strip()
        if s:
            sentences.append(s)
    return sentences


# ── Sentence scoring ──────────────────────────────────────────────────────────
_RE_YEAR         = re.compile(r"\b(19|20)\d{2}\b")
_RE_NUMBER       = re.compile(r"\b\d[\d,]*\.?\d*\s*(%|percent|million|billion|trillion|thousand)?\b", re.IGNORECASE)
_RE_ENTITY       = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")   # Title-case runs
_RE_ACRONYM      = re.compile(r"\b[A-Z]{2,5}\b")                          # WHO, NASA, etc.
_RE_QUOTE_MARKER = re.compile(r"\b(said|stated|announced|confirmed|reported|claimed)\b", re.IGNORECASE)

# Sentences containing these patterns are likely opinions/hedges, not checkable facts
_OPINION_MARKERS = re.compile(
    r"\b(i think|in my opinion|allegedly|supposedly|rumoured|some say|"
    r"many believe|could be|might be|may be|perhaps|possibly|seems to|"
    r"appears to|according to some|unverified|unclear if|not confirmed)\b",
    re.IGNORECASE,
)


def _score_sentence(sentence: str) -> float:
    """Score a sentence on its likelihood of being a checkable factual claim."""
    length = len(sentence)

    # Hard disqualifiers
    if length < 20 or length > 400:
        return 0.0
    if sentence.endswith("?"):
        return 0.0   # questions are not claims
    if _OPINION_MARKERS.search(sentence):
        return 0.0

    score = 1.0

    # Strong signals
    if _RE_NUMBER.search(sentence):
        score += 2.5
    if _RE_YEAR.search(sentence):
        score += 1.5
    if _RE_ENTITY.search(sentence):
        score += 2.0
    if _RE_ACRONYM.search(sentence):
        score += 1.0
    if _RE_QUOTE_MARKER.search(sentence):
        score += 1.0   # attributed statement — checkable

    # Prefer medium-length sentences (70–200 chars) — specific but not rambling
    if 70 <= length <= 200:
        score += 0.5

    return score


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class ExtractedClaim:
    """A single factual claim extracted from article text."""
    text:     str
    position: int          # 1-based ordinal in the original text
    score:    float = 0.0  # extraction confidence (used for ordering only)


def extract_claims(
    text:      str,
    max_claims: int = 5,
) -> list[ExtractedClaim]:
    """
    Extract up to `max_claims` discrete factual claims from text.

    Parameters
    ----------
    text       : Raw article text or claim string.
    max_claims : Maximum number of claims to return (default 5).
                 For a single-sentence input this is always 1.

    Returns
    -------
    List of ExtractedClaim objects, ordered by position in the source text.
    Returns at least one claim (the full text if nothing better is found).
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Short inputs are themselves a single claim
    if len(text) <= 300:
        return [ExtractedClaim(text=text[:300], position=1, score=5.0)]

    sentences = _split_sentences(text)

    # Score each sentence
    scored: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences, start=1):
        score = _score_sentence(sentence)
        if score > 0:
            scored.append((score, idx, sentence))

    # Sort by score desc, keep top N, then re-sort by original position
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_claims]
    top_by_position = sorted(top, key=lambda x: x[1])

    claims = [
        ExtractedClaim(text=sentence[:300], position=idx, score=score)
        for score, idx, sentence in top_by_position
    ]

    # Always return at least one claim
    if not claims:
        first_sentence = sentences[0][:300] if sentences else text[:300]
        return [ExtractedClaim(text=first_sentence, position=1, score=1.0)]

    return claims
