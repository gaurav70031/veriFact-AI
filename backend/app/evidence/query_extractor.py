"""
Query extractor.

Converts a raw user claim into one or more search queries suitable for
submitting to news APIs and search providers.

Strategy
--------
1. Remove boilerplate phrases that add noise but no search signal
   ("according to", "it has been reported that", etc.).
2. Strip stop words using an inline list (no NLTK dependency at import time).
3. Extract capitalised named entities heuristically (no spaCy needed).
4. Score remaining tokens by approximate importance:
     - Named entities score highest
     - Numbers, years score medium
     - Other content words score lower
5. Build a primary query from the top-N tokens.
6. Build 1–2 fallback queries by dropping the lowest-scored terms.

This is intentionally lightweight — no heavy NLP libraries are required at
runtime so the backend starts quickly.  If spaCy is installed it is used
for better NER; if not, heuristic extraction is used.

Returns a QuerySet with:
  primary_query : str      — best single-string query
  fallback_queries: list   — alternatives to try if primary returns nothing
  keywords      : list     — extracted terms (for logging/display)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field


# ── Stop words (inline — no NLTK import needed at startup) ───────────────────
_STOP_WORDS = frozenset("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further get got had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves
""".split())

# Phrases to strip before extraction
_NOISE_PHRASES = [
    r"\baccording to\b", r"\bit has been reported\b", r"\bsources say\b",
    r"\bexperts claim\b", r"\bsome claim\b", r"\bpeople say\b",
    r"\bwidely reported\b", r"\bmany believe\b", r"\bsome allege\b",
    r"\bso-called\b", r"\bso called\b", r"\bthe so-called\b",
]

_NOISE_RE = re.compile("|".join(_NOISE_PHRASES), re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s\-]")
_SPACE_RE = re.compile(r"\s+")


# ── Named entity heuristics ───────────────────────────────────────────────────

def _extract_entities_heuristic(text: str) -> list[str]:
    """
    Extract likely named entities as capitalised consecutive words
    (e.g. "Joe Biden", "World Health Organization", "COVID-19").
    """
    # Find runs of Title-Case or ALL-CAPS words (min 2 chars each)
    pattern = re.compile(
        r'\b(?:[A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,})*)\b'
    )
    candidates = pattern.findall(text)
    # Filter out sentence-start capitalisation by requiring multi-word or
    # known-pattern tokens (years, acronyms, hyphenated terms)
    entities = []
    for c in candidates:
        words = c.split()
        if len(words) >= 2:
            entities.append(c)
        elif len(c) <= 5 and c.isupper():
            # Short acronym: WHO, NASA, COVID, etc.
            entities.append(c)
    return entities


def _try_spacy_entities(text: str) -> list[str]:
    """Use spaCy for NER if installed; returns [] otherwise."""
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return []
        doc = nlp(text)
        return [ent.text for ent in doc.ents
                if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "LAW")]
    except ImportError:
        return []


# ── Token scoring ─────────────────────────────────────────────────────────────

def _score_token(token: str, entities: set[str]) -> float:
    token_lower = token.lower()
    if token_lower in _STOP_WORDS:
        return 0.0
    if len(token) < 3:
        return 0.0

    score = 1.0

    # Named entity bonus
    for ent in entities:
        if token.lower() in ent.lower() or ent.lower() in token.lower():
            score += 3.0
            break

    # Number / year bonus (often key facts in claims)
    if re.match(r"^\d{4}$", token):  # year
        score += 2.0
    elif re.match(r"^\d+(\.\d+)?%?$", token):  # percentage / number
        score += 1.5

    # Length bonus (longer words tend to be more specific)
    score += min(len(token) / 20, 0.5)

    return score


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class QuerySet:
    primary_query:    str
    fallback_queries: list[str] = field(default_factory=list)
    keywords:         list[str] = field(default_factory=list)


def extract_queries(
    claim:      str,
    max_keywords: int = 8,
    max_queries:  int = 3,
) -> QuerySet:
    """
    Extract search queries from a claim string.

    Parameters
    ----------
    claim        : The raw factual claim text.
    max_keywords : Max tokens to include in the primary query.
    max_queries  : Total number of queries to generate (1 primary + fallbacks).

    Returns
    -------
    QuerySet with primary_query, fallback_queries, keywords.
    """
    if not claim or not claim.strip():
        return QuerySet(primary_query=claim or "", keywords=[])

    # 1. Strip noise phrases
    clean = _NOISE_RE.sub(" ", claim)
    # 2. Strip punctuation except hyphens
    clean = _PUNCT_RE.sub(" ", clean)
    clean = _SPACE_RE.sub(" ", clean).strip()

    # 3. Extract named entities (spaCy preferred, heuristic fallback)
    entities_list = _try_spacy_entities(clean) or _extract_entities_heuristic(clean)
    entities_set  = {e.lower() for e in entities_list}

    # 4. Tokenise and score
    tokens = clean.split()
    scored = []
    for token in tokens:
        stripped = token.strip(string.punctuation)
        if not stripped:
            continue
        score = _score_token(stripped, entities_set)
        if score > 0:
            scored.append((stripped, score))

    # Sort by score descending, deduplicate (case-insensitive)
    seen: set[str] = set()
    ranked: list[str] = []
    for token, _ in sorted(scored, key=lambda x: x[1], reverse=True):
        if token.lower() not in seen:
            ranked.append(token)
            seen.add(token.lower())

    keywords = ranked[:max_keywords]

    if not keywords:
        # Fallback: use the raw claim truncated
        primary_query = claim[:200]
        return QuerySet(primary_query=primary_query, keywords=[])

    # 5. Build queries
    primary_query = " ".join(keywords)

    fallback_queries: list[str] = []
    if len(keywords) > 3 and max_queries > 1:
        # Fallback 1: top half of keywords
        half = max(3, len(keywords) // 2)
        fallback_queries.append(" ".join(keywords[:half]))
    if len(keywords) > 5 and max_queries > 2:
        # Fallback 2: named entities only (if any)
        ent_tokens = [k for k in keywords if k.lower() in entities_set]
        if ent_tokens and len(ent_tokens) < len(keywords):
            fallback_queries.append(" ".join(ent_tokens[:5]))

    return QuerySet(
        primary_query=primary_query,
        fallback_queries=fallback_queries[:max_queries - 1],
        keywords=keywords,
    )
