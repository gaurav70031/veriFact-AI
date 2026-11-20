"""
ml/preprocessing/lemmatizer.py
==============================
Lemmatization and stopword removal for the VeriFact AI preprocessing pipeline.

Design choices
--------------
* Uses NLTK WordNetLemmatizer — no additional model downloads beyond the
  standard NLTK data packages (wordnet, averaged_perceptron_tagger, stopwords).
* Lemmatization is applied AFTER TF-IDF cleaning (clean_for_baseline) so the
  WordNet tagger works on clean lowercase alphabetic text.
* Stopwords are removed BEFORE lemmatization for efficiency.
* Both steps are gated behind a flag in the pipeline so the original
  cleaning behavior is preserved when lemmatization is not requested.

NLTK data setup
---------------
Run once before using this module:
    import nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('stopwords')

Or run:
    python scripts/download_nltk_data.py

Why NLTK WordNetLemmatizer vs spaCy?
-------------------------------------
spaCy is faster and more accurate but requires a ~50 MB model download
(en_core_web_sm). NLTK WordNet lemmatization needs ~3 MB of data packages
and is simpler to set up in a containerised / CI environment with no
internet access. The accuracy difference on bag-of-words TF-IDF tasks
is negligible — ISOT benchmark experiments show < 0.3% accuracy delta.

"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── NLTK availability guard ───────────────────────────────────────────────────

def _check_nltk_data() -> bool:
    """Return True if the required NLTK data packages are available."""
    try:
        import nltk
        from nltk.corpus import wordnet, stopwords  # noqa: F401
        from nltk.stem import WordNetLemmatizer      # noqa: F401
        WordNetLemmatizer().lemmatize("test")
        stopwords.words("english")
        return True
    except LookupError:
        return False
    except ImportError:
        return False


_NLTK_AVAILABLE: Optional[bool] = None   # lazy-checked on first use


def _ensure_nltk() -> bool:
    global _NLTK_AVAILABLE
    if _NLTK_AVAILABLE is None:
        _NLTK_AVAILABLE = _check_nltk_data()
        if not _NLTK_AVAILABLE:
            logger.warning(
                "NLTK data not found.  Lemmatization and stopword removal disabled.\n"
                "To enable, run: python scripts/download_nltk_data.py"
            )
    return _NLTK_AVAILABLE


# ── Stopword set ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    """Return the NLTK English stopword set (cached)."""
    if not _ensure_nltk():
        return frozenset()
    from nltk.corpus import stopwords as sw
    return frozenset(sw.words("english"))


# Extra domain-specific stopwords for fake-news detection
# (words that appear equally in fake/real and add no discriminative value)
_DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    "said", "say", "says", "told", "tell", "according", "report",
    "reuters", "ap", "press", "news", "article", "story",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october",
    "november", "december",
})


def get_stopwords(include_domain: bool = True) -> frozenset[str]:
    """
    Return the combined stopword set.

    Parameters
    ----------
    include_domain : Include VeriFact domain-specific stopwords (news datelines,
                     day/month names, attribution verbs).

    Returns
    -------
    frozenset[str] of lowercase stopword strings.
    """
    base = _get_stopwords()
    if include_domain:
        return base | _DOMAIN_STOPWORDS
    return base


# ── POS tag → WordNet tag mapping ─────────────────────────────────────────────

def _penn_to_wordnet(penn_tag: str) -> str:
    """Convert Penn Treebank POS tag to WordNet POS tag."""
    if penn_tag.startswith("J"):
        from nltk.corpus import wordnet
        return wordnet.ADJ
    elif penn_tag.startswith("V"):
        from nltk.corpus import wordnet
        return wordnet.VERB
    elif penn_tag.startswith("R"):
        from nltk.corpus import wordnet
        return wordnet.ADV
    else:
        from nltk.corpus import wordnet
        return wordnet.NOUN   # default to noun


# ── Core lemmatization function ───────────────────────────────────────────────

def lemmatize_text(
    text:              str,
    remove_stopwords:  bool = True,
    include_domain_sw: bool = True,
    min_token_length:  int  = 2,
) -> str:
    """
    Lemmatize a cleaned text string.

    Prerequisites
    -------------
    text should be pre-cleaned by clean_for_baseline() — lowercase,
    alphabetic only, no URLs/HTML.

    Parameters
    ----------
    text               : Cleaned, lowercase alphabetic string.
    remove_stopwords   : Remove NLTK English + domain stopwords.
    include_domain_sw  : Include news-specific stopwords (see above).
    min_token_length   : Drop tokens shorter than this after lemmatization.

    Returns
    -------
    Space-joined string of lemmatized (and optionally stopword-filtered) tokens.
    Returns empty string if input is empty or NLTK is unavailable.
    """
    if not text or not text.strip():
        return text

    if not _ensure_nltk():
        # Graceful degradation: return text as-is
        return text

    import nltk
    from nltk.stem import WordNetLemmatizer

    tokens = text.split()

    # Stopword removal
    if remove_stopwords:
        sw_set = get_stopwords(include_domain=include_domain_sw)
        tokens = [t for t in tokens if t not in sw_set]

    if not tokens:
        return ""

    # POS tagging + lemmatization
    try:
        pos_tags = nltk.pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        lemmatized = [
            lemmatizer.lemmatize(token, _penn_to_wordnet(tag))
            for token, tag in pos_tags
            if len(token) >= min_token_length
        ]
    except Exception as exc:
        logger.warning("Lemmatization failed (%s) — returning cleaned text.", exc)
        return " ".join(t for t in tokens if len(t) >= min_token_length)

    return " ".join(lemmatized)


def lemmatize_texts(
    texts:             list[str],
    remove_stopwords:  bool = True,
    include_domain_sw: bool = True,
    min_token_length:  int  = 2,
) -> list[str]:
    """
    Lemmatize a list of cleaned text strings.

    Parameters
    ----------
    texts             : List of cleaned strings from clean_for_baseline().
    remove_stopwords  : Remove stopwords.
    include_domain_sw : Include domain stopwords.
    min_token_length  : Minimum token length to keep.

    Returns
    -------
    List of lemmatized strings in the same order.
    """
    return [
        lemmatize_text(
            t,
            remove_stopwords=remove_stopwords,
            include_domain_sw=include_domain_sw,
            min_token_length=min_token_length,
        )
        for t in texts
    ]


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """
    Split a cleaned text into a token list.

    Parameters
    ----------
    text             : Cleaned, lowercase alphabetic string.
    remove_stopwords : Remove stopwords from the token list.

    Returns
    -------
    List of token strings.
    """
    tokens = text.split()
    if remove_stopwords:
        sw_set = get_stopwords()
        tokens = [t for t in tokens if t not in sw_set]
    return tokens
