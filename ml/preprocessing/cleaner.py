"""
Text cleaning pipeline.

Design goals
------------
* Reproducible — same input always produces the same output.
* Lossless enough — preserves enough signal for ML while removing noise.
* Two modes:
    - clean_for_baseline()  : heavy normalisation for TF-IDF models
    - clean_for_transformer(): light normalisation for BERT-family models
      (they benefit from natural punctuation and casing)

Steps (baseline mode)
---------------------
1.  Decode HTML entities           (&amp; → &, &lt; → <)
2.  Strip HTML/XML tags            (<b>text</b> → text)
3.  Remove URLs                    (http://... → "")
4.  Remove email addresses
5.  Normalise Unicode              (smart quotes, em-dashes → ASCII)
6.  Lowercase
7.  Expand common contractions     (don't → do not)
8.  Remove non-alphabetic chars    (keep spaces)
9.  Compress repeated chars        (looool → lool)
10. Collapse whitespace
11. Length filter                  (drop if shorter than MIN_TEXT_LENGTH)

Steps (transformer mode)
-------------------------
1.  Decode HTML entities
2.  Strip HTML/XML tags
3.  Remove URLs
4.  Normalise Unicode
5.  Collapse whitespace
6.  Truncate to MAX_TEXT_LENGTH characters
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

import pandas as pd

from ml.preprocessing.config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, MAX_REPEATED_CHARS

logger = logging.getLogger(__name__)

# ── Compiled regex patterns (module-level for performance) ────────────────────
_RE_HTML_TAG      = re.compile(r"<[^>]+>")
_RE_URL           = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL         = re.compile(r"\S+@\S+\.\S+")
_RE_NON_ALPHA     = re.compile(r"[^a-z\s]")
_RE_WHITESPACE    = re.compile(r"\s+")
_RE_REUTERS_TAG   = re.compile(
    r"^\s*\(reuters\)[\s\-–—]*", re.IGNORECASE
)  # strip Reuters dateline

# Compress N+ repeated characters down to MAX_REPEATED_CHARS
_RE_REPEATED      = re.compile(
    r"(.)\1{" + str(MAX_REPEATED_CHARS) + r",}", re.UNICODE
)
_REPEATED_REPL    = r"\1" * MAX_REPEATED_CHARS

# ── Contraction expansion table ───────────────────────────────────────────────
_CONTRACTIONS: dict[str, str] = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
    "it's": "it is", "that's": "that is", "what's": "what is",
    "he's": "he is", "she's": "she is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
}


def _expand_contractions(text: str) -> str:
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def _normalise_unicode(text: str) -> str:
    """
    Normalise Unicode to NFKC, then replace common non-ASCII punctuation
    with ASCII equivalents.
    """
    text = unicodedata.normalize("NFKC", text)
    # Smart quotes → straight
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dashes → space
    text = text.replace("\u2013", " ").replace("\u2014", " ")
    # Ellipsis → space
    text = text.replace("\u2026", " ")
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def clean_for_baseline(text: str) -> str:
    """
    Heavy cleaning for TF-IDF baseline models.
    Returns a lowercase alphabetic string or empty string if too short.
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML entities
    text = html.unescape(text)
    # 2. HTML tags
    text = _RE_HTML_TAG.sub(" ", text)
    # 3. Reuters dateline
    text = _RE_REUTERS_TAG.sub("", text)
    # 4. URLs
    text = _RE_URL.sub(" ", text)
    # 5. Emails
    text = _RE_EMAIL.sub(" ", text)
    # 6. Unicode normalisation
    text = _normalise_unicode(text)
    # 7. Lowercase
    text = text.lower()
    # 8. Contractions
    text = _expand_contractions(text)
    # 9. Non-alpha
    text = _RE_NON_ALPHA.sub(" ", text)
    # 10. Repeated chars
    text = _RE_REPEATED.sub(_REPEATED_REPL, text)
    # 11. Whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    if len(text) < MIN_TEXT_LENGTH:
        return ""
    return text[:MAX_TEXT_LENGTH]


def clean_for_transformer(text: str) -> str:
    """
    Light cleaning for transformer (DistilBERT) models.
    Preserves natural punctuation and casing.
    Removes only HTML, URLs, and normalises whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_REUTERS_TAG.sub("", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _normalise_unicode(text)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text[:MAX_TEXT_LENGTH]


def clean_series(
    series: pd.Series,
    mode: str = "baseline",
) -> pd.Series:
    """
    Apply cleaning to a pandas Series of texts.

    Parameters
    ----------
    series : pd.Series of str
    mode   : "baseline" | "transformer"

    Returns
    -------
    pd.Series of cleaned strings.  Rows that become empty strings after
    cleaning are returned as empty strings (caller decides whether to drop).
    """
    fn = clean_for_baseline if mode == "baseline" else clean_for_transformer

    cleaned = series.map(fn)

    empty_count = (cleaned == "").sum()
    if empty_count:
        logger.info(
            "clean_series(mode=%s): %d rows became empty after cleaning",
            mode, empty_count,
        )
    return cleaned
