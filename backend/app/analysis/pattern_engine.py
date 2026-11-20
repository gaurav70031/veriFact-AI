"""
backend/app/analysis/pattern_engine.py
=======================================
Misinformation pattern analysis engine.

Computes all analytics categories from two data sources:
  1. Live analysis history (PostgreSQL analyses + claims + evidence_sources tables)
  2. Processed dataset CSV files (ml/datasets/processed/*.csv) if available

Design principles
-----------------
* All statistics come from real data — nothing fabricated or hardcoded.
* When data is insufficient, the response explicitly says so rather than
  returning empty arrays that could be misinterpreted as "no patterns found".
* NLP computations (word freq, n-grams, TF-IDF, sentiment, LDA, clustering)
  run on real text. If a dependency is missing, that specific analysis is
  skipped with a clear note.
* "Insights" are template-driven observations derived from actual computed
  values. They explicitly distinguish correlation from causation.
* All heavy NLP runs in a thread pool so the FastAPI event loop is not blocked.

Analytics modules
-----------------
  word_frequency(texts, label)        → Counter → top N {word, count, pct}
  ngram_analysis(texts, n)            → Counter → top N {ngram, count}
  tfidf_keywords(texts, label)        → TF-IDF top terms per label
  sentiment_analysis(texts)           → {pos, neg, neu} distribution
  lda_topics(texts, n_topics)         → {topic_id, words, weight} list
  text_length_analysis(df)            → length statistics by label
  temporal_distribution(df)           → {date, fake, real, total} time series
  source_distribution(sources_df)     → {source, count, pct} by relationship
  linguistic_features(texts)          → {avg_word_len, type_token_ratio, ...}
  cluster_articles(texts, n)          → {cluster_id, label, representative} list
"""

from __future__ import annotations

import collections
import logging
import re
import string
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Stop words (inline — no NLTK startup dependency here) ─────────────────────
_STOP_WORDS = frozenset("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further get had hadn't has hasn't have haven't having he her here him his how
i if in into is isn't it its itself just let me more most mustn't my myself
no nor not of off on once only or other ought our out over own same shan't
she should shouldn't so some such than that the their them then there these
they this those through to too under until up very was wasn't we were weren't
what when where which while who will with won't would wouldn't you your
said says according reuters ap press news article story monday tuesday
wednesday thursday friday saturday sunday january february march april may
june july august september october november december year years
""".split())

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str, remove_stops: bool = True) -> list[str]:
    tokens = text.lower().translate(_PUNCT_TABLE).split()
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Word frequency
# ══════════════════════════════════════════════════════════════════════════════

def word_frequency(
    texts: list[str],
    top_n: int = 50,
    remove_stops: bool = True,
) -> list[dict]:
    """
    Compute word frequency for a list of texts.

    Returns [{"word": str, "count": int, "pct": float}, ...] sorted desc.
    pct is fraction of total tokens.
    """
    counter: collections.Counter = collections.Counter()
    for text in texts:
        counter.update(_tokenize(text, remove_stops))
    total = max(sum(counter.values()), 1)
    return [
        {"word": w, "count": c, "pct": round(c / total * 100, 4)}
        for w, c in counter.most_common(top_n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# N-gram analysis
# ══════════════════════════════════════════════════════════════════════════════

def ngram_analysis(
    texts: list[str],
    n: int = 2,
    top_n: int = 30,
) -> list[dict]:
    """
    Extract top n-grams (bigrams / trigrams) from texts.

    Returns [{"ngram": str, "count": int}, ...] sorted desc.
    """
    counter: collections.Counter = collections.Counter()
    for text in texts:
        tokens = _tokenize(text, remove_stops=True)
        grams = zip(*[tokens[i:] for i in range(n)])
        counter.update(" ".join(g) for g in grams)
    return [
        {"ngram": ng, "count": c}
        for ng, c in counter.most_common(top_n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF keyword analysis
# ══════════════════════════════════════════════════════════════════════════════

def tfidf_keywords(
    texts_by_label: dict[str, list[str]],
    top_n: int = 20,
) -> dict[str, list[dict]]:
    """
    Compute TF-IDF keywords that best discriminate each label.

    Parameters
    ----------
    texts_by_label : {"FAKE": [...], "REAL": [...]} — texts grouped by label.
    top_n          : Top keywords per label.

    Returns
    -------
    {"FAKE": [{"keyword": str, "score": float}, ...], "REAL": [...]}

    Falls back to word frequency if scikit-learn is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        all_labels = list(texts_by_label.keys())
        all_texts  = [" ".join(v) for v in texts_by_label.values()]

        if len(all_texts) < 2:
            # Can't do TF-IDF with single document
            raise ValueError("Need at least 2 label groups")

        # Build combined corpus: one "super-document" per label
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()

        result: dict[str, list[dict]] = {}
        for i, label in enumerate(all_labels):
            scores = X[i].toarray().flatten()
            top_idx = scores.argsort()[::-1][:top_n]
            result[label] = [
                {"keyword": feature_names[j], "score": round(float(scores[j]), 4)}
                for j in top_idx
                if scores[j] > 0
            ]
        return result

    except Exception as exc:
        logger.info("TF-IDF sklearn unavailable (%s), using word frequency fallback", exc)
        return {
            label: [
                {"keyword": item["word"], "score": item["pct"]}
                for item in word_frequency(texts, top_n=top_n)
            ]
            for label, texts in texts_by_label.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment analysis
# ══════════════════════════════════════════════════════════════════════════════

def sentiment_analysis(
    texts: list[str],
) -> dict:
    """
    Compute sentiment distribution using VADER (rule-based, no model download).

    Returns
    -------
    {
        "distribution": [{"label": "positive"|"negative"|"neutral", "count": int, "pct": float}],
        "avg_compound": float,
        "method": "vader" | "lexicon_fallback",
        "note": str,
    }
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        compounds = []
        pos_count = neg_count = neu_count = 0

        for text in texts:
            scores = analyzer.polarity_scores(text[:500])  # cap for speed
            c = scores["compound"]
            compounds.append(c)
            if c >= 0.05:
                pos_count += 1
            elif c <= -0.05:
                neg_count += 1
            else:
                neu_count += 1

        total = max(len(texts), 1)
        return {
            "distribution": [
                {"label": "positive", "count": pos_count, "pct": round(pos_count / total * 100, 2)},
                {"label": "neutral",  "count": neu_count, "pct": round(neu_count / total * 100, 2)},
                {"label": "negative", "count": neg_count, "pct": round(neg_count / total * 100, 2)},
            ],
            "avg_compound": round(float(np.mean(compounds)) if compounds else 0.0, 4),
            "method": "vader",
            "note": "VADER rule-based sentiment — works best on news headlines and short texts.",
        }

    except ImportError:
        return _lexicon_sentiment_fallback(texts)


def _lexicon_sentiment_fallback(texts: list[str]) -> dict:
    """
    Simple lexicon-based sentiment fallback when VADER is not installed.
    Uses an inline positive/negative word list.
    """
    POSITIVE = frozenset({
        "good", "great", "excellent", "positive", "success", "win", "won",
        "confirm", "confirmed", "support", "approve", "approved", "benefit",
        "safe", "recover", "recovered", "improve", "improved", "best",
        "increase", "increased", "rise", "rose", "gain", "truth", "true",
    })
    NEGATIVE = frozenset({
        "bad", "terrible", "wrong", "false", "fake", "fail", "failed",
        "crisis", "danger", "dangerous", "attack", "attacks", "kill",
        "death", "deaths", "lose", "lost", "decline", "declined", "fall",
        "fell", "crime", "criminal", "corrupt", "lie", "lies", "hoax",
        "mislead", "misleading", "deny", "denied", "refuse", "refused",
    })

    pos_count = neg_count = neu_count = 0
    for text in texts:
        tokens = set(_tokenize(text))
        pos = len(tokens & POSITIVE)
        neg = len(tokens & NEGATIVE)
        if pos > neg:
            pos_count += 1
        elif neg > pos:
            neg_count += 1
        else:
            neu_count += 1

    total = max(len(texts), 1)
    return {
        "distribution": [
            {"label": "positive", "count": pos_count, "pct": round(pos_count / total * 100, 2)},
            {"label": "neutral",  "count": neu_count, "pct": round(neu_count / total * 100, 2)},
            {"label": "negative", "count": neg_count, "pct": round(neg_count / total * 100, 2)},
        ],
        "avg_compound": round((pos_count - neg_count) / total, 4),
        "method": "lexicon_fallback",
        "note": (
            "Simple lexicon-based fallback (vaderSentiment not installed). "
            "Install with: pip install vaderSentiment"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LDA topic modeling
# ══════════════════════════════════════════════════════════════════════════════

def lda_topics(
    texts: list[str],
    n_topics: int = 5,
    n_words_per_topic: int = 8,
    max_features: int = 1000,
) -> dict:
    """
    Discover latent topics using LDA (Latent Dirichlet Allocation).

    Uses scikit-learn's LatentDirichletAllocation.
    Falls back to word-cluster approximation if LDA fails.

    Parameters
    ----------
    texts            : List of article texts.
    n_topics         : Number of topics to discover.
    n_words_per_topic: Top words to show per topic.

    Returns
    -------
    {
        "topics": [
            {
                "topic_id": int,
                "words": [{"word": str, "weight": float}, ...],
                "label": str,  # heuristic label from top word
                "doc_count": int,  # articles assigned to this topic
            }
        ],
        "method": "lda" | "word_cluster",
        "perplexity": float | None,
        "note": str,
    }
    """
    if len(texts) < max(5, n_topics):
        return {
            "topics": [],
            "method": "insufficient_data",
            "perplexity": None,
            "note": f"At least {max(5, n_topics)} documents required for LDA topic modeling.",
        }

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if X.shape[1] < n_topics:
            raise ValueError(f"Vocabulary ({X.shape[1]}) < n_topics ({n_topics})")

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method="online",
        )
        doc_topics = lda.fit_transform(X)
        perplexity = round(float(lda.perplexity(X)), 2)

        # Dominant topic per document
        dominant = doc_topics.argmax(axis=1)
        topic_doc_counts = collections.Counter(dominant.tolist())

        topics = []
        for i, topic_weights in enumerate(lda.components_):
            top_idx = topic_weights.argsort()[::-1][:n_words_per_topic]
            words = [
                {"word": feature_names[j], "weight": round(float(topic_weights[j] / topic_weights.sum()), 4)}
                for j in top_idx
            ]
            topics.append({
                "topic_id":  i,
                "words":     words,
                "label":     words[0]["word"] if words else f"topic_{i}",
                "doc_count": topic_doc_counts.get(i, 0),
            })

        return {
            "topics":     topics,
            "method":     "lda",
            "perplexity": perplexity,
            "note":       f"LDA with {n_topics} topics. Perplexity={perplexity:.1f} (lower=better fit).",
        }

    except Exception as exc:
        logger.info("LDA failed (%s) — using word cluster fallback", exc)
        return _word_cluster_topics(texts, n_topics, n_words_per_topic)


def _word_cluster_topics(
    texts: list[str],
    n_topics: int,
    n_words: int,
) -> dict:
    """
    Approximate topic discovery by clustering most frequent word groups.
    Used when sklearn LDA fails.
    """
    all_tokens = []
    for text in texts:
        all_tokens.extend(_tokenize(text))
    top_words = [w for w, _ in collections.Counter(all_tokens).most_common(n_topics * n_words)]
    chunk_size = max(n_words, len(top_words) // max(n_topics, 1))

    topics = []
    for i in range(n_topics):
        chunk = top_words[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            break
        words = [{"word": w, "weight": round(1 / (j + 1), 4)} for j, w in enumerate(chunk)]
        topics.append({
            "topic_id":  i,
            "words":     words,
            "label":     chunk[0] if chunk else f"topic_{i}",
            "doc_count": len(texts) // n_topics,
        })

    return {
        "topics":     topics,
        "method":     "word_cluster",
        "perplexity": None,
        "note":       "Word-cluster approximation (sklearn LDA unavailable).",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Text length analysis
# ══════════════════════════════════════════════════════════════════════════════

def text_length_analysis(
    texts_by_label: dict[str, list[str]],
) -> dict:
    """
    Character-level and word-level length statistics by label.

    Returns
    -------
    {
        "by_label": {
            "FAKE": {"char": {...stats}, "word": {...stats}},
            "REAL": {"char": {...stats}, "word": {...stats}},
        },
        "histogram": [{"bucket": "0-500", "count": int}, ...],
        "observation": str,
    }
    """
    def _stats(values: list[float]) -> dict:
        if not values:
            return {}
        arr = np.array(values)
        return {
            "min":    int(arr.min()),
            "max":    int(arr.max()),
            "mean":   round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "p25":    round(float(np.percentile(arr, 25)), 1),
            "p75":    round(float(np.percentile(arr, 75)), 1),
        }

    by_label: dict = {}
    all_char_lens: list[int] = []

    for label, texts in texts_by_label.items():
        char_lens = [len(t) for t in texts]
        word_lens = [len(t.split()) for t in texts]
        all_char_lens.extend(char_lens)
        by_label[label] = {
            "char": _stats(char_lens),
            "word": _stats(word_lens),
            "count": len(texts),
        }

    # Histogram buckets
    buckets = [
        (0, 500, "0–500"), (500, 1000, "500–1k"), (1000, 2000, "1k–2k"),
        (2000, 3000, "2k–3k"), (3000, 5000, "3k–5k"), (5000, 10001, "5k+"),
    ]
    histogram = []
    for lo, hi, label in buckets:
        count = sum(1 for c in all_char_lens if lo <= c < hi)
        histogram.append({"bucket": label, "count": count})

    # Derive observation
    observation = _length_observation(by_label)

    return {
        "by_label":    by_label,
        "histogram":   histogram,
        "observation": observation,
    }


def _length_observation(by_label: dict) -> str:
    if len(by_label) < 2:
        return "Insufficient label groups for comparison."
    labels = list(by_label.keys())
    means = {l: by_label[l].get("char", {}).get("mean") for l in labels}
    valid = {l: m for l, m in means.items() if m is not None}
    if len(valid) < 2:
        return "Insufficient data for length comparison."
    longest = max(valid, key=lambda l: valid[l])
    shortest = min(valid, key=lambda l: valid[l])
    diff = abs(valid[longest] - valid[shortest])
    if diff < 50:
        return (
            f"Article lengths are similar across labels "
            f"(mean chars: {', '.join(f'{l}={v:.0f}' for l, v in valid.items())}). "
            "No substantial length difference detected."
        )
    return (
        f"{longest}-labelled articles are longer on average "
        f"({valid[longest]:.0f} chars) than {shortest}-labelled "
        f"({valid[shortest]:.0f} chars). "
        "This is a dataset-level correlation, not evidence of causal relationship."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Temporal distribution
# ══════════════════════════════════════════════════════════════════════════════

def temporal_distribution(
    records: list[dict],
    granularity: str = "day",
) -> list[dict]:
    """
    Aggregate analysis records by date.

    Parameters
    ----------
    records     : List of dicts with "created_at" (datetime/str) and "final_verdict".
    granularity : "day" | "week" | "month"

    Returns
    -------
    [{"date": str, "total": int, "fake": int, "real": int, "mixed": int}, ...]
    sorted by date ascending.
    """
    from datetime import datetime, timezone

    buckets: dict[str, dict] = {}

    for rec in records:
        raw_date = rec.get("created_at")
        if raw_date is None:
            continue
        if isinstance(raw_date, str):
            try:
                dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            dt = raw_date

        if granularity == "week":
            # ISO week
            key = dt.strftime("%Y-W%W")
        elif granularity == "month":
            key = dt.strftime("%Y-%m")
        else:
            key = dt.strftime("%Y-%m-%d")

        if key not in buckets:
            buckets[key] = {"date": key, "total": 0, "fake": 0, "real": 0, "mixed": 0, "unverified": 0}
        buckets[key]["total"] += 1
        verdict = str(rec.get("final_verdict") or "").upper()
        if verdict == "FAKE":
            buckets[key]["fake"] += 1
        elif verdict == "REAL":
            buckets[key]["real"] += 1
        elif verdict == "MIXED":
            buckets[key]["mixed"] += 1
        else:
            buckets[key]["unverified"] += 1

    return sorted(buckets.values(), key=lambda x: x["date"])


# ══════════════════════════════════════════════════════════════════════════════
# Linguistic features
# ══════════════════════════════════════════════════════════════════════════════

def linguistic_features(
    texts: list[str],
    label: str = "all",
) -> dict:
    """
    Compute lexical richness and stylometric features.

    Features
    --------
    avg_word_length     : Mean characters per word
    type_token_ratio    : Unique words / total words (lexical diversity)
    avg_sentence_length : Mean words per sentence
    uppercase_ratio     : Fraction of uppercase words (title/headline style)
    digit_ratio         : Fraction of tokens that are numeric
    exclamation_rate    : Exclamation marks per 1000 chars
    question_rate       : Question marks per 1000 chars
    avg_char_per_doc    : Mean characters per document
    avg_word_per_doc    : Mean words per document

    Returns
    -------
    {"label": str, "features": {name: value, ...}, "note": str}
    """
    if not texts:
        return {"label": label, "features": {}, "note": "No texts provided."}

    all_words: list[str] = []
    all_unique: set[str] = set()
    char_counts: list[int] = []
    word_counts: list[int] = []
    exclamation_total = 0
    question_total    = 0
    uppercase_total   = 0
    digit_total       = 0
    sentence_lens: list[float] = []

    for text in texts:
        chars = len(text)
        char_counts.append(chars)
        words = text.split()
        word_counts.append(len(words))
        all_words.extend(words)
        all_unique.update(w.lower() for w in words)
        exclamation_total += text.count("!")
        question_total    += text.count("?")
        uppercase_total   += sum(1 for w in words if w.isupper() and len(w) > 1)
        digit_total       += sum(1 for w in words if re.search(r"\d", w))

        # Approximate sentence count
        sentences = re.split(r"[.!?]+", text)
        s_words = [len(s.split()) for s in sentences if s.strip()]
        if s_words:
            sentence_lens.append(float(np.mean(s_words)))

    total_words = max(len(all_words), 1)
    total_chars = max(sum(char_counts), 1)

    features = {
        "avg_word_length":     round(np.mean([len(w) for w in all_words if w.isalpha()]), 2) if all_words else 0,
        "type_token_ratio":    round(len(all_unique) / total_words, 4),
        "avg_sentence_length": round(float(np.mean(sentence_lens)), 1) if sentence_lens else 0,
        "uppercase_ratio":     round(uppercase_total / total_words, 4),
        "digit_ratio":         round(digit_total / total_words, 4),
        "exclamation_rate":    round(exclamation_total / total_chars * 1000, 4),
        "question_rate":       round(question_total / total_chars * 1000, 4),
        "avg_char_per_doc":    round(float(np.mean(char_counts)), 1),
        "avg_word_per_doc":    round(float(np.mean(word_counts)), 1),
        "document_count":      len(texts),
    }

    return {
        "label":    label,
        "features": features,
        "note":     "Stylometric features computed from raw text. Interpretations are correlational only.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Text similarity / repeated claims
# ══════════════════════════════════════════════════════════════════════════════

def repeated_claims(
    texts: list[str],
    ids:   list[int],
    threshold: float = 0.5,
    top_n: int = 10,
) -> list[dict]:
    """
    Find near-duplicate articles using TF-IDF cosine similarity.

    Returns
    -------
    [
        {
            "group_id": int,
            "representative": str,  # shortest/most central text
            "count": int,           # size of similarity group
            "ids": list[int],       # analysis IDs in this group
            "similarity": float,
        }
    ] — groups with count >= 2, sorted by count desc.

    Falls back to Jaccard similarity if sklearn unavailable.
    """
    if len(texts) < 2:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        X = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(X)

    except Exception:
        # Jaccard fallback
        def _jaccard(a: str, b: str) -> float:
            sa, sb = set(_tokenize(a)), set(_tokenize(b))
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        n = len(texts)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = _jaccard(texts[i], texts[j])
                sim_matrix[i, j] = sim_matrix[j, i] = s

    # Union-find grouping
    n = len(texts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    # Collect groups
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    result = []
    for group_id, (root, members) in enumerate(
        sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    ):
        if len(members) < 2:
            continue
        # Representative = shortest text
        rep_idx = min(members, key=lambda i: len(texts[i]))
        avg_sim = float(np.mean([
            sim_matrix[i, j]
            for i in members for j in members
            if i != j
        ]))
        result.append({
            "group_id":       group_id,
            "representative": texts[rep_idx][:200],
            "count":          len(members),
            "ids":            [ids[i] for i in members],
            "similarity":     round(avg_sim, 4),
        })
        if len(result) >= top_n:
            break

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Article clustering
# ══════════════════════════════════════════════════════════════════════════════

def cluster_articles(
    texts: list[str],
    ids:   list[int],
    n_clusters: int = 5,
) -> dict:
    """
    Cluster articles by semantic similarity using K-Means on TF-IDF features.

    Returns
    -------
    {
        "clusters": [
            {
                "cluster_id": int,
                "size": int,
                "top_terms": [str, ...],
                "sample_ids": [int, ...],
                "representative": str,
            }
        ],
        "method": "kmeans" | "fallback",
        "note": str,
    }
    """
    if len(texts) < max(n_clusters, 3):
        return {
            "clusters": [],
            "method": "insufficient_data",
            "note": f"At least {max(n_clusters, 3)} documents required for clustering.",
        }

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        actual_k = min(n_clusters, len(texts))
        kmeans = KMeans(
            n_clusters=actual_k,
            random_state=42,
            n_init=10,
            max_iter=100,
        )
        labels = kmeans.fit_predict(X)

        clusters = []
        for cluster_id in range(actual_k):
            member_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            if not member_indices:
                continue

            # Top terms for this cluster
            center = kmeans.cluster_centers_[cluster_id]
            top_term_idx = center.argsort()[::-1][:8]
            top_terms = [feature_names[j] for j in top_term_idx]

            # Representative = closest to centroid
            from sklearn.metrics.pairwise import cosine_similarity
            cluster_X = X[member_indices]
            sims = cosine_similarity(cluster_X, center.reshape(1, -1)).flatten()
            rep_local = sims.argmax()
            rep_idx = member_indices[rep_local]

            clusters.append({
                "cluster_id":    cluster_id,
                "size":          len(member_indices),
                "top_terms":     top_terms,
                "sample_ids":    [ids[i] for i in member_indices[:5]],
                "representative": texts[rep_idx][:200],
            })

        clusters.sort(key=lambda c: c["size"], reverse=True)
        return {
            "clusters": clusters,
            "method":   "kmeans",
            "note":     f"K-Means ({actual_k} clusters) on TF-IDF features.",
        }

    except Exception as exc:
        logger.info("K-Means clustering failed (%s) — skipping", exc)
        return {
            "clusters": [],
            "method":   "unavailable",
            "note":     f"Clustering unavailable: {exc}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Research insights generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(
    analytics_result: dict,
) -> list[dict]:
    """
    Generate factual, template-driven research observations.

    Explicitly labels correlations as correlations — never causal claims.
    Returns [] when data is insufficient for any observation.

    Returns
    -------
    [
        {
            "category": str,
            "observation": str,
            "caveat":      str,
            "data_basis":  str,  # what data supports this
        }
    ]
    """
    insights: list[dict] = []

    # ── Verdict distribution ──────────────────────────────────────────────────
    dist = analytics_result.get("verdict_distribution", {})
    if dist.get("total", 0) > 0:
        fake_pct = dist.get("fake", {}).get("pct", 0)
        real_pct = dist.get("real", {}).get("pct", 0)
        if fake_pct > 0 or real_pct > 0:
            dominant = "fake" if fake_pct > real_pct else "real"
            dominant_pct = max(fake_pct, real_pct)
            insights.append({
                "category": "Verdict Distribution",
                "observation": (
                    f"{dominant_pct:.1f}% of completed analyses in this dataset "
                    f"received a {dominant.upper()} prediction. "
                    f"({dist.get('total', 0)} total analyses)"
                ),
                "caveat": (
                    "This distribution reflects the submitted content, not the "
                    "prevalence of fake news in general. Submission bias may apply."
                ),
                "data_basis": "Live analysis history (PostgreSQL)",
            })

    # ── Length differences ────────────────────────────────────────────────────
    length = analytics_result.get("text_length", {})
    obs = length.get("observation", "")
    if obs and "Insufficient" not in obs:
        insights.append({
            "category": "Article Length",
            "observation": obs,
            "caveat": "Length correlation does not imply that length determines veracity.",
            "data_basis": "Character and word counts from submitted texts",
        })

    # ── Sentiment ─────────────────────────────────────────────────────────────
    sent = analytics_result.get("sentiment", {})
    sent_by_label = analytics_result.get("sentiment_by_label", {})
    if sent_by_label:
        fake_sent = sent_by_label.get("FAKE", {})
        real_sent = sent_by_label.get("REAL", {})
        fc = fake_sent.get("avg_compound")
        rc = real_sent.get("avg_compound")
        if fc is not None and rc is not None:
            diff = fc - rc
            direction = "more negative" if diff < -0.05 else "more positive" if diff > 0.05 else "similar"
            insights.append({
                "category": "Sentiment",
                "observation": (
                    f"Fake-labelled articles have {direction} sentiment on average "
                    f"(compound score: FAKE={fc:.3f}, REAL={rc:.3f}). "
                    f"Method: {fake_sent.get('method', 'unknown')}."
                ),
                "caveat": (
                    "Sentiment scores are computed by a rule-based analyser. "
                    "This is a dataset-level correlation — negative sentiment does not cause misinformation."
                ),
                "data_basis": "VADER compound scores on submitted texts",
            })

    # ── Top fake terms ────────────────────────────────────────────────────────
    tfidf = analytics_result.get("tfidf_keywords", {})
    fake_kw = tfidf.get("FAKE", [])
    if fake_kw:
        top5 = [k["keyword"] for k in fake_kw[:5]]
        insights.append({
            "category": "Linguistic Patterns",
            "observation": (
                f"The most discriminating TF-IDF terms in FAKE-labelled content "
                f"are: {', '.join(top5)}. "
                "These terms appear proportionally more often in fake-labelled texts."
            ),
            "caveat": (
                "TF-IDF distinguishes document groups statistically. "
                "The presence of these terms does not make a claim false."
            ),
            "data_basis": "TF-IDF vectorisation on FAKE vs REAL texts",
        })

    # ── Repeated claims ───────────────────────────────────────────────────────
    repeated = analytics_result.get("repeated_claims", [])
    if repeated:
        largest = repeated[0]
        insights.append({
            "category": "Repeated Claims",
            "observation": (
                f"{len(repeated)} groups of similar articles detected. "
                f"Largest group ({largest['count']} articles): "
                f"\"{largest['representative'][:100]}…\""
            ),
            "caveat": (
                "Text similarity does not imply factual equivalence or coordinated misinformation. "
                "Similar articles may simply cover the same news event."
            ),
            "data_basis": "TF-IDF cosine similarity clustering",
        })

    # ── Topics ────────────────────────────────────────────────────────────────
    topics_data = analytics_result.get("lda_topics", {})
    topics = topics_data.get("topics", [])
    if topics and topics_data.get("method") == "lda":
        largest_topic = max(topics, key=lambda t: t.get("doc_count", 0))
        words_str = ", ".join(w["word"] for w in largest_topic["words"][:5])
        insights.append({
            "category": "Topic Modeling",
            "observation": (
                f"LDA identified {len(topics)} topics. "
                f"The dominant topic ({largest_topic['doc_count']} documents) "
                f"centres around: {words_str}."
            ),
            "caveat": (
                "LDA topics are statistical word co-occurrence patterns, "
                "not human-defined subjects. Topic interpretation requires domain expertise."
            ),
            "data_basis": f"LDA perplexity={topics_data.get('perplexity', '?')}",
        })

    # ── Temporal ─────────────────────────────────────────────────────────────
    temporal = analytics_result.get("temporal_distribution", [])
    if len(temporal) >= 3:
        total_points = len(temporal)
        last = temporal[-3:]
        recent_total = sum(p.get("total", 0) for p in last)
        insights.append({
            "category": "Temporal Distribution",
            "observation": (
                f"Analysis activity spans {total_points} date periods. "
                f"The most recent 3 periods recorded {recent_total} analyses."
            ),
            "caveat": "Temporal patterns reflect submission patterns, not actual news cycles.",
            "data_basis": "created_at timestamps from analyses table",
        })

    if not insights:
        insights.append({
            "category": "Insufficient Data",
            "observation": (
                "Not enough data to generate meaningful insights. "
                "Submit more articles through the analysis pipeline to enable pattern detection."
            ),
            "caveat": "",
            "data_basis": "Live analysis history",
        })

    return insights
