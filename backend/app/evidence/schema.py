"""
Evidence schema — common data model for all provider results.

Design constraints
------------------
* Full article text is NOT stored — only title, URL, and a short description
  (≤ 500 chars) to respect copyright.
* retrieved_at is always populated by the provider, not the caller.
* relevance_score is set by the ranker after all results are assembled.
* provider_name identifies which adapter produced the result (for logging).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class SourceType(str, enum.Enum):
    NEWS_API   = "news_api"
    GNEWS      = "gnews"
    RSS_FEED   = "rss_feed"
    WEB_SEARCH = "web_search"
    OFFICIAL   = "official"
    UNKNOWN    = "unknown"


@dataclass
class EvidenceItem:
    """
    Normalised evidence article.

    All providers must return results in this shape.
    relevance_score is set to 0.0 by default and updated by the ranker.
    """

    # Required fields
    source_name:    str               # e.g. "Reuters", "BBC News"
    title:          str               # Article headline
    url:            str               # Canonical article URL
    source_type:    SourceType

    # Optional fields
    description:    Optional[str]     = None   # ≤ 500 chars, copyright-safe excerpt
    published_at:   Optional[datetime] = None  # Article publication datetime (UTC)
    retrieved_at:   datetime           = field(default_factory=datetime.utcnow)
    provider_name:  str                = "unknown"
    relevance_score: float             = 0.0

    def to_dict(self) -> dict:
        return {
            "source_name":    self.source_name,
            "title":          self.title,
            "url":            self.url,
            "source_type":    self.source_type.value,
            "description":    self.description,
            "published_at":   self.published_at.isoformat() if self.published_at else None,
            "retrieved_at":   self.retrieved_at.isoformat(),
            "provider_name":  self.provider_name,
            "relevance_score": round(self.relevance_score, 4),
        }


class EvidenceStatus(str, enum.Enum):
    FOUND               = "found"
    INSUFFICIENT        = "INSUFFICIENT_EVIDENCE"
    ALL_PROVIDERS_FAILED = "all_providers_failed"


@dataclass
class EvidenceResult:
    """
    Container returned by the evidence service to the analysis pipeline.
    """

    claim:          str
    query:          str
    items:          list[EvidenceItem]
    status:         EvidenceStatus
    providers_used: list[str]          # provider_names that returned results
    providers_failed: list[str]        # provider_names that errored
    total_found:    int                = 0
    after_dedup:    int                = 0
    search_time_ms: float              = 0.0

    @property
    def has_evidence(self) -> bool:
        return self.status == EvidenceStatus.FOUND and len(self.items) > 0

    def to_dict(self) -> dict:
        return {
            "claim":            self.claim,
            "query":            self.query,
            "status":           self.status.value,
            "total_found":      self.total_found,
            "after_dedup":      self.after_dedup,
            "search_time_ms":   round(self.search_time_ms, 2),
            "providers_used":   self.providers_used,
            "providers_failed": self.providers_failed,
            "items":            [i.to_dict() for i in self.items],
        }
