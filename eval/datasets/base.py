"""Unified data format for all evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AmbiguousQuery:
    """Unified format across all datasets."""

    query: str                              # The ambiguous user query
    gold_clarifying_question: str           # Gold-standard clarifying question
    is_ambiguous: bool                      # Whether the query is actually ambiguous
    ambiguity_type: Optional[str] = None    # Ambiguity category (if available)
    dataset: str = ""                       # Source dataset name
    example_id: str = ""                    # Unique ID within dataset
    # Optional extra fields
    context: Optional[str] = None           # Additional context (CLAMBER has this)
    facet: Optional[str] = None             # Target facet/interpretation (Qulac)
    topic_id: Optional[str] = None          # Topic grouping ID (Qulac, ClariQ)
    raw: dict = field(default_factory=dict) # Original raw data for debugging

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "query": self.query,
            "gold_clarifying_question": self.gold_clarifying_question,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_type": self.ambiguity_type,
            "dataset": self.dataset,
            "example_id": self.example_id,
            "context": self.context,
            "facet": self.facet,
            "topic_id": self.topic_id,
        }
