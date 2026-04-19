"""Data loader for ClarifyMT-Bench dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClarifyMTBenchItem:
    """A single ClarifyMT-Bench example."""

    query: str                  # The ambiguous user query (turn 1)
    gold_clarifying_question: str  # The reference clarifying question (turn 2)
    user_response: str          # The user's clarified response (turn 3)
    category: str               # Ambiguity type (e.g. "Lexical Ambiguity")
    user_type: str              # User persona (e.g. "Precise", "Partial-Vague")
    model_name: str             # Model that generated this example
    explanation: str            # Why the query is ambiguous
    conversation: list[dict]    # Full raw conversation

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "gold_clarifying_question": self.gold_clarifying_question,
            "user_response": self.user_response,
            "category": self.category,
            "user_type": self.user_type,
            "model_name": self.model_name,
            "explanation": self.explanation,
        }


def load_clarifymt_bench(
    path: str | Path = "data/clarifymt_bench.jsonl",
    categories: list[str] | None = None,
    user_types: list[str] | None = None,
    limit: int | None = None,
) -> list[ClarifyMTBenchItem]:
    """Load and optionally filter the ClarifyMT-Bench dataset.

    Args:
        path: Path to the JSONL file.
        categories: If set, only include these ambiguity categories.
        user_types: If set, only include these user types.
        limit: If set, return at most this many items.
    """
    items: list[ClarifyMTBenchItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if categories and obj.get("category") not in categories:
                continue
            if user_types and obj.get("user_type") not in user_types:
                continue

            conv = obj["conversation"]
            # Conversation structure: [user query, assistant clarification, user response]
            query = conv[0]["content"] if len(conv) > 0 else ""
            gold_cq = conv[1]["content"] if len(conv) > 1 else ""
            user_resp = conv[2]["content"] if len(conv) > 2 else ""

            items.append(ClarifyMTBenchItem(
                query=query,
                gold_clarifying_question=gold_cq,
                user_response=user_resp,
                category=obj.get("category", ""),
                user_type=obj.get("user_type", ""),
                model_name=obj.get("model_name", ""),
                explanation=obj.get("explanation", ""),
                conversation=conv,
            ))

            if limit and len(items) >= limit:
                break

    return items


# ---------------------------------------------------------------------------
# Convenience: list available categories and user types
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Lexical Ambiguity",
    "Syntactic Ambiguity",
    "Semantic Ambiguity",
    "Goal Ambiguity",
    "Scope Ambiguity",
    "Intent Conflict Ambiguity",
    "Entity Ambiguity",
    "Spatial Ambiguity",
    "Temporal Ambiguity",
    "Knowledge Gap Ambiguity",
    "Unfamiliarity Ambiguity",
    "Value Ambiguity",
]

USER_TYPES = [
    "Precise",
    "Partial-Vague",
    "Off-Focus",
    "Contradictory",
    "Factually-Wrong",
    "Refusal",
]
