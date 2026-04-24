"""Binary LLM-as-judge for clarifying-question eval.

A DIFFERENT model from the system under test (default: gemma:7b) reads an
ambiguous query, the gold clarifying question(s), and the predicted
clarifying question, and decides whether the prediction targets the same
underlying ambiguity as any gold. Output is schema-constrained JSON so
the judge can't produce malformed output.

For datasets with multiple golds per query (ClariQ, Qulac), the judge
sees all golds at once and does max-over-golds internally — match=1 if
the prediction matches ANY gold.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from d2c.llm import LLMClient

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are evaluating a clarifying-question system for ambiguous user queries.

You will see:
1. An ambiguous USER QUERY.
2. One or more GOLD clarifying questions (what a human expert would ask).
3. A PREDICTED clarifying question from the system.

Decide: does the PREDICTED question target the SAME underlying ambiguity as ANY of the gold questions?

Rules:
- "Same ambiguity" means the predicted question would help resolve the same axis of uncertainty in the user's intent as at least one gold. Different wording, phrasing, or specificity is fine; different axis of ambiguity is NOT.
- If there are multiple golds, match=1 if the predicted question matches ANY of them (max-over-golds).
- If the predicted is generic ("can you clarify?", "what do you mean?"), match=0.
- If the predicted answers the query instead of asking for clarification, match=0.

Output JSON: {"match": 0 or 1, "reason": "one short sentence"}
"""

JUDGE_USER = """USER QUERY: {query}

GOLD clarifying question(s):
{golds}

PREDICTED clarifying question: {predicted}

Does PREDICTED target the same ambiguity as any gold?"""


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["match", "reason"],
}


@dataclass
class JudgeResult:
    match: int  # 0 or 1
    reason: str
    raw: str
    format_failed: bool = False

    def to_dict(self) -> dict:
        return {
            "match": self.match,
            "reason": self.reason,
            "format_failed": self.format_failed,
        }


def binary_judge(
    query: str,
    predicted: str,
    golds: list[str],
    llm: LLMClient,
) -> JudgeResult:
    """Ask the judge whether `predicted` matches any of `golds` on ambiguity."""
    if not predicted.strip():
        # Empty prediction can't match anything.
        return JudgeResult(match=0, reason="(empty prediction)", raw="", format_failed=False)
    if not golds:
        return JudgeResult(match=0, reason="(no gold)", raw="", format_failed=False)

    golds_block = "\n".join(f"- {g}" for g in golds if g.strip())
    user = JUDGE_USER.format(query=query, golds=golds_block, predicted=predicted)

    raw = llm.chat(
        system_prompt=JUDGE_SYSTEM,
        user_prompt=user,
        temperature=0.0,
        max_tokens=300,
        format_schema=JUDGE_SCHEMA,
    )
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a dict")
        return JudgeResult(
            match=int(data.get("match", 0)),
            reason=str(data.get("reason", "")).strip(),
            raw=raw,
            format_failed=False,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Judge parse failed: %s | raw=%r", e, raw[:200])
        return JudgeResult(
            match=0, reason="(judge parse failed)", raw=raw, format_failed=True
        )
