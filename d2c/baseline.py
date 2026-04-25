"""Single-agent vanilla clarifying-question baseline.

One LLM call per query with a generic "ask a clarifying question" system
prompt. This is the floor D2C must beat to justify its multi-agent
machinery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from d2c.llm import LLMClient
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER

logger = logging.getLogger(__name__)


VANILLA_SCHEMA = {
    "type": "object",
    "properties": {
        "clarifying_question": {"type": "string", "maxLength": 200},
    },
    "required": ["clarifying_question"],
}


@dataclass
class VanillaResult:
    query: str
    clarifying_question: str
    raw: str
    format_failed: bool = False

    def to_dict(self) -> dict:
        return {
            "clarifying_question": self.clarifying_question,
            "raw": self.raw,
            "format_failed": self.format_failed,
        }


def run_vanilla_cqg(
    query: str,
    llm: LLMClient,
    max_tokens: int = 300,
) -> VanillaResult:
    """One-shot vanilla clarifying-question generation."""
    user = VANILLA_CQG_USER.format(query=query)
    raw = llm.chat(
        system_prompt=VANILLA_CQG_SYSTEM,
        user_prompt=user,
        temperature=0.3,
        max_tokens=max_tokens,
        format_schema=VANILLA_SCHEMA,
    )
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a dict")
        return VanillaResult(
            query=query,
            clarifying_question=str(data.get("clarifying_question", "")).strip(),
            raw=raw,
            format_failed=False,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Vanilla parse failed: %s | raw=%r", e, raw[:200])
        return VanillaResult(
            query=query,
            clarifying_question=raw.strip(),
            raw=raw,
            format_failed=True,
        )
