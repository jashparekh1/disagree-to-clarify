"""Interpretive agents for the D2C clarification policy.

Each agent produces a short JSON reading of the query. Structured outputs
(Ollama `format` field) constrain generation to valid JSON — no more
prompt-based JSON begging, no retry loop, no format failures in practice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from d2c.llm import LLMClient
from d2c.prompts import (
    DIALOGUE_ROUND_USER,
    ILLOCUTIONARY_SYSTEM,
    INTENT_SEEKER_SYSTEM,
    LITERALIST_SYSTEM,
    LOCUTIONARY_SYSTEM,
    PERLOCUTIONARY_SYSTEM,
    SCOPE_EXPANDER_SYSTEM,
    LOCUTIONARY_IMPROVED_SYSTEM,
    ILLOCUTIONARY_IMPROVED_SYSTEM,
    PERLOCUTIONARY_IMPROVED_SYSTEM,
    LOCUTIONARY_CLEAR_SYSTEM,
    ILLOCUTIONARY_CLEAR_SYSTEM,
    PERLOCUTIONARY_CLEAR_SYSTEM,
    LOCUTIONARY_HYBRID_SYSTEM,
    ILLOCUTIONARY_HYBRID_SYSTEM,
    PERLOCUTIONARY_HYBRID_SYSTEM,
    LOCUTIONARY_SURGICAL_SYSTEM,
    ILLOCUTIONARY_SURGICAL_SYSTEM,
    PERLOCUTIONARY_SURGICAL_SYSTEM,
    WORD_MEANING_SYSTEM,
    USER_GOAL_SYSTEM,
    MISSING_DETAILS_SYSTEM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role + stance enums
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    LITERALIST = "literalist"
    INTENT_SEEKER = "intent_seeker"
    SCOPE_EXPANDER = "scope_expander"
    LOCUTIONARY = "locutionary"
    ILLOCUTIONARY = "illocutionary"
    PERLOCUTIONARY = "perlocutionary"
    LOCUTIONARY_IMPROVED = "locutionary_improved"
    ILLOCUTIONARY_IMPROVED = "illocutionary_improved"
    PERLOCUTIONARY_IMPROVED = "perlocutionary_improved"
    LOCUTIONARY_CLEAR = "locutionary_clear"
    ILLOCUTIONARY_CLEAR = "illocutionary_clear"
    PERLOCUTIONARY_CLEAR = "perlocutionary_clear"
    LOCUTIONARY_HYBRID = "locutionary_hybrid"
    ILLOCUTIONARY_HYBRID = "illocutionary_hybrid"
    PERLOCUTIONARY_HYBRID = "perlocutionary_hybrid"
    LOCUTIONARY_SURGICAL = "locutionary_surgical"
    ILLOCUTIONARY_SURGICAL = "illocutionary_surgical"
    PERLOCUTIONARY_SURGICAL = "perlocutionary_surgical"
    WORD_MEANING = "word_meaning"
    USER_GOAL = "user_goal"
    MISSING_DETAILS = "missing_details"


class Stance(Enum):
    HOLD = "HOLD"
    CONCEDE = "CONCEDE"


_ROLE_TO_SYSTEM_PROMPT = {
    AgentRole.LITERALIST: LITERALIST_SYSTEM,
    AgentRole.INTENT_SEEKER: INTENT_SEEKER_SYSTEM,
    AgentRole.SCOPE_EXPANDER: SCOPE_EXPANDER_SYSTEM,
    AgentRole.LOCUTIONARY: LOCUTIONARY_SYSTEM,
    AgentRole.ILLOCUTIONARY: ILLOCUTIONARY_SYSTEM,
    AgentRole.PERLOCUTIONARY: PERLOCUTIONARY_SYSTEM,
    AgentRole.LOCUTIONARY_IMPROVED: LOCUTIONARY_IMPROVED_SYSTEM,
    AgentRole.ILLOCUTIONARY_IMPROVED: ILLOCUTIONARY_IMPROVED_SYSTEM,
    AgentRole.PERLOCUTIONARY_IMPROVED: PERLOCUTIONARY_IMPROVED_SYSTEM,
    AgentRole.LOCUTIONARY_CLEAR: LOCUTIONARY_CLEAR_SYSTEM,
    AgentRole.ILLOCUTIONARY_CLEAR: ILLOCUTIONARY_CLEAR_SYSTEM,
    AgentRole.PERLOCUTIONARY_CLEAR: PERLOCUTIONARY_CLEAR_SYSTEM,
    AgentRole.LOCUTIONARY_HYBRID: LOCUTIONARY_HYBRID_SYSTEM,
    AgentRole.ILLOCUTIONARY_HYBRID: ILLOCUTIONARY_HYBRID_SYSTEM,
    AgentRole.PERLOCUTIONARY_HYBRID: PERLOCUTIONARY_HYBRID_SYSTEM,
    AgentRole.LOCUTIONARY_SURGICAL: LOCUTIONARY_SURGICAL_SYSTEM,
    AgentRole.ILLOCUTIONARY_SURGICAL: ILLOCUTIONARY_SURGICAL_SYSTEM,
    AgentRole.PERLOCUTIONARY_SURGICAL: PERLOCUTIONARY_SURGICAL_SYSTEM,
    AgentRole.WORD_MEANING: WORD_MEANING_SYSTEM,
    AgentRole.USER_GOAL: USER_GOAL_SYSTEM,
    AgentRole.MISSING_DETAILS: MISSING_DETAILS_SYSTEM,
}

_ROLE_DISPLAY = {
    AgentRole.LITERALIST: "Literalist",
    AgentRole.INTENT_SEEKER: "Intent Seeker",
    AgentRole.SCOPE_EXPANDER: "Scope Expander",
    AgentRole.LOCUTIONARY: "Locutionary",
    AgentRole.ILLOCUTIONARY: "Illocutionary",
    AgentRole.PERLOCUTIONARY: "Perlocutionary",
    AgentRole.LOCUTIONARY_IMPROVED: "Locutionary (Improved)",
    AgentRole.ILLOCUTIONARY_IMPROVED: "Illocutionary (Improved)",
    AgentRole.PERLOCUTIONARY_IMPROVED: "Perlocutionary (Improved)",
    AgentRole.LOCUTIONARY_CLEAR: "Locutionary (Clear)",
    AgentRole.ILLOCUTIONARY_CLEAR: "Illocutionary (Clear)",
    AgentRole.PERLOCUTIONARY_CLEAR: "Perlocutionary (Clear)",
    AgentRole.LOCUTIONARY_HYBRID: "Locutionary (Hybrid)",
    AgentRole.ILLOCUTIONARY_HYBRID: "Illocutionary (Hybrid)",
    AgentRole.PERLOCUTIONARY_HYBRID: "Perlocutionary (Hybrid)",
    AgentRole.LOCUTIONARY_SURGICAL: "Locutionary (Surgical)",
    AgentRole.ILLOCUTIONARY_SURGICAL: "Illocutionary (Surgical)",
    AgentRole.PERLOCUTIONARY_SURGICAL: "Perlocutionary (Surgical)",
    AgentRole.WORD_MEANING: "Word Meaning Agent",
    AgentRole.USER_GOAL: "User Goal Agent",
    AgentRole.MISSING_DETAILS: "Missing Details Agent",
}


_STANCE_ALIASES = {
    "hold": Stance.HOLD,
    "concede": Stance.CONCEDE,
}


def _parse_stance(text: str) -> Stance:
    return _STANCE_ALIASES.get(text.strip().lower(), Stance.HOLD)


# ---------------------------------------------------------------------------
# JSON schemas for structured outputs.
# ---------------------------------------------------------------------------

ROUND_ZERO_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string", "maxLength": 400},
    },
    "required": ["interpretation"],
}

ROUND_N_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string", "maxLength": 400},
        "stance": {"type": "string", "enum": ["HOLD", "CONCEDE"]},
        "stance_reason": {"type": "string", "maxLength": 250},
    },
    "required": ["interpretation", "stance", "stance_reason"],
}


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    role: AgentRole
    round_num: int
    raw_text: str
    interpretation: str
    stance: Stance = Stance.HOLD
    stance_reason: str = ""
    format_failed: bool = False

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "round_num": self.round_num,
            "interpretation": self.interpretation,
            "stance": self.stance.value,
            "stance_reason": self.stance_reason,
            "format_failed": self.format_failed,
        }

    def format_for_others(self) -> str:
        """Short block other agents see in the next round."""
        base = f"[{_ROLE_DISPLAY[self.role]}] {self.interpretation}"
        if self.round_num > 0:
            base += f" (stance: {self.stance.value}"
            if self.stance_reason:
                base += f" — {self.stance_reason}"
            base += ")"
        return base


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_agent_json(
    raw: str, role: AgentRole, round_num: int
) -> AgentResponse:
    """Parse an agent response. With structured outputs, ``raw`` should be
    a clean JSON object. Defensive fallback if it isn't.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return AgentResponse(
            role=role,
            round_num=round_num,
            raw_text=raw,
            interpretation=raw.strip(),
            format_failed=True,
        )
    if not isinstance(data, dict):
        return AgentResponse(
            role=role,
            round_num=round_num,
            raw_text=raw,
            interpretation=raw.strip(),
            format_failed=True,
        )
    return AgentResponse(
        role=role,
        round_num=round_num,
        raw_text=raw,
        interpretation=str(data.get("interpretation", "")).strip(),
        stance=_parse_stance(str(data.get("stance", ""))),
        stance_reason=str(data.get("stance_reason", "")).strip(),
        format_failed=False,
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, role: AgentRole, llm: LLMClient, max_tokens: int = 2048):
        self.role = role
        self.llm = llm
        self.system_prompt = _ROLE_TO_SYSTEM_PROMPT[role]
        self.max_tokens = max_tokens

    def respond_initial(self, query: str) -> AgentResponse:
        raw = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=query,
            temperature=0.7,
            max_tokens=self.max_tokens,
            format_schema=ROUND_ZERO_SCHEMA,
        )
        return _parse_agent_json(raw, self.role, round_num=0)

    def respond_dialogue(
        self,
        query: str,
        other_responses: list[AgentResponse],
        round_num: int,
    ) -> AgentResponse:
        other_text = "\n".join(r.format_for_others() for r in other_responses)
        user_prompt = DIALOGUE_ROUND_USER.format(
            query=query,
            other_agent_responses=other_text,
        )
        raw = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=self.max_tokens,
            format_schema=ROUND_N_SCHEMA,
        )
        return _parse_agent_json(raw, self.role, round_num=round_num)
