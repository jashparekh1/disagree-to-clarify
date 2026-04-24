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
}

_ROLE_DISPLAY = {
    AgentRole.LITERALIST: "Literalist",
    AgentRole.INTENT_SEEKER: "Intent Seeker",
    AgentRole.SCOPE_EXPANDER: "Scope Expander",
    AgentRole.LOCUTIONARY: "Locutionary",
    AgentRole.ILLOCUTIONARY: "Illocutionary",
    AgentRole.PERLOCUTIONARY: "Perlocutionary",
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

# Length caps are enforced at the decoder level by Ollama's schema-constrained
# generation. Prose-level "1-2 sentences" hints were consistently ignored by
# qwen3:4b (rounds 0 often ran 2,000-6,000 chars). 400 chars ≈ 2 sentences.
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
            "raw_text": self.raw_text,
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
