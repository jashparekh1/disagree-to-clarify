"""Interpretive agents for the D2C clarification policy.

Each agent applies a distinct reading lens to a user turn. Agents are not
adversaries competing to win an argument; they are independent readers whose
divergence signals where the user–system common ground is thin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from d2c.llm import LLMClient
from d2c.prompts import (
    DIALOGUE_ROUND_USER,
    INTENT_SEEKER_SYSTEM,
    LITERALIST_SYSTEM,
    SCOPE_EXPANDER_SYSTEM,
    LOCUTIONARY_SYSTEM,
    ILLOCUTIONARY_SYSTEM,
    PERLOCUTIONARY_SYSTEM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role & stance enums
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    # Original D2C Roles
    LITERALIST = "literalist"
    INTENT_SEEKER = "intent_seeker"
    SCOPE_EXPANDER = "scope_expander"

    # Speech Act Theory Roles
    LOCUTIONARY = "locutionary"
    ILLOCUTIONARY = "illocutionary"
    PERLOCUTIONARY = "perlocutionary"


class Stance(Enum):
    """Agent's position on its own reading after seeing others'.

    HOLD = my lens still sees something theirs don't; keep the divergence.
    CONCEDE = another agent's reading supersedes mine; converge.
    """

    HOLD = "hold"
    CONCEDE = "concede"


_STANCE_ALIASES = {
    "hold": Stance.HOLD,
    "holding": Stance.HOLD,
    "concede": Stance.CONCEDE,
    "conceded": Stance.CONCEDE,
    "concedes": Stance.CONCEDE,
    "conceding": Stance.CONCEDE,
}


def _parse_stance(text: str) -> Stance:
    """Map free-form stance text to the enum. Unknown values default to HOLD.

    HOLD is the safe default because a spurious CONCEDE would falsely trigger
    early-stop and destroy the grounding-gap signal.
    """
    return _STANCE_ALIASES.get(text.strip().lower(), Stance.HOLD)


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
    AgentRole.LOCUTIONARY: "Locutionary Parser",
    AgentRole.ILLOCUTIONARY: "Illocutionary Analyst",
    AgentRole.PERLOCUTIONARY: "Perlocutionary Evaluator",
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
    assumptions: str
    answer_type: str
    disagreements: str
    stance: Stance = Stance.HOLD
    stance_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "round_num": self.round_num,
            "raw_text": self.raw_text,
            "interpretation": self.interpretation,
            "assumptions": self.assumptions,
            "answer_type": self.answer_type,
            "disagreements": self.disagreements,
            "stance": self.stance.value,
            "stance_reason": self.stance_reason,
        }

    def format_for_others(self) -> str:
        """Format this response so other agents can read it in the next round."""
        base = (
            f"[{_ROLE_DISPLAY[self.role]}]\n"
            f"INTERPRETATION: {self.interpretation}\n"
            f"ASSUMPTIONS: {self.assumptions}\n"
            f"ANSWER_TYPE: {self.answer_type}\n"
            f"DISAGREEMENTS: {self.disagreements}"
        )
        # Round 0 has no prior context, so stance doesn't apply yet.
        if self.round_num > 0:
            base += f"\nSTANCE: {self.stance.value.upper()}"
            if self.stance_reason:
                base += f"\nSTANCE_REASON: {self.stance_reason}"
        return base


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

_FIELDS = [
    "INTERPRETATION",
    "ASSUMPTIONS",
    "ANSWER_TYPE",
    "DISAGREEMENTS",
    "STANCE",
    "STANCE_REASON",
]


def _parse_response(raw: str, role: AgentRole, round_num: int) -> AgentResponse:
    """Parse structured fields from raw LLM output. Lenient — never crashes."""
    sections: dict[str, str] = {}
    for i, key in enumerate(_FIELDS):
        marker = f"{key}:"
        start = raw.find(marker)
        if start == -1:
            continue
        start += len(marker)
        # Find where the next section starts
        end = len(raw)
        for next_key in _FIELDS[i + 1 :]:
            next_start = raw.find(f"{next_key}:", start)
            if next_start != -1:
                end = next_start
                break
        sections[key] = raw[start:end].strip()

    return AgentResponse(
        role=role,
        round_num=round_num,
        raw_text=raw,
        interpretation=sections.get("INTERPRETATION", raw.strip()),
        assumptions=sections.get("ASSUMPTIONS", ""),
        answer_type=sections.get("ANSWER_TYPE", ""),
        disagreements=sections.get("DISAGREEMENTS", ""),
        stance=_parse_stance(sections.get("STANCE", "")),
        stance_reason=sections.get("STANCE_REASON", ""),
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
        """Round 0: agent sees only the query."""
        raw = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=query,
            temperature=0.7,
            max_tokens=self.max_tokens,
        )
        return _parse_response(raw, self.role, round_num=0)

    def respond_dialogue(
        self,
        query: str,
        other_responses: list[AgentResponse],
        round_num: int,
    ) -> AgentResponse:
        """Round 1+: agent sees query + other agents' previous responses."""
        other_text = "\n\n".join(r.format_for_others() for r in other_responses)
        user_prompt = DIALOGUE_ROUND_USER.format(
            query=query,
            other_agent_responses=other_text,
        )
        raw = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=self.max_tokens,
        )
        return _parse_response(raw, self.role, round_num=round_num)
