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
    ALEATORIC_AGENT_SYSTEM,
    CLARIFIER_AGENT_SYSTEM,
    CLASSIFIER_AGENT_SYSTEM,
    CRITIC_AGENT_SYSTEM,
    ENTITY_GENERATOR_SYSTEM,
    EPISTEMIC_AGENT_SYSTEM,
    FACET_FINDER_SYSTEM,
    FACT_FINDER_SYSTEM,
    ILLOCUTIONARY_SYSTEM,
    INTENT_FINDER_SYSTEM,
    INTENT_GENERATOR_SYSTEM,
    INTENT_SEEKER_SYSTEM,
    LEXICAL_AGENT_SYSTEM,
    LITERALIST_SYSTEM,
    LOCUTIONARY_SYSTEM,
    ORACLE_AGENT_SYSTEM,
    PERLOCUTIONARY_SYSTEM,
    ROUND_N_FORMAT,
    ROUND_ZERO_USER_SUFFIX,
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
    FACT_FINDER = "fact_finder"
    FACET_FINDER = "facet_finder"
    INTENT_FINDER = "intent_finder"
    LEXICAL = "lexical"
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    CLASSIFIER = "classifier"
    INTENT_GEN = "intent_gen"
    ENTITY_GEN = "entity_gen"
    ORACLE = "oracle"
    CRITIC = "critic"
    CLARIFIER = "clarifier"


class Stance(Enum):
    HOLD = "HOLD"
    UPDATE = "UPDATE"
    CONCEDE = "CONCEDE"


_ROLE_TO_SYSTEM_PROMPT = {
    AgentRole.LITERALIST: LITERALIST_SYSTEM,
    AgentRole.INTENT_SEEKER: INTENT_SEEKER_SYSTEM,
    AgentRole.SCOPE_EXPANDER: SCOPE_EXPANDER_SYSTEM,
    AgentRole.LOCUTIONARY: LOCUTIONARY_SYSTEM,
    AgentRole.ILLOCUTIONARY: ILLOCUTIONARY_SYSTEM,
    AgentRole.PERLOCUTIONARY: PERLOCUTIONARY_SYSTEM,
    AgentRole.FACT_FINDER: FACT_FINDER_SYSTEM,
    AgentRole.FACET_FINDER: FACET_FINDER_SYSTEM,
    AgentRole.INTENT_FINDER: INTENT_FINDER_SYSTEM,
    AgentRole.LEXICAL: LEXICAL_AGENT_SYSTEM,
    AgentRole.ALEATORIC: ALEATORIC_AGENT_SYSTEM,
    AgentRole.EPISTEMIC: EPISTEMIC_AGENT_SYSTEM,
    AgentRole.CLASSIFIER: CLASSIFIER_AGENT_SYSTEM,
    AgentRole.INTENT_GEN: INTENT_GENERATOR_SYSTEM,
    AgentRole.ENTITY_GEN: ENTITY_GENERATOR_SYSTEM,
    AgentRole.ORACLE: ORACLE_AGENT_SYSTEM,
    AgentRole.CRITIC: CRITIC_AGENT_SYSTEM,
    AgentRole.CLARIFIER: CLARIFIER_AGENT_SYSTEM,
}

_ROLE_DISPLAY = {
    AgentRole.LITERALIST: "Literalist",
    AgentRole.INTENT_SEEKER: "Intent Seeker",
    AgentRole.SCOPE_EXPANDER: "Scope Expander",
    AgentRole.LOCUTIONARY: "Locutionary",
    AgentRole.ILLOCUTIONARY: "Illocutionary",
    AgentRole.PERLOCUTIONARY: "Perlocutionary",
    AgentRole.FACT_FINDER: "Fact Finder",
    AgentRole.FACET_FINDER: "Facet Finder",
    AgentRole.INTENT_FINDER: "Intent Finder",
    AgentRole.LEXICAL: "Lexical Agent",
    AgentRole.ALEATORIC: "Aleatoric Agent",
    AgentRole.EPISTEMIC: "Epistemic Agent",
    AgentRole.CLASSIFIER: "Classifier",
    AgentRole.INTENT_GEN: "Intent Generator",
    AgentRole.ENTITY_GEN: "Entity Generator",
    AgentRole.ORACLE: "Oracle",
    AgentRole.CRITIC: "Critic",
    AgentRole.CLARIFIER: "Clarifier",
}


_STANCE_ALIASES = {
    "hold": Stance.HOLD,
    "update": Stance.UPDATE,
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
        "stance_reason": {"type": "string", "maxLength": 250},
        "stance": {"type": "string", "enum": ["HOLD", "UPDATE", "CONCEDE"]},
    },
    "required": ["interpretation", "stance_reason", "stance"],
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
# Per-round user-turn builder
# ---------------------------------------------------------------------------

_STANCE_INSTRUCTIONS = """\
Decide your stance using this decision tree:
1. Can you name a specific gap or ambiguity that NO other agent has captured? → HOLD (name the gap explicitly in stance_reason)
2. Have the others partially shifted your view but not fully? → UPDATE (state what changed)
3. Does another agent's reading cover EXACTLY what you see, with nothing left out? → CONCEDE (name which agent and why their reading is complete)

Default to HOLD unless you can point to a specific agent whose reading already captures your exact concern.\
"""


def _format_others_turn(
    latest_round: list[AgentResponse],
    own_role: AgentRole,
    conceded_roles: set,
) -> str:
    """Build the user-turn content showing other agents' latest responses."""
    parts = ["Other agents' latest readings:"]
    for resp in latest_round:
        if resp.role == own_role:
            continue
        if resp.role in conceded_roles:
            continue
        label = _ROLE_DISPLAY[resp.role]
        reason = f" — {resp.stance_reason}" if resp.stance_reason else ""
        parts.append(f"[{label}] {resp.stance.value}{reason}: {resp.interpretation}")

    dropped = [_ROLE_DISPLAY[r] for r in conceded_roles if r != own_role]
    if dropped:
        parts.append(f"\n(Dropped out — conceded previously: {', '.join(dropped)})")

    parts.append(f"\n{_STANCE_INSTRUCTIONS}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, role: AgentRole, llm: LLMClient, max_tokens: int = 2048):
        self.role = role
        self.llm = llm
        self.system_prompt = _ROLE_TO_SYSTEM_PROMPT[role]
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def respond_initial(self, query: str) -> AgentResponse:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": query + ROUND_ZERO_USER_SUFFIX},
        ]
        raw = self.llm.chat_with_messages(
            messages=self.messages,
            temperature=0.7,
            max_tokens=self.max_tokens,
            format_schema=ROUND_ZERO_SCHEMA,
        )
        self.messages.append({"role": "assistant", "content": raw})
        return _parse_agent_json(raw, self.role, round_num=0)

    def respond_dialogue(
        self,
        query: str,
        all_prior_rounds: list[list[AgentResponse]],
        round_num: int,
        conceded_roles: set | None = None,
    ) -> AgentResponse:
        others_text = _format_others_turn(
            all_prior_rounds[-1], self.role, conceded_roles or set()
        )
        self.messages.append({"role": "user", "content": others_text + "\n\n" + ROUND_N_FORMAT})
        raw = self.llm.chat_with_messages(
            messages=self.messages,
            temperature=0.7,
            max_tokens=self.max_tokens,
            format_schema=ROUND_N_SCHEMA,
        )
        self.messages.append({"role": "assistant", "content": raw})
        return _parse_agent_json(raw, self.role, round_num=round_num)
