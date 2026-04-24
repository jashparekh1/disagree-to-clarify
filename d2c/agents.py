"""Interpretive agents for the D2C clarification policy.

Each agent applies a distinct reading lens to a user turn. Agents are not
adversaries competing to win an argument; they are independent readers whose
divergence signals where the user–system common ground is thin.
"""

from __future__ import annotations

import json
import logging
import re
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

# Retry nudge appended to the user turn when the first response doesn't parse
# as JSON. Kept in this module (not prompts.py) because it's coupled to the
# retry mechanism rather than to agent content.
_JSON_RETRY_NUDGE = (
    "\n\nYour previous response did not parse as valid JSON. Return ONLY a "
    "single JSON object matching the schema — no prose before or after, no "
    "markdown fences, no <think> blocks. The response must start with '{' "
    "and end with '}'."
)

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
    format_failed: bool = False  # True if JSON parsing failed even after retry

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
            "format_failed": self.format_failed,
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
# JSON parsing
# ---------------------------------------------------------------------------

# Matches a fenced code block like ```json\n{...}\n``` or ```\n{...}\n```.
_MD_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_blob(raw: str) -> str | None:
    """Best-effort extraction of a JSON object from an LLM response.

    Handles three common patterns seen with local models:
      1. Clean ``{...}`` response.
      2. Response wrapped in a markdown fence (```json ... ```).
      3. Prose prefix/suffix around a JSON object.

    Returns the candidate substring or None if nothing plausible is found.
    """
    if not raw:
        return None
    # 1. Markdown fence wins if present — some models emit fenced JSON with
    #    commentary after the closing fence, so we grab inside the fence first.
    fence = _MD_FENCE_RE.search(raw)
    if fence:
        inside = fence.group(1).strip()
        if inside.startswith("{"):
            return inside
    # 2. Fall back to first '{' .. matching last '}'.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return None


def _parse_agent_json(
    raw: str, role: AgentRole, round_num: int
) -> AgentResponse:
    """Parse the LLM response as JSON and build an AgentResponse.

    Sets `format_failed=True` on the returned response if the payload can't be
    coerced into the expected schema. The caller decides whether to retry.
    """
    blob = _extract_json_blob(raw)
    if blob is None:
        return _make_failed_response(raw, role, round_num)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return _make_failed_response(raw, role, round_num)
    if not isinstance(data, dict):
        return _make_failed_response(raw, role, round_num)

    return AgentResponse(
        role=role,
        round_num=round_num,
        raw_text=raw,
        interpretation=str(data.get("interpretation", "")).strip(),
        assumptions=str(data.get("assumptions", "")).strip(),
        answer_type=str(data.get("answer_type", "")).strip(),
        disagreements=str(data.get("disagreements", "")).strip(),
        stance=_parse_stance(str(data.get("stance", ""))),
        stance_reason=str(data.get("stance_reason", "")).strip(),
        format_failed=False,
    )


def _make_failed_response(
    raw: str, role: AgentRole, round_num: int
) -> AgentResponse:
    """Last-resort fallback when JSON parsing fails.

    We stash the raw text in ``interpretation`` so the synthesizer still has
    something to read, and flag ``format_failed`` so the pipeline can report
    the failure rate as a first-class metric.
    """
    return AgentResponse(
        role=role,
        round_num=round_num,
        raw_text=raw,
        interpretation=raw.strip(),
        assumptions="",
        answer_type="",
        disagreements="",
        stance=Stance.HOLD,
        stance_reason="",
        format_failed=True,
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

    def _chat_with_json_retry(
        self, user_prompt: str, round_num: int
    ) -> AgentResponse:
        """Call the LLM, parse as JSON; on parse failure, retry once with a
        strict reminder at a lower temperature. If both attempts fail, return
        the first attempt's fallback response with ``format_failed=True``.
        """
        raw = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=self.max_tokens,
        )
        resp = _parse_agent_json(raw, self.role, round_num)
        if not resp.format_failed:
            return resp

        logger.warning(
            "Agent %s round %d: JSON parse failed, retrying once",
            self.role.value,
            round_num,
        )
        raw_retry = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt + _JSON_RETRY_NUDGE,
            temperature=0.2,  # Tighter sampling on retry to curb drift.
            max_tokens=self.max_tokens,
        )
        resp_retry = _parse_agent_json(raw_retry, self.role, round_num)
        if not resp_retry.format_failed:
            return resp_retry

        logger.warning(
            "Agent %s round %d: JSON parse failed after retry; using fallback",
            self.role.value,
            round_num,
        )
        return resp  # Return the first attempt's fallback (format_failed=True).

    def respond_initial(self, query: str) -> AgentResponse:
        """Round 0: agent sees only the query."""
        return self._chat_with_json_retry(user_prompt=query, round_num=0)

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
        return self._chat_with_json_retry(
            user_prompt=user_prompt, round_num=round_num
        )
