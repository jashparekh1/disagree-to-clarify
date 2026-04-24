"""Synthesizer: turn divergent readings into a single grounding move.

Reads the multi-round interpretation transcript, identifies the most
consequential grounding gap (the divergence whose resolution would most change
the appropriate response), and emits one clarifying question directed at the
user to close that gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from d2c.agents import AgentResponse, _ROLE_DISPLAY
from d2c.dialogue import DialogueResult
from d2c.llm import LLMClient
from d2c.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER

logger = logging.getLogger(__name__)


@dataclass
class SynthesizerResult:
    key_disagreement: str
    clarifying_question: str
    raw_text: str

    def to_dict(self) -> dict:
        return {
            "key_disagreement": self.key_disagreement,
            "clarifying_question": self.clarifying_question,
            "raw_text": self.raw_text,
        }


def _format_transcript(dialogue: DialogueResult) -> str:
    """Format the full dialogue as a human-readable transcript.

    Stance is included for rounds 1+ (round 0 has nothing to concede to).
    A convergence note is appended if the dialogue early-stopped.
    """
    parts: list[str] = []
    for round_idx, rnd in enumerate(dialogue.rounds):
        parts.append(f"=== Round {round_idx} ===")
        for resp in rnd:
            block = (
                f"[{_ROLE_DISPLAY[resp.role]}]\n"
                f"INTERPRETATION: {resp.interpretation}\n"
                f"ASSUMPTIONS: {resp.assumptions}\n"
                f"ANSWER_TYPE: {resp.answer_type}\n"
                f"DISAGREEMENTS: {resp.disagreements}"
            )
            if round_idx > 0:
                block += f"\nSTANCE: {resp.stance.value.upper()}"
                if resp.stance_reason:
                    block += f"\nSTANCE_REASON: {resp.stance_reason}"
            parts.append(block)
        parts.append("")
    if dialogue.converged:
        parts.append(
            f"[Dialogue converged at round {dialogue.converged_at_round}: "
            "all agents CONCEDEd.]"
        )
    return "\n".join(parts)


def _parse_synthesizer(raw: str) -> SynthesizerResult:
    """Parse KEY_DISAGREEMENT and CLARIFYING_QUESTION from synthesizer output."""
    key_dis = ""
    clarifying_q = ""

    kd_marker = "KEY_DISAGREEMENT:"
    cq_marker = "CLARIFYING_QUESTION:"

    kd_start = raw.find(kd_marker)
    cq_start = raw.find(cq_marker)

    if kd_start != -1:
        kd_end = cq_start if cq_start > kd_start else len(raw)
        key_dis = raw[kd_start + len(kd_marker) : kd_end].strip()

    if cq_start != -1:
        clarifying_q = raw[cq_start + len(cq_marker) :].strip()

    # Fallback: if we couldn't parse, use the whole raw text as the question
    if not clarifying_q:
        clarifying_q = raw.strip()

    return SynthesizerResult(
        key_disagreement=key_dis,
        clarifying_question=clarifying_q,
        raw_text=raw,
    )


def synthesize(
    query: str,
    dialogue: DialogueResult,
    llm: LLMClient,
    max_tokens: int = 300,
) -> SynthesizerResult:
    """Format dialogue transcript, call synthesizer LLM, parse output."""
    transcript = _format_transcript(dialogue)
    user_prompt = SYNTHESIZER_USER.format(query=query, transcript=transcript)

    raw = llm.chat(
        system_prompt=SYNTHESIZER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return _parse_synthesizer(raw)
