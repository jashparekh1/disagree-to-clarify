"""Synthesizer: turn divergent readings into a single grounding move.

Reads the multi-round interpretation transcript, identifies the most
consequential grounding gap (the divergence whose resolution would most change
the appropriate response), and emits one clarifying question directed at the
user to close that gap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from d2c.agents import _ROLE_DISPLAY, _extract_json_blob
from d2c.dialogue import DialogueResult
from d2c.llm import LLMClient
from d2c.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER

logger = logging.getLogger(__name__)

_JSON_RETRY_NUDGE = (
    "\n\nYour previous response did not parse as valid JSON. Return ONLY a "
    "single JSON object matching the schema — no prose before or after, no "
    "markdown fences, no <think> blocks. The response must start with '{' "
    "and end with '}'."
)


@dataclass
class SynthesizerResult:
    key_disagreement: str
    clarifying_question: str
    raw_text: str
    format_failed: bool = False

    def to_dict(self) -> dict:
        return {
            "key_disagreement": self.key_disagreement,
            "clarifying_question": self.clarifying_question,
            "raw_text": self.raw_text,
            "format_failed": self.format_failed,
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


def _parse_synthesizer_json(raw: str) -> SynthesizerResult:
    """Parse the synthesizer's JSON response. On parse failure, stash the
    raw text in ``clarifying_question`` so the pipeline still produces *a*
    question, and flag ``format_failed`` so the failure can be reported.
    """
    blob = _extract_json_blob(raw)
    if blob is not None:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return SynthesizerResult(
                key_disagreement=str(data.get("key_disagreement", "")).strip(),
                clarifying_question=str(
                    data.get("clarifying_question", "")
                ).strip(),
                raw_text=raw,
                format_failed=False,
            )
    return SynthesizerResult(
        key_disagreement="",
        clarifying_question=raw.strip(),
        raw_text=raw,
        format_failed=True,
    )


def synthesize(
    query: str,
    dialogue: DialogueResult,
    llm: LLMClient,
    max_tokens: int = 300,
) -> SynthesizerResult:
    """Format dialogue transcript, call synthesizer LLM, parse output.

    Retries once with a strict JSON reminder if the first parse fails.
    """
    transcript = _format_transcript(dialogue)
    user_prompt = SYNTHESIZER_USER.format(query=query, transcript=transcript)

    raw = llm.chat(
        system_prompt=SYNTHESIZER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    result = _parse_synthesizer_json(raw)
    if not result.format_failed:
        return result

    logger.warning("Synthesizer: JSON parse failed, retrying once")
    raw_retry = llm.chat(
        system_prompt=SYNTHESIZER_SYSTEM,
        user_prompt=user_prompt + _JSON_RETRY_NUDGE,
        temperature=0.1,
        max_tokens=max_tokens,
    )
    result_retry = _parse_synthesizer_json(raw_retry)
    if not result_retry.format_failed:
        return result_retry

    logger.warning("Synthesizer: JSON parse failed after retry; using fallback")
    return result  # Return the first attempt's fallback (format_failed=True).
