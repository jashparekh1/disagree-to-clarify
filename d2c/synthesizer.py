"""Synthesizer: turn divergent readings into a single clarifying question.

Uses Ollama structured outputs so the JSON is always valid.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from d2c.agents import _ROLE_DISPLAY
from d2c.dialogue import DialogueResult
from d2c.llm import LLMClient
from d2c.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER

logger = logging.getLogger(__name__)


# maxLength ~ 200 chars ≈ 30 words. Enforces "under 25 words, just the
# question" at the decoder level; prose-level hints were unreliable.
SYNTHESIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "clarifying_question": {"type": "string", "maxLength": 200},
    },
    "required": ["clarifying_question"],
}


@dataclass
class SynthesizerResult:
    clarifying_question: str
    raw_text: str
    format_failed: bool = False
    # Retained for backward-compat with older output JSONL; not populated.
    key_disagreement: str = ""

    def to_dict(self) -> dict:
        return {
            "clarifying_question": self.clarifying_question,
            "raw_text": self.raw_text,
            "format_failed": self.format_failed,
        }


def _format_transcript(dialogue: DialogueResult) -> str:
    """Short human-readable transcript for the synthesizer."""
    parts: list[str] = []
    for round_idx, rnd in enumerate(dialogue.rounds):
        parts.append(f"=== Round {round_idx} ===")
        for resp in rnd:
            line = f"[{_ROLE_DISPLAY[resp.role]}] {resp.interpretation}"
            if round_idx > 0:
                line += f" (stance: {resp.stance.value}"
                if resp.stance_reason:
                    line += f" — {resp.stance_reason}"
                line += ")"
            parts.append(line)
        parts.append("")
    if dialogue.converged:
        parts.append(
            f"[Dialogue converged at round {dialogue.converged_at_round}: "
            "all agents CONCEDEd.]"
        )
    return "\n".join(parts)


def _parse_synthesizer_json(raw: str) -> SynthesizerResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return SynthesizerResult(
            clarifying_question=raw.strip(),
            raw_text=raw,
            format_failed=True,
        )
    if not isinstance(data, dict):
        return SynthesizerResult(
            clarifying_question=raw.strip(),
            raw_text=raw,
            format_failed=True,
        )
    return SynthesizerResult(
        clarifying_question=str(data.get("clarifying_question", "")).strip(),
        raw_text=raw,
        format_failed=False,
    )


def synthesize(
    query: str,
    dialogue: DialogueResult,
    llm: LLMClient,
    max_tokens: int = 300,
) -> SynthesizerResult:
    transcript = _format_transcript(dialogue)
    user_prompt = SYNTHESIZER_USER.format(query=query, transcript=transcript)

    raw = llm.chat(
        system_prompt=SYNTHESIZER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=max_tokens,
        format_schema=SYNTHESIZER_SCHEMA,
    )
    return _parse_synthesizer_json(raw)
