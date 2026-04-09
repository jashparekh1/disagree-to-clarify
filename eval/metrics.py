"""Evaluation metrics for the D2C system."""

import json
import logging

from d2c.llm import LLMClient
from d2c.prompts import JUDGE_SYSTEM, JUDGE_USER

logger = logging.getLogger(__name__)


def semantic_similarity(generated: str, reference: str) -> float:
    """Compute semantic similarity between generated and reference clarifying question."""
    # Stub for Akash
    return 0.0


def llm_judge_score(
    query: str, interpretations: list[str], clarifying_question: str, llm: LLMClient
) -> dict:
    """Use LLM-as-judge to score clarifying question quality."""
    formatted_ints = "\n".join(f"- {i}" for i in interpretations)
    user_prompt = JUDGE_USER.format(
        query=query, interpretations=formatted_ints, clarifying_question=clarifying_question
    )

    try:
        raw = llm.chat(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,  # Zero for consistency in eval
        )
        # Attempt to parse JSON from the response
        # Sometimes LLMs wrap JSON in backticks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        return json.loads(raw)
    except Exception as e:
        logger.error(f"LLM judge failed: {e}. Raw response: {raw if 'raw' in locals() else 'N/A'}")
        return {
            "score": 1,
            "reasoning": f"Failed to parse LLM response: {e}",
            "covers_interpretations": False,
        }
