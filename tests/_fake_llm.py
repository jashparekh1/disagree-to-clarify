"""Test helper: scripted fake LLM client.

Matches the LLMClient.chat signature and returns pre-canned responses keyed by
role marker found in the system prompt. Each role's response list is consumed
in order (one per round).
"""

from __future__ import annotations

import re


# Use word-boundary patterns: LOCUTIONARY is a suffix of ILLOCUTIONARY and
# PERLOCUTIONARY, so naive substring matching mis-routes two of three SAT
# prompts to the LOCUTIONARY queue.
_ROLE_PATTERNS = [
    ("LITERALIST", re.compile(r"\bLITERALIST\b")),
    ("INTENT SEEKER", re.compile(r"\bINTENT SEEKER\b")),
    ("SCOPE EXPANDER", re.compile(r"\bSCOPE EXPANDER\b")),
    ("LOCUTIONARY", re.compile(r"\bLOCUTIONARY\b")),
    ("ILLOCUTIONARY", re.compile(r"\bILLOCUTIONARY\b")),
    ("PERLOCUTIONARY", re.compile(r"\bPERLOCUTIONARY\b")),
]


class ScriptedLLM:
    def __init__(self, responses_by_role: dict[str, list[str]]):
        self._remaining = {k: list(v) for k, v in responses_by_role.items()}
        self.model = "fake-model"
        self.calls: list[tuple[str, str, str]] = []  # (role, system, user)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        strip_thinking: bool = True,
    ) -> str:
        marker = self._match_role(system_prompt)
        queue = self._remaining.get(marker)
        if not queue:
            raise AssertionError(
                f"ScriptedLLM ran out of responses for {marker!r}"
            )
        response = queue.pop(0)
        self.calls.append((marker, system_prompt, user_prompt))
        return response

    def _match_role(self, system_prompt: str) -> str:
        matched = [name for name, pat in _ROLE_PATTERNS if pat.search(system_prompt)]
        if not matched:
            raise AssertionError(
                "No known role marker found in system_prompt; "
                "ScriptedLLM only knows agent roles."
            )
        if len(matched) > 1:
            raise AssertionError(
                f"Ambiguous role match in system_prompt: {matched!r}"
            )
        return matched[0]
