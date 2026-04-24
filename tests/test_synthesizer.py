"""Synthesizer parsing + transcript tests."""

import json
import unittest
from unittest.mock import MagicMock

from d2c.agents import AgentResponse, AgentRole, Stance
from d2c.dialogue import DialogueResult
from d2c.synthesizer import (
    SYNTHESIZER_SCHEMA,
    _format_transcript,
    _parse_synthesizer_json,
    synthesize,
)


def _resp(role, round_num, stance=Stance.HOLD, reason=""):
    return AgentResponse(
        role=role,
        round_num=round_num,
        raw_text="",
        interpretation="i",
        stance=stance,
        stance_reason=reason,
    )


class TestTranscriptFormatting(unittest.TestCase):
    def test_round_zero_omits_stance(self):
        rounds = [[_resp(AgentRole.LOCUTIONARY, 0)]]
        d = DialogueResult(query="q", rounds=rounds, num_rounds=1)
        text = _format_transcript(d)
        self.assertNotIn("stance:", text)

    def test_later_rounds_include_stance(self):
        rounds = [
            [_resp(AgentRole.LOCUTIONARY, 0)],
            [_resp(AgentRole.LOCUTIONARY, 1, stance=Stance.HOLD, reason="still divergent")],
        ]
        d = DialogueResult(query="q", rounds=rounds, num_rounds=2)
        text = _format_transcript(d)
        self.assertIn("HOLD", text)
        self.assertIn("still divergent", text)

    def test_convergence_note_appended(self):
        rounds = [
            [_resp(AgentRole.LOCUTIONARY, 0)],
            [_resp(AgentRole.LOCUTIONARY, 1, stance=Stance.CONCEDE)],
        ]
        d = DialogueResult(
            query="q",
            rounds=rounds,
            num_rounds=3,
            converged=True,
            converged_at_round=1,
        )
        text = _format_transcript(d)
        self.assertIn("converged at round 1", text)


class TestSynthesizerParsing(unittest.TestCase):
    def test_valid_json_parses(self):
        raw = json.dumps({"clarifying_question": "which aspect: A or B?"})
        r = _parse_synthesizer_json(raw)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.clarifying_question, "which aspect: A or B?")

    def test_invalid_json_flags_failure(self):
        r = _parse_synthesizer_json("not json")
        self.assertTrue(r.format_failed)
        self.assertEqual(r.clarifying_question, "not json")


class TestSynthesizeUsesSchema(unittest.TestCase):
    def test_passes_schema_to_llm(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps({"clarifying_question": "cq?"})
        dlg = DialogueResult(
            query="q",
            rounds=[[_resp(AgentRole.LOCUTIONARY, 0)]],
            num_rounds=1,
        )
        result = synthesize("q", dlg, llm)
        self.assertFalse(result.format_failed)
        # Check that format_schema was passed through.
        _, kwargs = llm.chat.call_args
        self.assertEqual(kwargs["format_schema"], SYNTHESIZER_SCHEMA)

    def test_schema_requires_only_clarifying_question(self):
        self.assertEqual(
            SYNTHESIZER_SCHEMA["required"], ["clarifying_question"]
        )

    def test_clarifying_question_has_max_length(self):
        # Enforced at the decoder so the synthesizer can't dump a
        # multi-sentence paragraph instead of a clarifying question.
        self.assertIn(
            "maxLength",
            SYNTHESIZER_SCHEMA["properties"]["clarifying_question"],
        )
        self.assertLessEqual(
            SYNTHESIZER_SCHEMA["properties"]["clarifying_question"]["maxLength"],
            250,
        )


if __name__ == "__main__":
    unittest.main()
