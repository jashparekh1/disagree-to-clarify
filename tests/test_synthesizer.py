"""Synthesizer transcript + JSON parsing tests."""

import json
import unittest
from unittest.mock import MagicMock

from d2c.agents import AgentResponse, AgentRole, Stance
from d2c.dialogue import DialogueResult
from d2c.synthesizer import (
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
        assumptions="a",
        answer_type="t",
        disagreements="d",
        stance=stance,
        stance_reason=reason,
    )


class TestTranscriptFormatting(unittest.TestCase):
    def test_round_zero_omits_stance(self):
        rounds = [
            [
                _resp(AgentRole.LOCUTIONARY, 0),
                _resp(AgentRole.ILLOCUTIONARY, 0),
                _resp(AgentRole.PERLOCUTIONARY, 0),
            ]
        ]
        d = DialogueResult(query="q", rounds=rounds, num_rounds=1)
        text = _format_transcript(d)
        self.assertNotIn("STANCE:", text)

    def test_later_rounds_include_stance_and_reason(self):
        rounds = [
            [_resp(AgentRole.LOCUTIONARY, 0)],
            [_resp(AgentRole.LOCUTIONARY, 1, stance=Stance.HOLD, reason="lex ambiguity remains")],
        ]
        d = DialogueResult(query="q", rounds=rounds, num_rounds=2)
        text = _format_transcript(d)
        self.assertIn("STANCE: HOLD", text)
        self.assertIn("lex ambiguity remains", text)

    def test_convergence_note_appended(self):
        rounds = [
            [_resp(AgentRole.LOCUTIONARY, 0)],
            [_resp(AgentRole.LOCUTIONARY, 1, stance=Stance.CONCEDE, reason="yielded")],
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

    def test_no_convergence_note_when_not_converged(self):
        rounds = [
            [_resp(AgentRole.LOCUTIONARY, 0)],
            [_resp(AgentRole.LOCUTIONARY, 1, stance=Stance.HOLD)],
        ]
        d = DialogueResult(query="q", rounds=rounds, num_rounds=2)
        text = _format_transcript(d)
        self.assertNotIn("converged", text.lower())


class TestSynthesizerJsonParsing(unittest.TestCase):
    def test_clean_json_parses(self):
        raw = json.dumps(
            {
                "key_disagreement": "illocutionary force",
                "clarifying_question": "Are you asking for help debugging or for a tutorial?",
            }
        )
        r = _parse_synthesizer_json(raw)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.key_disagreement, "illocutionary force")
        self.assertIn("debugging", r.clarifying_question)

    def test_fenced_json_parses(self):
        raw = '```json\n{"key_disagreement": "kd", "clarifying_question": "cq?"}\n```'
        r = _parse_synthesizer_json(raw)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.clarifying_question, "cq?")

    def test_invalid_json_falls_back_with_format_failed(self):
        raw = "KEY_DISAGREEMENT: x\nCLARIFYING_QUESTION: y"
        r = _parse_synthesizer_json(raw)
        self.assertTrue(r.format_failed)
        # Raw gets stashed as the question so the pipeline still outputs something.
        self.assertIn("KEY_DISAGREEMENT", r.clarifying_question)


class TestSynthesizeRetry(unittest.TestCase):
    def _valid(self) -> str:
        return json.dumps(
            {"key_disagreement": "kd", "clarifying_question": "cq?"}
        )

    def _make_dialogue(self) -> DialogueResult:
        return DialogueResult(
            query="q",
            rounds=[[_resp(AgentRole.LOCUTIONARY, 0)]],
            num_rounds=1,
        )

    def test_first_call_ok_no_retry(self):
        llm = MagicMock()
        llm.chat.return_value = self._valid()
        result = synthesize("q", self._make_dialogue(), llm)
        self.assertFalse(result.format_failed)
        self.assertEqual(llm.chat.call_count, 1)

    def test_first_fails_retry_succeeds(self):
        llm = MagicMock()
        llm.chat.side_effect = ["not valid json", self._valid()]
        result = synthesize("q", self._make_dialogue(), llm)
        self.assertFalse(result.format_failed)
        self.assertEqual(llm.chat.call_count, 2)

    def test_both_fail_returns_fallback(self):
        llm = MagicMock()
        llm.chat.side_effect = ["garbage one", "garbage two"]
        result = synthesize("q", self._make_dialogue(), llm)
        self.assertTrue(result.format_failed)
        self.assertEqual(llm.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
