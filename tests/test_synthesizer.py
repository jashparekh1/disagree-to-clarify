"""Synthesizer transcript-formatting tests."""

import unittest

from d2c.agents import AgentResponse, AgentRole, Stance
from d2c.dialogue import DialogueResult
from d2c.synthesizer import _format_transcript


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


if __name__ == "__main__":
    unittest.main()
