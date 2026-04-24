"""Dialogue loop + early-stop + format-failure aggregation tests.

Uses a scripted fake LLM so nothing hits Ollama. Fixtures are JSON payloads
to match the new strict-JSON output contract.
"""

import json
import unittest

from d2c.agents import Agent, AgentRole
from d2c.dialogue import run_dialogue

from tests._fake_llm import ScriptedLLM


def _round_zero() -> str:
    return json.dumps(
        {
            "interpretation": "initial reading",
            "assumptions": "some context",
            "answer_type": "direct",
            "disagreements": "",
        }
    )


def _hold() -> str:
    return json.dumps(
        {
            "interpretation": "refined reading",
            "assumptions": "some context",
            "answer_type": "direct",
            "disagreements": "others miss X",
            "stance": "HOLD",
            "stance_reason": "my lens still sees something theirs don't",
        }
    )


def _concede() -> str:
    return json.dumps(
        {
            "interpretation": "yielded to other view",
            "assumptions": "same",
            "answer_type": "direct",
            "disagreements": "none remaining",
            "stance": "CONCEDE",
            "stance_reason": "the other agents' reading subsumes mine",
        }
    )


def _sat_agents(llm):
    return [
        Agent(AgentRole.LOCUTIONARY, llm),
        Agent(AgentRole.ILLOCUTIONARY, llm),
        Agent(AgentRole.PERLOCUTIONARY, llm),
    ]


class TestDialogueLoop(unittest.TestCase):
    def test_runs_full_budget_when_no_one_concedes(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _hold(), _hold()],
                "ILLOCUTIONARY": [_round_zero(), _hold(), _hold()],
                "PERLOCUTIONARY": [_round_zero(), _hold(), _hold()],
            }
        )
        result = run_dialogue("What is Python?", _sat_agents(llm), num_rounds=3)
        self.assertEqual(result.rounds_completed, 3)
        self.assertFalse(result.converged)
        self.assertIsNone(result.converged_at_round)
        self.assertEqual(result.format_failure_rate, 0.0)

    def test_stops_early_when_all_agents_concede(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _concede()],
                "ILLOCUTIONARY": [_round_zero(), _concede()],
                "PERLOCUTIONARY": [_round_zero(), _concede()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=5)
        self.assertEqual(result.rounds_completed, 2)
        self.assertTrue(result.converged)
        self.assertEqual(result.converged_at_round, 1)
        self.assertEqual(result.num_rounds, 5)  # Configured budget preserved.

    def test_does_not_stop_on_partial_concede(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _concede(), _concede()],
                "ILLOCUTIONARY": [_round_zero(), _hold(), _hold()],
                "PERLOCUTIONARY": [_round_zero(), _hold(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=3)
        self.assertEqual(result.rounds_completed, 3)
        self.assertFalse(result.converged)

    def test_round_zero_default_stance_is_hold(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _hold()],
                "ILLOCUTIONARY": [_round_zero(), _hold()],
                "PERLOCUTIONARY": [_round_zero(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=2)
        for resp in result.rounds[0]:
            self.assertEqual(resp.stance.value, "hold")


class TestFormatFailureAggregation(unittest.TestCase):
    def test_rate_reflects_failures_including_retries(self):
        # LOCUTIONARY: both attempts in round 0 fail, then round 1 valid.
        # Other agents: clean throughout.
        garbage = "not json at all"
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [garbage, garbage, _hold()],
                "ILLOCUTIONARY": [_round_zero(), _hold()],
                "PERLOCUTIONARY": [_round_zero(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=2)
        # Total responses: 3 agents × 2 rounds = 6. One failed (LOCUTIONARY r0).
        self.assertAlmostEqual(result.format_failure_rate, 1 / 6)
        # Round 0 LOCUTIONARY response carries the failure flag.
        locu_r0 = next(r for r in result.rounds[0] if r.role == AgentRole.LOCUTIONARY)
        self.assertTrue(locu_r0.format_failed)

    def test_zero_failures_gives_zero_rate(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _hold()],
                "ILLOCUTIONARY": [_round_zero(), _hold()],
                "PERLOCUTIONARY": [_round_zero(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=2)
        self.assertEqual(result.format_failure_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
