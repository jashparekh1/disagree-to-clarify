"""Dialogue loop + early-stop + format-failure aggregation tests."""

import json
import unittest

from d2c.agents import Agent, AgentRole
from d2c.dialogue import run_dialogue

from tests._fake_llm import ScriptedLLM


def _r0() -> str:
    return json.dumps({"interpretation": "initial reading"})


def _hold() -> str:
    return json.dumps(
        {
            "interpretation": "refined reading",
            "stance": "HOLD",
            "stance_reason": "my lens still sees something theirs don't",
        }
    )


def _concede() -> str:
    return json.dumps(
        {
            "interpretation": "yielded to other view",
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
                "LOCUTIONARY": [_r0(), _hold(), _hold()],
                "ILLOCUTIONARY": [_r0(), _hold(), _hold()],
                "PERLOCUTIONARY": [_r0(), _hold(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=3)
        self.assertEqual(result.rounds_completed, 3)
        self.assertFalse(result.converged)
        self.assertIsNone(result.converged_at_round)
        self.assertEqual(result.format_failure_rate, 0.0)

    def test_stops_early_when_all_agents_concede(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_r0(), _concede()],
                "ILLOCUTIONARY": [_r0(), _concede()],
                "PERLOCUTIONARY": [_r0(), _concede()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=5)
        self.assertEqual(result.rounds_completed, 2)
        self.assertTrue(result.converged)
        self.assertEqual(result.converged_at_round, 1)
        self.assertEqual(result.num_rounds, 5)

    def test_does_not_stop_on_partial_concede(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_r0(), _concede(), _concede()],
                "ILLOCUTIONARY": [_r0(), _hold(), _hold()],
                "PERLOCUTIONARY": [_r0(), _hold(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=3)
        self.assertEqual(result.rounds_completed, 3)
        self.assertFalse(result.converged)


class TestFormatFailureAggregation(unittest.TestCase):
    def test_zero_failures_gives_zero_rate(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_r0(), _hold()],
                "ILLOCUTIONARY": [_r0(), _hold()],
                "PERLOCUTIONARY": [_r0(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=2)
        self.assertEqual(result.format_failure_rate, 0.0)

    def test_rate_reflects_failed_responses(self):
        garbage = "not json"
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [garbage, _hold()],
                "ILLOCUTIONARY": [_r0(), _hold()],
                "PERLOCUTIONARY": [_r0(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=2)
        # 1 failure out of 6 responses.
        self.assertAlmostEqual(result.format_failure_rate, 1 / 6)


if __name__ == "__main__":
    unittest.main()
