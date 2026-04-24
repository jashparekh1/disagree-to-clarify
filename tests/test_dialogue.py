"""Dialogue loop + early-stop tests.

Uses a scripted fake LLM so nothing hits Ollama.
"""

import unittest

from d2c.agents import Agent, AgentRole
from d2c.dialogue import run_dialogue

from tests._fake_llm import ScriptedLLM


def _round_zero():
    return (
        "INTERPRETATION: initial reading\n"
        "ASSUMPTIONS: some context\n"
        "ANSWER_TYPE: direct\n"
        "DISAGREEMENTS: \n"
    )


def _hold():
    return (
        "INTERPRETATION: refined reading\n"
        "ASSUMPTIONS: some context\n"
        "ANSWER_TYPE: direct\n"
        "DISAGREEMENTS: others miss X\n"
        "STANCE: HOLD\n"
        "STANCE_REASON: my lens still sees something theirs don't\n"
    )


def _concede():
    return (
        "INTERPRETATION: yielded to other view\n"
        "ASSUMPTIONS: same\n"
        "ANSWER_TYPE: direct\n"
        "DISAGREEMENTS: none remaining\n"
        "STANCE: CONCEDE\n"
        "STANCE_REASON: the other agents' reading subsumes mine\n"
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

    def test_stops_early_when_all_agents_concede(self):
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _concede()],
                "ILLOCUTIONARY": [_round_zero(), _concede()],
                "PERLOCUTIONARY": [_round_zero(), _concede()],
            }
        )
        # Budget is 5 rounds but all concede in round 1 → should stop.
        result = run_dialogue("q", _sat_agents(llm), num_rounds=5)
        self.assertEqual(result.rounds_completed, 2)  # round 0 + round 1
        self.assertTrue(result.converged)
        self.assertEqual(result.converged_at_round, 1)
        # Configured budget preserved for reporting/reproducibility.
        self.assertEqual(result.num_rounds, 5)

    def test_does_not_stop_on_partial_concede(self):
        # 1 concede + 2 hold should not trigger early-stop: the residual HOLDs
        # still encode grounding gaps the synthesizer needs.
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

    def test_round_zero_never_triggers_convergence(self):
        # Even if round-0 responses somehow parse as CONCEDE (they shouldn't,
        # since there's no prior context to concede to), the loop should still
        # run at least one interpretive round before checking convergence.
        # In practice round 0 has no STANCE field → defaults to HOLD — test that.
        llm = ScriptedLLM(
            {
                "LOCUTIONARY": [_round_zero(), _hold()],
                "ILLOCUTIONARY": [_round_zero(), _hold()],
                "PERLOCUTIONARY": [_round_zero(), _hold()],
            }
        )
        result = run_dialogue("q", _sat_agents(llm), num_rounds=2)
        for resp in result.rounds[0]:
            # Round 0 → default HOLD, never CONCEDE.
            self.assertEqual(resp.stance.value, "hold")


if __name__ == "__main__":
    unittest.main()
