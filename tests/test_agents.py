"""Parser + stance tests for d2c.agents.

With structured outputs the LLM response is guaranteed-valid JSON per
schema; these tests cover the parser's happy path and defensive fallback.
"""

import json
import unittest

from d2c.agents import (
    Agent,
    AgentRole,
    ROUND_N_SCHEMA,
    ROUND_ZERO_SCHEMA,
    Stance,
    _parse_agent_json,
    _parse_stance,
)

from tests._fake_llm import ScriptedLLM


class TestStanceParsing(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(_parse_stance("HOLD"), Stance.HOLD)
        self.assertEqual(_parse_stance("hold"), Stance.HOLD)
        self.assertEqual(_parse_stance("CONCEDE"), Stance.CONCEDE)
        self.assertEqual(_parse_stance("concede"), Stance.CONCEDE)

    def test_unknown_defaults_to_hold(self):
        # Safety: spurious stance must not trigger false early-stop.
        self.assertEqual(_parse_stance(""), Stance.HOLD)
        self.assertEqual(_parse_stance("maybe"), Stance.HOLD)


class TestParseAgentJson(unittest.TestCase):
    def test_round_zero_response(self):
        raw = json.dumps({"interpretation": "surface reading"})
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=0)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.interpretation, "surface reading")
        # Round 0 has no stance from the model → defaults.
        self.assertEqual(r.stance, Stance.HOLD)
        self.assertEqual(r.stance_reason, "")

    def test_round_n_response(self):
        raw = json.dumps(
            {
                "interpretation": "refined reading",
                "stance": "CONCEDE",
                "stance_reason": "Illocutionary subsumes my reading.",
            }
        )
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.stance, Stance.CONCEDE)
        self.assertIn("Illocutionary", r.stance_reason)

    def test_invalid_json_flags_format_failed(self):
        raw = "not JSON at all"
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertTrue(r.format_failed)
        self.assertEqual(r.interpretation, "not JSON at all")

    def test_non_dict_json_flags_format_failed(self):
        raw = '["list", "not", "dict"]'
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertTrue(r.format_failed)


class TestSchemas(unittest.TestCase):
    """Schemas shape must match what the parser expects."""

    def test_round_zero_schema_requires_interpretation(self):
        self.assertIn("interpretation", ROUND_ZERO_SCHEMA["required"])
        self.assertEqual(ROUND_ZERO_SCHEMA["required"], ["interpretation"])

    def test_round_n_schema_requires_stance_and_reason(self):
        self.assertEqual(
            set(ROUND_N_SCHEMA["required"]),
            {"interpretation", "stance", "stance_reason"},
        )

    def test_stance_enum_constrained(self):
        self.assertEqual(
            ROUND_N_SCHEMA["properties"]["stance"]["enum"],
            ["HOLD", "CONCEDE"],
        )


class TestAgentCallsSchemaConstrainedLLM(unittest.TestCase):
    """Agent.respond_* must pass the appropriate schema through to the LLM."""

    def test_respond_initial_uses_round_zero_schema(self):
        llm = ScriptedLLM({"LOCUTIONARY": [json.dumps({"interpretation": "x"})]})
        agent = Agent(AgentRole.LOCUTIONARY, llm)
        agent.respond_initial("q")
        # ScriptedLLM records kwargs; verify the schema argument was passed.
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0].format_schema, ROUND_ZERO_SCHEMA)

    def test_respond_dialogue_uses_round_n_schema(self):
        payload_r0 = json.dumps({"interpretation": "x"})
        payload_r1 = json.dumps(
            {"interpretation": "y", "stance": "HOLD", "stance_reason": "r"}
        )
        llm = ScriptedLLM(
            {"LOCUTIONARY": [payload_r0, payload_r1]}
        )
        agent = Agent(AgentRole.LOCUTIONARY, llm)
        r0 = agent.respond_initial("q")
        agent.respond_dialogue("q", other_responses=[r0], round_num=1)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0].format_schema, ROUND_ZERO_SCHEMA)
        self.assertEqual(llm.calls[1].format_schema, ROUND_N_SCHEMA)


class TestFormatForOthers(unittest.TestCase):
    def test_round_zero_omits_stance(self):
        raw = json.dumps({"interpretation": "surface reading"})
        r = _parse_agent_json(raw, AgentRole.LITERALIST, round_num=0)
        formatted = r.format_for_others()
        self.assertIn("surface reading", formatted)
        self.assertNotIn("stance:", formatted)

    def test_round_one_includes_stance(self):
        raw = json.dumps(
            {"interpretation": "x", "stance": "CONCEDE", "stance_reason": "yielded"}
        )
        r = _parse_agent_json(raw, AgentRole.LITERALIST, round_num=1)
        formatted = r.format_for_others()
        self.assertIn("CONCEDE", formatted)
        self.assertIn("yielded", formatted)


if __name__ == "__main__":
    unittest.main()
