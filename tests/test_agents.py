"""Parser, stance, and JSON-retry tests for d2c.agents."""

import unittest

from d2c.agents import (
    Agent,
    AgentRole,
    Stance,
    _extract_json_blob,
    _parse_agent_json,
    _parse_stance,
)

from tests._fake_llm import ScriptedLLM


class TestStanceParsing(unittest.TestCase):
    def test_hold(self):
        self.assertEqual(_parse_stance("HOLD"), Stance.HOLD)
        self.assertEqual(_parse_stance("hold"), Stance.HOLD)
        self.assertEqual(_parse_stance("  Hold  "), Stance.HOLD)

    def test_concede(self):
        self.assertEqual(_parse_stance("CONCEDE"), Stance.CONCEDE)
        self.assertEqual(_parse_stance("concede"), Stance.CONCEDE)
        self.assertEqual(_parse_stance("conceded"), Stance.CONCEDE)
        self.assertEqual(_parse_stance("Conceding"), Stance.CONCEDE)

    def test_unknown_defaults_to_hold(self):
        # Safety: a format glitch must not accidentally trigger early-stop.
        self.assertEqual(_parse_stance("gibberish"), Stance.HOLD)
        self.assertEqual(_parse_stance(""), Stance.HOLD)
        self.assertEqual(_parse_stance("AGREE"), Stance.HOLD)


class TestExtractJsonBlob(unittest.TestCase):
    def test_clean_object(self):
        self.assertEqual(_extract_json_blob('{"a": 1}'), '{"a": 1}')

    def test_markdown_fence_json(self):
        raw = 'Here is the response:\n```json\n{"a": 1}\n```\n'
        self.assertIn('"a": 1', _extract_json_blob(raw))

    def test_markdown_fence_no_lang(self):
        raw = '```\n{"a": 1}\n```'
        self.assertIn('"a": 1', _extract_json_blob(raw))

    def test_prose_wrapping(self):
        # Some models prepend a thinking-style prefix even with strip_thinking.
        raw = 'Sure! Here you go: {"a": 1} — let me know if you need more.'
        self.assertEqual(_extract_json_blob(raw), '{"a": 1}')

    def test_no_json_returns_none(self):
        self.assertIsNone(_extract_json_blob("I cannot comply."))
        self.assertIsNone(_extract_json_blob(""))


class TestParseAgentJson(unittest.TestCase):
    def test_full_response_with_stance(self):
        raw = """{
  "interpretation": "The user wants to know X.",
  "assumptions": "Context Y.",
  "answer_type": "Factual.",
  "disagreements": "Literalist missed Z.",
  "stance": "HOLD",
  "stance_reason": "My reading catches pragmatic force theirs don't."
}"""
        r = _parse_agent_json(raw, AgentRole.ILLOCUTIONARY, round_num=1)
        self.assertFalse(r.format_failed)
        self.assertTrue(r.interpretation.startswith("The user wants"))
        self.assertEqual(r.assumptions, "Context Y.")
        self.assertEqual(r.stance, Stance.HOLD)
        self.assertIn("pragmatic force", r.stance_reason)

    def test_markdown_fence_wrapping_still_parses(self):
        raw = """```json
{
  "interpretation": "surface reading",
  "assumptions": "none",
  "answer_type": "list",
  "disagreements": ""
}
```"""
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=0)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.interpretation, "surface reading")

    def test_round_zero_omitting_stance_defaults_to_hold(self):
        raw = """{
  "interpretation": "Surface reading.",
  "assumptions": "None.",
  "answer_type": "List.",
  "disagreements": ""
}"""
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=0)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.stance, Stance.HOLD)
        self.assertEqual(r.stance_reason, "")

    def test_concede_case_insensitive(self):
        raw = """{"interpretation": "x", "assumptions": "y", "answer_type": "z",
                   "disagreements": "w", "stance": "concede",
                   "stance_reason": "Illocutionary subsumes my reading."}"""
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.stance, Stance.CONCEDE)

    def test_unknown_stance_defaults_to_hold_no_failure(self):
        # A malformed stance value (valid JSON, unrecognized value) must NOT
        # raise format_failed — the JSON parsed fine, stance just falls back.
        raw = """{"interpretation": "x", "assumptions": "y", "answer_type": "z",
                   "disagreements": "w", "stance": "idk",
                   "stance_reason": "unclear"}"""
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertFalse(r.format_failed)
        self.assertEqual(r.stance, Stance.HOLD)

    def test_invalid_json_flags_format_failed(self):
        raw = "INTERPRETATION: x\nSTANCE: HOLD"  # old text-marker format
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertTrue(r.format_failed)
        self.assertIn("INTERPRETATION", r.interpretation)  # raw stashed

    def test_non_dict_json_flags_format_failed(self):
        raw = '["interpretation", "assumptions"]'
        r = _parse_agent_json(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertTrue(r.format_failed)


def _sat_locutionary_agent(llm):
    return Agent(AgentRole.LOCUTIONARY, llm)


class TestAgentRetry(unittest.TestCase):
    """Behavioral tests for the one-shot JSON retry wrapper."""

    def _valid(self, stance: str | None = None):
        body = {
            "interpretation": "ok",
            "assumptions": "ok",
            "answer_type": "ok",
            "disagreements": "",
        }
        if stance is not None:
            body["stance"] = stance
            body["stance_reason"] = "because"
        # Serialise deterministically.
        import json as _json

        return _json.dumps(body)

    def test_first_attempt_ok_no_retry(self):
        llm = ScriptedLLM({"LOCUTIONARY": [self._valid()]})
        agent = _sat_locutionary_agent(llm)
        resp = agent.respond_initial("any query")
        self.assertFalse(resp.format_failed)
        self.assertEqual(len(llm.calls), 1)

    def test_first_attempt_fails_retry_succeeds(self):
        llm = ScriptedLLM(
            {"LOCUTIONARY": ["not valid json at all", self._valid()]}
        )
        agent = _sat_locutionary_agent(llm)
        resp = agent.respond_initial("any query")
        self.assertFalse(resp.format_failed)
        self.assertEqual(len(llm.calls), 2)  # initial + one retry

    def test_both_attempts_fail_returns_fallback(self):
        llm = ScriptedLLM(
            {"LOCUTIONARY": ["garbage one", "garbage two"]}
        )
        agent = _sat_locutionary_agent(llm)
        resp = agent.respond_initial("any query")
        self.assertTrue(resp.format_failed)
        self.assertEqual(len(llm.calls), 2)  # initial + one retry, no third


class TestFormatForOthers(unittest.TestCase):
    def test_round_zero_omits_stance(self):
        raw = """{"interpretation": "a", "assumptions": "b", "answer_type": "c", "disagreements": "d"}"""
        r = _parse_agent_json(raw, AgentRole.LITERALIST, round_num=0)
        formatted = r.format_for_others()
        self.assertNotIn("STANCE:", formatted)

    def test_round_one_includes_stance(self):
        raw = """{"interpretation": "a", "assumptions": "b", "answer_type": "c",
                   "disagreements": "d", "stance": "CONCEDE",
                   "stance_reason": "yielded to Illocutionary"}"""
        r = _parse_agent_json(raw, AgentRole.LITERALIST, round_num=1)
        formatted = r.format_for_others()
        self.assertIn("STANCE: CONCEDE", formatted)
        self.assertIn("yielded", formatted)


if __name__ == "__main__":
    unittest.main()
