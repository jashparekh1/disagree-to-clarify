"""Parser and stance-handling tests for d2c.agents."""

import unittest

from d2c.agents import AgentRole, Stance, _parse_response, _parse_stance


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


class TestResponseParsing(unittest.TestCase):
    def test_full_response_with_stance(self):
        raw = (
            "INTERPRETATION: The user wants to know X.\n"
            "ASSUMPTIONS: Context Y.\n"
            "ANSWER_TYPE: Factual.\n"
            "DISAGREEMENTS: Literalist missed Z.\n"
            "STANCE: HOLD\n"
            "STANCE_REASON: My reading catches pragmatic force theirs don't.\n"
        )
        r = _parse_response(raw, AgentRole.ILLOCUTIONARY, round_num=1)
        self.assertTrue(r.interpretation.startswith("The user wants"))
        self.assertTrue(r.assumptions.startswith("Context Y"))
        self.assertTrue(r.answer_type.startswith("Factual"))
        self.assertTrue(r.disagreements.startswith("Literalist"))
        self.assertEqual(r.stance, Stance.HOLD)
        self.assertIn("pragmatic force", r.stance_reason)

    def test_round_zero_response_has_default_stance(self):
        raw = (
            "INTERPRETATION: Surface reading.\n"
            "ASSUMPTIONS: None.\n"
            "ANSWER_TYPE: List.\n"
            "DISAGREEMENTS: \n"
        )
        r = _parse_response(raw, AgentRole.LOCUTIONARY, round_num=0)
        # Round 0 responses don't emit STANCE — default must be HOLD.
        self.assertEqual(r.stance, Stance.HOLD)
        self.assertEqual(r.stance_reason, "")

    def test_concede_parsed_case_insensitive(self):
        raw = (
            "INTERPRETATION: x\n"
            "ASSUMPTIONS: y\n"
            "ANSWER_TYPE: z\n"
            "DISAGREEMENTS: w\n"
            "STANCE: concede\n"
            "STANCE_REASON: Illocutionary subsumes my reading.\n"
        )
        r = _parse_response(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertEqual(r.stance, Stance.CONCEDE)
        self.assertIn("Illocutionary", r.stance_reason)

    def test_malformed_stance_defaults_to_hold(self):
        raw = (
            "INTERPRETATION: x\n"
            "ASSUMPTIONS: y\n"
            "ANSWER_TYPE: z\n"
            "DISAGREEMENTS: w\n"
            "STANCE: idk\n"
            "STANCE_REASON: unclear.\n"
        )
        r = _parse_response(raw, AgentRole.LOCUTIONARY, round_num=1)
        self.assertEqual(r.stance, Stance.HOLD)

    def test_format_for_others_omits_stance_round_zero(self):
        raw = "INTERPRETATION: a\nASSUMPTIONS: b\nANSWER_TYPE: c\nDISAGREEMENTS: d\n"
        r = _parse_response(raw, AgentRole.LITERALIST, round_num=0)
        formatted = r.format_for_others()
        self.assertNotIn("STANCE:", formatted)

    def test_format_for_others_includes_stance_round_one(self):
        raw = (
            "INTERPRETATION: a\n"
            "ASSUMPTIONS: b\n"
            "ANSWER_TYPE: c\n"
            "DISAGREEMENTS: d\n"
            "STANCE: CONCEDE\n"
            "STANCE_REASON: yielded.\n"
        )
        r = _parse_response(raw, AgentRole.LITERALIST, round_num=1)
        formatted = r.format_for_others()
        self.assertIn("STANCE: CONCEDE", formatted)
        self.assertIn("yielded", formatted)


if __name__ == "__main__":
    unittest.main()
