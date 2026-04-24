"""Smoke tests: hardened SAT prompts embed their theoretical scaffolding
and the pipeline defaults to the SAT variant.

These tests are deliberately shallow — we are not judging prompt quality
here, only asserting that the theoretical markers we rely on in the class
writeup are actually present in the prompt strings the agents will send.
"""

import inspect
import unittest

from d2c import pipeline, prompts


class TestSATPromptContent(unittest.TestCase):
    def test_locutionary_embeds_austin_decomposition(self):
        p = prompts.LOCUTIONARY_SYSTEM
        # Theoretical backing: Austin's tripartite locutionary decomposition.
        self.assertIn("Austin", p)
        self.assertIn("phatic", p)
        self.assertIn("rhetic", p)
        # Grounding-gap framing.
        self.assertIn("REFERENTIAL", p)

    def test_illocutionary_embeds_searle_five_classes(self):
        p = prompts.ILLOCUTIONARY_SYSTEM
        self.assertIn("Searle", p)
        # All five Searle-1976 categories must be named, or the agent can't
        # be asked to classify into them.
        for force in (
            "Assertive",
            "Directive",
            "Commissive",
            "Expressive",
            "Declaration",
        ):
            with self.subTest(force=force):
                self.assertIn(force, p)
        # Indirect speech acts (Searle 1975) must be mentioned so the agent
        # can flag polite-directive-as-question cases.
        self.assertIn("Indirect", p)
        self.assertIn("INTENT", p)

    def test_perlocutionary_distinguishes_effect_from_felicity(self):
        p = prompts.PERLOCUTIONARY_SYSTEM
        self.assertIn("Austin", p)
        self.assertIn("perlocutionary", p.lower())
        # The distinction from felicity conditions is the substantive point
        # that separates this agent from the Illocutionary agent.
        self.assertIn("felicity", p.lower())
        self.assertIn("PRAGMATIC", p)


class TestSynthesizerGroundingFraming(unittest.TestCase):
    def test_synthesizer_cites_clark_and_maps_divergence_types(self):
        p = prompts.SYNTHESIZER_SYSTEM
        self.assertIn("Clark", p)
        # The synthesizer must know the three divergence → gap mappings.
        self.assertIn("REFERENTIAL", p)
        self.assertIn("INTENT", p)
        self.assertIn("PRAGMATIC", p)


class TestPipelineDefault(unittest.TestCase):
    def test_run_d2c_defaults_to_speech_act_variant(self):
        sig = inspect.signature(pipeline.run_d2c)
        self.assertEqual(sig.parameters["variant"].default, "speech_act")

    def test_run_d2c_batch_defaults_to_speech_act_variant(self):
        sig = inspect.signature(pipeline.run_d2c_batch)
        self.assertEqual(sig.parameters["variant"].default, "speech_act")


if __name__ == "__main__":
    unittest.main()
