"""Smoke tests: prompts are short enough to not contaminate outputs, and
the pipeline defaults to the SAT variant.
"""

import inspect
import unittest

from d2c import pipeline, prompts


# Hard ceiling on agent system prompts. Large prompts with concrete examples
# were shown to cause small models to copy-paste examples verbatim into
# their outputs. Keep system prompts terse.
_MAX_AGENT_SYSTEM_PROMPT_CHARS = 1000


class TestPromptsAreLean(unittest.TestCase):
    _AGENT_PROMPTS = [
        "LITERALIST_SYSTEM",
        "INTENT_SEEKER_SYSTEM",
        "SCOPE_EXPANDER_SYSTEM",
        "LOCUTIONARY_SYSTEM",
        "ILLOCUTIONARY_SYSTEM",
        "PERLOCUTIONARY_SYSTEM",
    ]

    def test_agent_prompts_under_ceiling(self):
        for name in self._AGENT_PROMPTS:
            with self.subTest(prompt=name):
                p = getattr(prompts, name)
                self.assertLess(
                    len(p),
                    _MAX_AGENT_SYSTEM_PROMPT_CHARS,
                    f"{name} is {len(p)} chars — too long, risks prompt "
                    "contamination on small models.",
                )


class TestPipelineDefault(unittest.TestCase):
    def test_run_d2c_defaults_to_speech_act_variant(self):
        sig = inspect.signature(pipeline.run_d2c)
        self.assertEqual(sig.parameters["variant"].default, "speech_act")

    def test_run_d2c_batch_defaults_to_speech_act_variant(self):
        sig = inspect.signature(pipeline.run_d2c_batch)
        self.assertEqual(sig.parameters["variant"].default, "speech_act")


if __name__ == "__main__":
    unittest.main()
