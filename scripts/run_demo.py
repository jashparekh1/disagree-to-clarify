"""Quick demo: single query -> clarifying question.

Usage:
    python -m scripts.run_demo "What is the best way to deal with a Python crash?"
    python -m scripts.run_demo --model qwen3:8b "How do I handle a merge conflict?" --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

from d2c.agents import _ROLE_DISPLAY
from d2c.pipeline import run_d2c


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D2C on a single query")
    parser.add_argument("query", help="The ambiguous query to disambiguate")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--rounds", type=int, default=3, help="Number of dialogue rounds")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per LLM call")
    parser.add_argument(
        "--variant",
        default="speech_act",
        choices=["original", "speech_act"],
        help="Agent variant. 'speech_act' (default) uses the SAT-grounded "
        "Locutionary/Illocutionary/Perlocutionary trio; 'original' is the "
        "pre-theory Literalist/Intent-Seeker/Scope-Expander trio, kept for "
        "ablation.",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Qwen3 thinking mode (pass think=false to Ollama). "
        "Recommended for small models (e.g., qwen3:0.6b) where thinking "
        "tokens eat the budget without producing useful reasoning.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full dialogue transcript")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                        help="LLM backend: ollama (default) or openai (vLLM / any OpenAI-compatible server)")
    parser.add_argument("--base-url", default=None,
                        help="Override LLM server URL (default: localhost:11434 for ollama, localhost:8000 for openai)")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\nQuery: {args.query}")
    print(f"Model: {args.model} | Rounds: {args.rounds} | Max Tokens: {args.max_tokens}")
    print("\nRunning D2C pipeline... (this might take a few seconds)")

    result = run_d2c(
        args.query,
        model=args.model,
        num_rounds=args.rounds,
        max_tokens=args.max_tokens,
        variant=args.variant,
        think=False if args.no_think else None,
        base_url=args.base_url,
        backend=args.backend,
    )

    # Verbose: print full round-by-round dialogue
    if args.verbose:
        for round_idx, rnd in enumerate(result.dialogue.rounds):
            print(f"{'='*60}")
            print(f"  Round {round_idx}")
            print(f"{'='*60}")
            for resp in rnd:
                stance = f" [{resp.stance.value}]" if round_idx > 0 else ""
                print(f"\n  [{_ROLE_DISPLAY[resp.role]}]{stance}")
                print(f"  INTERPRETATION: {resp.interpretation}")
                if round_idx > 0 and resp.stance_reason:
                    print(f"  STANCE_REASON: {resp.stance_reason}")
            print()

    # Always print final summary
    print("-" * 60)
    print("FINAL INTERPRETATIONS:")
    print("-" * 60)
    final_round = result.dialogue.rounds[-1]
    for resp in final_round:
        print(f"  [{_ROLE_DISPLAY[resp.role]}] {resp.interpretation}")
    print()
    print(f"KEY DISAGREEMENT: {result.synthesizer_result.key_disagreement}")
    print(f"CLARIFYING QUESTION: {result.synthesizer_result.clarifying_question}")


if __name__ == "__main__":
    main()
