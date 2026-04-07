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
    parser.add_argument("--verbose", action="store_true", help="Print full dialogue transcript")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\nQuery: {args.query}")
    print(f"Model: {args.model} | Rounds: {args.rounds}\n")

    result = run_d2c(args.query, model=args.model, num_rounds=args.rounds)

    # Verbose: print full round-by-round dialogue
    if args.verbose:
        for round_idx, rnd in enumerate(result.dialogue.rounds):
            print(f"{'='*60}")
            print(f"  Round {round_idx}")
            print(f"{'='*60}")
            for resp in rnd:
                print(f"\n  [{_ROLE_DISPLAY[resp.role]}]")
                print(f"  INTERPRETATION: {resp.interpretation}")
                print(f"  ASSUMPTIONS: {resp.assumptions}")
                print(f"  ANSWER_TYPE: {resp.answer_type}")
                print(f"  DISAGREEMENTS: {resp.disagreements}")
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
