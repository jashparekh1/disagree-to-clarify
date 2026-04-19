"""Print random examples from any dataset for inspection.

Usage:
  python -m scripts.inspect_data --dataset clamber --n 5
  python -m scripts.inspect_data --dataset qulac --n 3
  python -m scripts.inspect_data --dataset clariq --n 5
"""

from __future__ import annotations

import argparse
import random

from eval.datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset examples")
    parser.add_argument(
        "--dataset", required=True,
        choices=["clamber", "qulac", "clariq"],
    )
    parser.add_argument("--n", type=int, default=5, help="Number of examples to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    random.seed(args.seed)
    sample = random.sample(items, min(args.n, len(items)))

    print(f"\n--- {args.n} random examples from {args.dataset} ---\n")
    for i, item in enumerate(sample, 1):
        print(f"[{i}] Query: {item.query}")
        print(f"    Gold CQ: {item.gold_clarifying_question}")
        print(f"    Ambiguous: {item.is_ambiguous}")
        if item.ambiguity_type:
            print(f"    Ambiguity type: {item.ambiguity_type}")
        if item.facet:
            print(f"    Facet: {item.facet}")
        if item.topic_id:
            print(f"    Topic ID: {item.topic_id}")
        print()


if __name__ == "__main__":
    main()
