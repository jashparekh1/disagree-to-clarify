"""Build fixed, reproducible test sets from ClariQ / Qulac / CLAMBER / Abg-CoQA.

Now includes BOTH clear and ambiguous queries for detection metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from eval.datasets import load_dataset

OUT_DIR = Path("test_sets")


def build_topic_grouped(dataset_name: str, n: int | None, seed: int) -> list[dict]:
    """For datasets where multiple entries share a topic (ClariQ, Qulac)."""
    items = load_dataset(dataset_name)
    by_topic = defaultdict(list)
    for it in items:
        by_topic[it.topic_id].append(it)

    topic_ids = sorted(by_topic.keys())
    random.Random(seed).shuffle(topic_ids)
    if n is not None:
        topic_ids = topic_ids[:n]

    records = []
    for tid in topic_ids:
        entries = by_topic[tid]
        rep = entries[0]
        golds = sorted(
            {e.gold_clarifying_question for e in entries if e.gold_clarifying_question}
        )
        records.append(
            {
                "dataset": dataset_name,
                "topic_id": tid,
                "example_id": rep.example_id,
                "query": rep.query,
                "context": rep.context,
                "is_ambiguous": any(e.is_ambiguous for e in entries),
                "gold_clarifying_questions": golds,
                "ambiguity_type": rep.ambiguity_type,
                "n_golds": len(golds),
            }
        )
    return records


def build_flat(dataset_name: str, n: int | None, seed: int) -> list[dict]:
    """For datasets without topic grouping (CLAMBER, Abg-CoQA)."""
    items = load_dataset(dataset_name)
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    return [
        {
            "dataset": dataset_name,
            "topic_id": None,
            "example_id": it.example_id,
            "query": it.query,
            "context": it.context,
            "is_ambiguous": it.is_ambiguous,
            "gold_clarifying_questions": [it.gold_clarifying_question]
            if it.gold_clarifying_question
            else [],
            "ambiguity_type": it.ambiguity_type,
            "n_golds": 1 if it.gold_clarifying_question else 0,
        }
        for it in items
    ]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clariq-n", type=int, default=None)
    parser.add_argument("--qulac-n", type=int, default=None)
    parser.add_argument("--clamber-n", type=int, default=500)
    parser.add_argument("--abgcoqa-n", type=int, default=500)
    args = parser.parse_args()

    print(f"Building test sets (Mixed Ambiguity) → {OUT_DIR}/ (seed={args.seed})")

    jobs = [
        ("clariq", build_topic_grouped, args.clariq_n),
        ("qulac", build_topic_grouped, args.qulac_n),
        ("clamber", build_flat, args.clamber_n),
        ("abgcoqa", build_flat, args.abgcoqa_n),
    ]

    for name, builder, n in jobs:
        recs = builder(name, n, args.seed)
        out = OUT_DIR / f"{name}_test.jsonl"
        write_jsonl(recs, out)
        amb = sum(1 for r in recs if r["is_ambiguous"])
        print(f"  [{name}] {len(recs)} total, {amb} ambiguous → {out}")


if __name__ == "__main__":
    main()
