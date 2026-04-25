"""Build fixed, reproducible test sets from ClariQ / Qulac / CLAMBER.

Each test set is a JSONL file, one record per unique query, with all gold
clarifying questions attached (max-over-facets scoring depends on this).

Outputs to ./test_sets/{dataset}_test.jsonl.

Usage:
    python -m scripts.build_test_sets
    python -m scripts.build_test_sets --clamber-n 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from eval.datasets import load_dataset
from eval.datasets.base import AmbiguousQuery

OUT_DIR = Path("test_sets")


def build_topic_grouped(dataset_name: str, n: int | None, seed: int) -> list[dict]:
    """For datasets with topic-level queries (ClariQ, Qulac), return one
    record per unique topic with all gold questions merged into a list.
    """
    items = [q for q in load_dataset(dataset_name) if q.is_ambiguous]
    by_topic: dict[str, list[AmbiguousQuery]] = defaultdict(list)
    for q in items:
        by_topic[q.topic_id or q.query].append(q)

    topic_ids = list(by_topic.keys())
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
                "gold_clarifying_questions": golds,
                "ambiguity_type": rep.ambiguity_type,
                "n_golds": len(golds),
            }
        )
    return records


def build_flat(dataset_name: str, n: int | None, seed: int) -> list[dict]:
    """For datasets without topic grouping (CLAMBER), sample N ambiguous
    items, one gold per record.
    """
    items = [q for q in load_dataset(dataset_name) if q.is_ambiguous]
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    return [
        {
            "dataset": dataset_name,
            "topic_id": q.topic_id,
            "example_id": q.example_id,
            "query": q.query,
            "gold_clarifying_questions": [q.gold_clarifying_question]
            if q.gold_clarifying_question
            else [],
            "ambiguity_type": q.ambiguity_type,
            "n_golds": 1 if q.gold_clarifying_question else 0,
        }
        for q in items
    ]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    # Defaults: ClariQ and Qulac capped at all-unique-topics; CLAMBER at 500.
    parser.add_argument(
        "--clariq-n",
        type=int,
        default=None,
        help="Cap ClariQ unique-topic count (default: all topics).",
    )
    parser.add_argument(
        "--qulac-n",
        type=int,
        default=None,
        help="Cap Qulac unique-topic count (default: all topics).",
    )
    parser.add_argument(
        "--clamber-n",
        type=int,
        default=500,
        help="Cap CLAMBER sampled-row count (default: 500).",
    )
    args = parser.parse_args()

    print(f"Building test sets → {OUT_DIR}/ (seed={args.seed})")
    print()

    jobs = [
        ("clariq", build_topic_grouped, args.clariq_n),
        ("qulac", build_topic_grouped, args.qulac_n),
        ("clamber", build_flat, args.clamber_n),
    ]

    summary = {}
    for name, builder, n in jobs:
        recs = builder(name, n, args.seed)
        out = OUT_DIR / f"{name}_test.jsonl"
        write_jsonl(recs, out)
        avg_golds = (
            sum(r["n_golds"] for r in recs) / len(recs) if recs else 0.0
        )
        summary[name] = {"n": len(recs), "avg_golds": avg_golds, "path": str(out)}
        print(
            f"  [{name}] {len(recs)} records  "
            f"avg_golds={avg_golds:.1f}  → {out}"
        )

    print()
    print("SUMMARY")
    total = sum(s["n"] for s in summary.values())
    print(f"  total queries across datasets: {total}")
    for name, s in summary.items():
        print(f"  {name}: {s['n']} queries, avg {s['avg_golds']:.1f} gold(s) each")


if __name__ == "__main__":
    main()
