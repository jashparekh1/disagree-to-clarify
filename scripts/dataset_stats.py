"""Print summary statistics for all evaluation datasets.

Usage: python -m scripts.dataset_stats
"""

from __future__ import annotations

from collections import Counter

from eval.datasets import load_dataset


def _avg_tokens(texts: list[str]) -> float:
    """Average whitespace-split token count."""
    if not texts:
        return 0.0
    return sum(len(t.split()) for t in texts) / len(texts)


def stats_clamber() -> None:
    print("\n" + "=" * 50)
    print("CLAMBER")
    print("=" * 50)
    try:
        items = load_dataset("clamber")
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    total = len(items)
    amb = sum(1 for it in items if it.is_ambiguous)
    unamb = total - amb
    print(f"  Total examples: {total}")
    print(f"  Ambiguous: {amb} ({amb/total*100:.1f}%)")
    print(f"  Unambiguous: {unamb} ({unamb/total*100:.1f}%)")

    type_counts = Counter(it.ambiguity_type for it in items if it.is_ambiguous)
    print(f"  Ambiguity types:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    queries = [it.query for it in items]
    cqs = [it.gold_clarifying_question for it in items if it.gold_clarifying_question]
    print(f"  Avg query length: {_avg_tokens(queries):.1f} tokens")
    print(f"  Avg clarifying question length: {_avg_tokens(cqs):.1f} tokens")


def stats_qulac() -> None:
    print("\n" + "=" * 50)
    print("Qulac")
    print("=" * 50)
    try:
        items = load_dataset("qulac")
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    topics = Counter(it.topic_id for it in items)
    n_topics = len(topics)
    avg_per_topic = len(items) / n_topics if n_topics else 0

    print(f"  Total entries: {len(items)}")
    print(f"  Unique topics: {n_topics}")
    print(f"  Avg entries per topic: {avg_per_topic:.1f}")

    # Avg clarifying questions per topic
    cqs_per_topic = Counter(it.topic_id for it in items if it.gold_clarifying_question)
    avg_cqs = sum(cqs_per_topic.values()) / len(cqs_per_topic) if cqs_per_topic else 0
    print(f"  Avg clarifying questions per topic: {avg_cqs:.1f}")


def stats_clariq() -> None:
    print("\n" + "=" * 50)
    print("ClariQ")
    print("=" * 50)
    try:
        items = load_dataset("clariq")
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    topics = Counter(it.topic_id for it in items)
    n_topics = len(topics)
    amb = sum(1 for it in items if it.is_ambiguous)

    print(f"  Total entries: {len(items)}")
    print(f"  Unique topics: {n_topics}")
    print(f"  Ambiguous (need>=2): {amb}")

    clar_dist = Counter(it.raw.get("clarification_need") for it in items)
    print(f"  Clarification need distribution:")
    for k in sorted(clar_dist):
        print(f"    {k}: {clar_dist[k]}")

    split_counts = Counter(it.raw.get("split") for it in items)
    print(f"  Train/Dev split: {split_counts.get('train', 0)}/{split_counts.get('dev', 0)}")


def main() -> None:
    print("D2C Evaluation — Dataset Statistics")
    stats_clamber()
    stats_qulac()
    stats_clariq()
    print()


if __name__ == "__main__":
    main()
