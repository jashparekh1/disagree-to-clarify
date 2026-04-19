"""ClariQ dataset loader.

Source: https://github.com/aliannejadi/ClariQ
Files: data/train.tsv, data/dev.tsv

TSV columns:
  topic_id  initial_request  topic_desc  clarification_need  facet_id  facet_desc  question_id  question  answer

Multiple rows per topic (one per facet), same as Qulac.
At eval time, score against ALL gold questions for the same topic and take MAX.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from eval.datasets.base import AmbiguousQuery

# Queries with clarification_need >= this threshold are considered ambiguous
AMBIGUITY_THRESHOLD = 2


def _load_tsv(path: Path, split: str) -> list[AmbiguousQuery]:
    """Load a single ClariQ TSV file."""
    items: list[AmbiguousQuery] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clar_need = int(row.get("clarification_need", 0))
            topic_id = str(row.get("topic_id", ""))
            question_id = str(row.get("question_id", ""))

            items.append(AmbiguousQuery(
                query=row.get("initial_request", ""),
                gold_clarifying_question=row.get("question", ""),
                is_ambiguous=clar_need >= AMBIGUITY_THRESHOLD,
                ambiguity_type=None,
                dataset="clariq",
                example_id=question_id or f"clariq_{split}_{len(items)}",
                facet=row.get("facet_desc", ""),
                topic_id=topic_id,
                raw={**row, "split": split, "clarification_need": clar_need},
            ))
    return items


def load_clariq(data_dir: str = "data/clariq") -> list[AmbiguousQuery]:
    """Load ClariQ dataset (train + dev splits). Returns list of AmbiguousQuery."""
    base = Path(data_dir) / "data"

    all_items: list[AmbiguousQuery] = []
    for split in ("train", "dev"):
        tsv_path = base / f"{split}.tsv"
        if not tsv_path.exists():
            print(f"  Warning: {tsv_path} not found, skipping.")
            continue
        items = _load_tsv(tsv_path, split)
        all_items.extend(items)
        print(f"  ClariQ {split}: {len(items)} entries")

    if not all_items:
        raise FileNotFoundError(
            f"ClariQ data not found in {base}. "
            "Run `python -m eval.datasets.download` first."
        )

    # Print stats
    topics = Counter(it.topic_id for it in all_items)
    n_topics = len(topics)
    amb = sum(1 for it in all_items if it.is_ambiguous)
    clar_dist = Counter(it.raw.get("clarification_need") for it in all_items)
    split_counts = Counter(it.raw.get("split") for it in all_items)

    print(f"ClariQ: {len(all_items)} total, {n_topics} unique topics, {amb} ambiguous (need>={AMBIGUITY_THRESHOLD})")
    print(f"  Clarification need distribution: {dict(sorted(clar_dist.items()))}")
    print(f"  Splits: {dict(split_counts)}")

    return all_items
