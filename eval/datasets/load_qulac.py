"""Qulac dataset loader.

Source: https://github.com/aliannejadi/qulac
File: data/qulac/qulac.json

Raw format: list of objects, each is one topic-facet-question combination:
{
  "topic_id": 21,
  "facet_id": 1,
  "topic": "dinosaur",
  "sc": "Find information about dinosaur exhibits at museums",
  "question": "Are you interested in visiting a dinosaur exhibit at a museum?",
  "answer": "Yes, I want to find museums with dinosaur exhibits near me.",
  "topic_facet_id": "21-1",
  "topic_facet_question_id": "21-1-5",
  "facet_type": "inf",
  "topic_desc": "I want to learn about dinosaurs..."
}

Multiple rows share the same topic but have different facets and questions.
Each row becomes a separate AmbiguousQuery. At eval time, score against ALL
gold questions for the same topic and take the MAX (max-over-facets).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.datasets.base import AmbiguousQuery


def load_qulac(data_dir: str = "data/qulac") -> list[AmbiguousQuery]:
    """Load Qulac dataset. Returns list of AmbiguousQuery.

    Note: multiple entries per topic (one per facet). Group by topic_id for eval.
    """
    data_path = Path(data_dir) / "data" / "qulac" / "qulac.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Qulac data not found at {data_path}. "
            "Run `python -m eval.datasets.download` first."
        )

    with open(data_path) as f:
        raw_data = json.load(f)

    # Current Qulac ships a column-oriented JSON: {field: {row_id: value, ...}, ...}
    # (the pandas .to_dict() default). Convert to row-oriented list of dicts.
    if isinstance(raw_data, dict):
        fields = list(raw_data.keys())
        row_ids = list(raw_data[fields[0]].keys())
        rows = [{f: raw_data[f].get(rid) for f in fields} for rid in row_ids]
    else:
        rows = raw_data  # Older dumps are already list-of-dicts.

    items: list[AmbiguousQuery] = []
    for obj in rows:
        topic_facet_question_id = str(obj.get("topic_facet_question_id", ""))
        topic_id = str(obj.get("topic_id", ""))

        items.append(AmbiguousQuery(
            query=obj.get("topic", ""),
            gold_clarifying_question=obj.get("question", ""),
            is_ambiguous=True,  # Qulac only contains ambiguous queries
            ambiguity_type=None,
            dataset="qulac",
            example_id=topic_facet_question_id or f"qulac_{len(items)}",
            facet=obj.get("facet_desc", obj.get("sc", "")),
            topic_id=topic_id,
            raw=obj,
        ))

    # Print stats
    topics = Counter(it.topic_id for it in items)
    n_topics = len(topics)
    avg_facets = len(items) / n_topics if n_topics else 0
    print(f"Qulac: {len(items)} entries, {n_topics} unique topics, {avg_facets:.1f} avg entries/topic")

    return items
