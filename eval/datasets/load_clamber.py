"""CLAMBER dataset loader.

Source: https://github.com/zt991211/CLAMBER
File: clamber_benchmark.jsonl

Raw format per line:
{
  "question": "What can you tell me about Mercury?",
  "context": "...",
  "clarifying_question": "Are you referring to the planet Mercury or the element Mercury?",
  "require_clarification": 1,
  "category": "LA",
  "subclass": "whom"
}
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.datasets.base import AmbiguousQuery

CATEGORY_MAP = {
    "LA": "Linguistic Ambiguity",
    "FD": "Epistemic Misalignment",
    "MC": "Aleatoric Output",
}


def load_clamber(data_dir: str = "data/clamber") -> list[AmbiguousQuery]:
    """Load CLAMBER benchmark. Returns list of AmbiguousQuery."""
    data_path = Path(data_dir) / "clamber_benchmark.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"CLAMBER data not found at {data_path}. "
            "Run `python -m eval.datasets.download` first."
        )

    items: list[AmbiguousQuery] = []
    for i, line in enumerate(open(data_path)):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        is_ambiguous = obj.get("require_clarification") == 1
        category_code = obj.get("category", "")
        subclass = obj.get("subclass", "")
        category_full = CATEGORY_MAP.get(category_code, category_code)
        ambiguity_type = f"{category_full}/{subclass}" if subclass else category_full

        items.append(AmbiguousQuery(
            query=obj.get("question", ""),
            gold_clarifying_question=obj.get("clarifying_question", "") if is_ambiguous else "",
            is_ambiguous=is_ambiguous,
            ambiguity_type=ambiguity_type,
            dataset="clamber",
            example_id=f"clamber_{i}",
            context=obj.get("context"),
            raw=obj,
        ))

    # Print stats
    total = len(items)
    amb = sum(1 for it in items if it.is_ambiguous)
    unamb = total - amb
    cat_counts = Counter(it.ambiguity_type for it in items if it.is_ambiguous)
    print(f"CLAMBER: {total} total, {amb} ambiguous, {unamb} unambiguous")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    return items
