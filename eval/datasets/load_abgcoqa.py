"""Abg-CoQA dataset loader.

Abg-CoQA (Clarifying Ambiguity in CoQA) extends CoQA with ambiguity labels 
and clarification questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.datasets.base import AmbiguousQuery


def load_abgcoqa(data_dir: str = "data/abgcoqa") -> list[AmbiguousQuery]:
    """Load Abg-CoQA benchmark. Returns list of AmbiguousQuery."""
    data_path = Path(data_dir) / "abg-coqa" / "coqa_abg_val.json"
    
    if not data_path.exists():
        data_path = Path(data_dir) / "abg-coqa" / "coqa_abg_test.json"
    
    if not data_path.exists():
        print(f"Abg-CoQA data not found at {data_path}. Returning empty list.")
        return []

    items: list[AmbiguousQuery] = []
    
    with open(data_path) as f:
        full_json = json.load(f)
        raw_data = full_json.get("data", [])

    for i, entry in enumerate(raw_data):
        story = entry.get("story", "")
        target_turn = entry.get("target_turn", {})
        query = target_turn.get("question", "")
        
        # In Abg-CoQA val/test, "ambiguity" field is "ambiguous" or "non_ambiguous"
        ambiguity_label = entry.get("ambiguity", "")
        is_ambiguous = ambiguity_label == "ambiguous"
        
        clarification_turn = entry.get("clarification_turn", {})
        gold_cq = clarification_turn.get("question", "")
        
        items.append(AmbiguousQuery(
            query=query,
            gold_clarifying_question=gold_cq if is_ambiguous else "",
            is_ambiguous=is_ambiguous,
            ambiguity_type=ambiguity_label,
            dataset="abgcoqa",
            example_id=entry.get("id", f"abgcoqa_{i}"),
            context=story,
            raw=entry,
        ))

    total = len(items)
    amb = sum(1 for it in items if it.is_ambiguous)
    print(f"Abg-CoQA: {total} total, {amb} ambiguous, {total - amb} unambiguous")

    return items
