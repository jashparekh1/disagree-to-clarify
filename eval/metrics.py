"""Unified evaluation metrics for D2C.

Three metrics:
1. Clarification Need Prediction (F1)
2. Clarifying Question Quality (LLM-as-Judge)
3. Semantic Similarity to Gold (Embedding Cosine)
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from eval.judge_prompts import JUDGE_SYSTEM, JUDGE_USER

if TYPE_CHECKING:
    from d2c.llm import LLMClient
    from eval.datasets.base import AmbiguousQuery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded sentence transformer model
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embed_model():
    """Load the sentence-transformer model once, cache globally."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


# ---------------------------------------------------------------------------
# Metric 1: Clarification Need Prediction (F1)
# ---------------------------------------------------------------------------

def clarification_need_f1(
    predictions: list[bool],
    gold_labels: list[bool],
) -> dict[str, float]:
    """Compute precision, recall, F1, and accuracy for the 'ambiguous' class.

    Args:
        predictions: True = system decided to ask a clarifying question.
        gold_labels: True = query is actually ambiguous.

    Returns:
        {"precision": ..., "recall": ..., "f1": ..., "accuracy": ...}
    """
    assert len(predictions) == len(gold_labels), "Length mismatch"

    tp = sum(p and g for p, g in zip(predictions, gold_labels))
    fp = sum(p and not g for p, g in zip(predictions, gold_labels))
    fn = sum(not p and g for p, g in zip(predictions, gold_labels))
    correct = sum(p == g for p, g in zip(predictions, gold_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Metric 3: Semantic Similarity (Embedding Cosine)
# ---------------------------------------------------------------------------

def semantic_similarity(generated: str, gold: str) -> float:
    """Cosine similarity between sentence embeddings.

    Uses all-MiniLM-L6-v2 (loaded once, cached).
    Returns float in [0, 1] (clamped — cosine can technically be negative).
    """
    if not generated.strip() or not gold.strip():
        return 0.0

    model = _get_embed_model()
    embeddings = model.encode([generated, gold], normalize_embeddings=True)
    score = float(embeddings[0] @ embeddings[1])
    return max(0.0, score)


def semantic_similarity_multi_ref(
    generated: str,
    gold_questions: list[str],
) -> float:
    """Max cosine similarity across multiple reference questions.

    For Qulac/ClariQ where a topic has multiple valid clarifying questions.
    """
    if not gold_questions:
        return 0.0
    return max(semantic_similarity(generated, gold) for gold in gold_questions)


# ---------------------------------------------------------------------------
# Metric 2: Clarifying Question Quality (LLM-as-Judge)
# ---------------------------------------------------------------------------

def _parse_judge_output(raw: str) -> dict[str, Any]:
    """Parse REASONING and SCORE from judge output."""
    result: dict[str, Any] = {"raw": raw, "score": 0, "reasoning": ""}

    score_idx = raw.find("SCORE:")
    if score_idx != -1:
        rest = raw[score_idx + len("SCORE:"):].strip()
        score_str = rest.split()[0] if rest else "0"
        try:
            result["score"] = max(1, min(5, int(score_str)))
        except ValueError:
            result["score"] = 0

    reasoning_idx = raw.find("REASONING:")
    if reasoning_idx != -1:
        end = score_idx if score_idx > reasoning_idx else len(raw)
        result["reasoning"] = raw[reasoning_idx + len("REASONING:"):end].strip()

    return result


def llm_judge_quality(
    query: str,
    generated_question: str,
    gold_question: str,
    llm: LLMClient,
) -> dict[str, Any]:
    """Score a generated clarifying question against a gold reference (1-5).

    Returns: {"score": int, "reasoning": str, "raw": str}
    """
    user_prompt = JUDGE_USER.format(
        query=query,
        gold_question=gold_question,
        candidate_question=generated_question,
    )
    raw = llm.chat(
        system_prompt=JUDGE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.1,
    )
    return _parse_judge_output(raw)


def llm_judge_quality_multi_ref(
    query: str,
    generated_question: str,
    gold_questions: list[str],
    llm: LLMClient,
) -> dict[str, Any]:
    """Score against each gold question, return the result with the max score."""
    if not gold_questions:
        return {"score": 0, "reasoning": "No gold questions", "raw": ""}

    best: dict[str, Any] = {"score": 0}
    for gold_q in gold_questions:
        result = llm_judge_quality(query, generated_question, gold_q, llm)
        if result["score"] > best["score"]:
            best = result
    return best


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

def evaluate_all(
    dataset_name: str,
    queries: list[AmbiguousQuery],
    generated_questions: list[str],
    predicted_ambiguous: list[bool],
    llm: LLMClient | None = None,
    skip_judge: bool = False,
) -> dict[str, Any]:
    """Run all applicable metrics for a dataset.

    Returns:
        {
            "dataset": str,
            "n_examples": int,
            "clarification_need": {...} or None,
            "judge_quality": {"mean": ..., "std": ..., "median": ..., "distribution": {...}} or None,
            "semantic_similarity": {"mean": ..., "std": ..., "median": ...},
            "per_example": [...]
        }
    """
    assert len(queries) == len(generated_questions) == len(predicted_ambiguous)

    result: dict[str, Any] = {
        "dataset": dataset_name,
        "n_examples": len(queries),
        "clarification_need": None,
        "judge_quality": None,
        "semantic_similarity": None,
        "per_example": [],
    }

    # --- Clarification Need F1 (skip for Qulac — all ambiguous) ---
    if dataset_name != "qulac":
        gold_labels = [q.is_ambiguous for q in queries]
        result["clarification_need"] = clarification_need_f1(predicted_ambiguous, gold_labels)

    # --- Filter to ambiguous-only for quality metrics ---
    amb_indices = [i for i, q in enumerate(queries) if q.is_ambiguous]

    # Group by topic for multi-ref datasets (Qulac, ClariQ)
    topic_gold_questions: dict[str, list[str]] = defaultdict(list)
    if dataset_name in ("qulac", "clariq"):
        for q in queries:
            if q.topic_id and q.gold_clarifying_question:
                topic_gold_questions[q.topic_id].append(q.gold_clarifying_question)

    # Compute per-example scores
    ss_scores: list[float] = []
    judge_scores: list[int] = []

    for i in amb_indices:
        q = queries[i]
        gen = generated_questions[i]
        per_ex: dict[str, Any] = {
            "example_id": q.example_id,
            "query": q.query,
            "generated": gen,
            "gold": q.gold_clarifying_question,
        }

        # Semantic similarity
        if dataset_name in ("qulac", "clariq") and q.topic_id:
            golds = topic_gold_questions.get(q.topic_id, [q.gold_clarifying_question])
            ss = semantic_similarity_multi_ref(gen, golds)
        else:
            ss = semantic_similarity(gen, q.gold_clarifying_question)
        per_ex["semantic_similarity"] = ss
        ss_scores.append(ss)

        # LLM judge
        if not skip_judge and llm is not None:
            if dataset_name in ("qulac", "clariq") and q.topic_id:
                golds = topic_gold_questions.get(q.topic_id, [q.gold_clarifying_question])
                judge = llm_judge_quality_multi_ref(q.query, gen, golds, llm)
            else:
                judge = llm_judge_quality(q.query, gen, q.gold_clarifying_question, llm)
            per_ex["judge_score"] = judge["score"]
            per_ex["judge_reasoning"] = judge.get("reasoning", "")
            judge_scores.append(judge["score"])

        result["per_example"].append(per_ex)

    # Aggregate semantic similarity
    if ss_scores:
        result["semantic_similarity"] = {
            "mean": statistics.mean(ss_scores),
            "std": statistics.stdev(ss_scores) if len(ss_scores) > 1 else 0.0,
            "median": statistics.median(ss_scores),
        }

    # Aggregate judge scores
    if judge_scores:
        dist = Counter(judge_scores)
        result["judge_quality"] = {
            "mean": statistics.mean(judge_scores),
            "std": statistics.stdev(judge_scores) if len(judge_scores) > 1 else 0.0,
            "median": statistics.median(judge_scores),
            "distribution": {k: dist.get(k, 0) for k in range(1, 6)},
        }

    return result
