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

_BERT_SCORER = None


def _get_embed_model():
    """Load the sentence-transformer model once, cache globally."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print("Loading SentenceTransformer (one-time cost)...")
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


def _get_bert_scorer(model_type: str = "roberta-large"):
    """Load the BERTScorer once, cache globally."""
    global _BERT_SCORER
    if _BERT_SCORER is None:
        from bert_score import BERTScorer
        _BERT_SCORER = BERTScorer(model_type=model_type, lang="en")
    return _BERT_SCORER


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


def internal_divergence_score(interpretations: list[str]) -> float:
    """Calculate the average pairwise semantic distance (1 - cosine similarity).
    
    A score of 0.0 means perfect agreement; 1.0 means total divergence.
    """
    if len(interpretations) < 2:
        return 0.0
    
    model = _get_embed_model()
    embeddings = model.encode(interpretations, normalize_embeddings=True)
    
    distances = []
    n = len(interpretations)
    for i in range(n):
        for j in range(i + 1, n):
            # Cosine similarity is dot product of normalized embeddings
            sim = float(embeddings[i] @ embeddings[j])
            distances.append(1.0 - max(0.0, min(1.0, sim)))
            
    return sum(distances) / len(distances) if distances else 0.0


def semantic_similarity_batch(
    generated: list[str],
    references: list[list[str]],
) -> dict[str, float]:
    """Compute max cosine similarity for a batch of predictions against multi-ref golds."""
    scores = []
    for gen, refs in zip(generated, references):
        if not gen or not refs:
            scores.append(0.0)
            continue
        scores.append(semantic_similarity_multi_ref(gen, refs))
    
    return {
        "mean": statistics.mean(scores) if scores else 0.0,
        "median": statistics.median(scores) if scores else 0.0,
    }


def bert_score_compute(
    generated: list[str],
    references: list[list[str]],
    model_type: str = "roberta-large",
) -> dict[str, float]:
    """Compute BERTScore (P, R, F1) using a cached Scorer.

    Args:
        generated: List of generated questions.
        references: List of lists of gold questions (multi-reference).
    """
    scorer = _get_bert_scorer(model_type)
    
    # scorer.score expects references as list[list[str]] where inner list
    # are multiple references for EACH candidate.
    P, R, F1 = scorer.score(generated, references)
    
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def mrr_score(relevant_indices: list[int], k: int = 20) -> float:
    """Compute Reciprocal Rank.
    Args:
        relevant_indices: Indices (0-based) of relevant items in the ranked list.
        k: Cutoff.
    """
    for idx in sorted(relevant_indices):
        if idx < k:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_score(relevance_scores: list[float], k: int = 20) -> float:
    """Compute Discounted Cumulative Gain (nDCG).
    Args:
        relevance_scores: Relevance scores in the ranked order.
    """
    import math
    
    def dcg(scores):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 2: Clarifying Question Quality (LLM-as-Judge)
# ---------------------------------------------------------------------------

def llm_judge_quality(
    query: str,
    generated_question: str,
    gold_question: str,
    llm: LLMClient,
) -> dict[str, Any]:
    """Score a generated clarifying question using a 2-step Reasoning->Scoring pipeline.
    
    Step 1: Get detailed qualitative reasoning.
    Step 2: Get strict quantitative JSON based on that reasoning.
    """
    user_prompt = JUDGE_USER.format(
        query=query,
        gold_question=gold_question,
        candidate_question=generated_question,
    )
    
    # STEP 1: REASONING (No JSON yet, just thinking)
    reasoning_sys = "You are a critical evaluator. Analyze the candidate question against the gold standard and the query. Identify specific strengths and weaknesses. Be thorough but concise (under 200 words)."
    full_reasoning = llm.chat(
        system_prompt=reasoning_sys,
        user_prompt=user_prompt,
        temperature=0.7, # Higher temp for better reasoning diversity
        max_tokens=500,
        strip_thinking=False,
    )
    
    # STEP 2: SCORING (Strict JSON based on reasoning)
    scoring_sys = f"You are a robotic scoring script. Based on the following reasoning, output ONLY the final objective metrics in JSON format.\n\nREASONING TO USE:\n{full_reasoning}"
    
    judge_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "covers_interpretations": {"type": "boolean"}
        },
        "required": ["score", "covers_interpretations"]
    }
    
    raw_json = llm.chat(
        system_prompt=scoring_sys,
        user_prompt="Output the JSON metrics now.",
        temperature=0.0,
        max_tokens=100,
        format_schema=judge_schema
    )
    
    # Combine results
    import json
    result: dict[str, Any] = {"raw": f"REASONING:\n{full_reasoning}\n\nRESULT:\n{raw_json}", "score": 0, "reasoning": full_reasoning, "covers": False}
    try:
        data = json.loads(raw_json)
        result["score"] = max(1, min(5, int(data.get("score", 0))))
        result["covers"] = bool(data.get("covers_interpretations", False))
    except (json.JSONDecodeError, ValueError):
        pass
        
    return result


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
