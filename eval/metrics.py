"""Unified evaluation metrics for D2C.

Three metrics:
1. Clarification Need Prediction (F1)
2. Clarifying Question Quality (LLM-as-Judge)
3. Semantic Similarity to Gold (Embedding Cosine)
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass

from d2c.llm import LLMClient
from d2c.prompts import JUDGE_SYSTEM, JUDGE_USER

logger = logging.getLogger(__name__)

_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embed_model():
    """Load the sentence-transformer model once, cache globally."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
        except ImportError:
            logger.warning("sentence_transformers not found. Semantic similarity will be skipped.")
            return None
    return _EMBED_MODEL


# ---------------------------------------------------------------------------
# Metric 1: Clarification Need Prediction (F1)
# ---------------------------------------------------------------------------

def clarification_need_f1(
    gold_ambiguous: list[bool],
    predicted_ambiguous: list[bool],
) -> dict:
    """Compute F1, Precision, and Recall for binary ambiguity detection."""
    tp = sum(g and p for g, p in zip(gold_ambiguous, predicted_ambiguous))
    fp = sum(not g and p for g, p in zip(gold_ambiguous, predicted_ambiguous))
    fn = sum(g and not p for g, p in zip(gold_ambiguous, predicted_ambiguous))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(gold_ambiguous),
    }


# ---------------------------------------------------------------------------
# Metric 2: Clarifying Question Quality (LLM-as-Judge)
# ---------------------------------------------------------------------------

def llm_judge_quality(
    query: str,
    generated_question: str,
    gold_question: str,
    llm: LLMClient,
) -> dict:
    """Score a question (1-5) relative to a reference question."""
    if not generated_question.strip():
        return {"score": 1, "reasoning": "Empty question", "covers_interpretations": False}

    # We use the gold question as a proxy for 'correct interpretations'
    # in the standard prompt schema.
    user_prompt = JUDGE_USER.format(
        query=query,
        interpretations=f"Reference Question: {gold_question}",
        clarifying_question=generated_question,
    )

    try:
        raw = llm.chat(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return _parse_judge_json(raw)
    except Exception as e:
        logger.error("Judge failed: %s", e)
        return {"score": 1, "reasoning": str(e), "covers_interpretations": False}


def llm_judge_quality_multi_ref(
    query: str,
    generated_question: str,
    gold_questions: list[str],
    llm: LLMClient,
) -> dict:
    """Score relative to multiple valid reference questions."""
    refs = "\n".join(f"- {q}" for q in gold_questions)
    user_prompt = JUDGE_USER.format(
        query=query,
        interpretations=f"Valid reference questions:\n{refs}",
        clarifying_question=generated_question,
    )

    try:
        raw = llm.chat(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return _parse_judge_json(raw)
    except Exception as e:
        logger.error("Judge failed: %s", e)
        return {"score": 1, "reasoning": str(e), "covers_interpretations": False}


def _parse_judge_json(raw: str) -> dict:
    """Defensive JSON parsing for judge output."""
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    
    try:
        return json.loads(raw.strip())
    except:
        # Fallback if the LLM produced junk
        return {"score": 1, "reasoning": "Parse failure", "covers_interpretations": False}


# ---------------------------------------------------------------------------
# Metric 3: Semantic Similarity
# ---------------------------------------------------------------------------

def semantic_similarity(generated: str, gold: str) -> float | None:
    """Cosine similarity between sentence embeddings."""
    if not generated.strip() or not gold.strip():
        return 0.0

    model = _get_embed_model()
    if model is None:
        return None

    embeddings = model.encode([generated, gold], normalize_embeddings=True)
    score = float(embeddings[0] @ embeddings[1])
    return max(0.0, score)


def semantic_similarity_multi_ref(
    generated: str,
    gold_questions: list[str],
) -> float | None:
    """Max cosine similarity across multiple reference questions."""
    if not gold_questions:
        return 0.0
    scores = [semantic_similarity(generated, gold) for gold in gold_questions]
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return None
    return max(valid_scores)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def evaluate_all(
    dataset_name: str,
    queries: list,  # List of dataset objects
    generated_questions: list[str],
    predicted_ambiguous: list[bool],
    llm: LLMClient | None = None,
    skip_judge: bool = False,
) -> dict:
    """Run full evaluation suite on a batch of results."""
    gold_ambiguous = [q.is_ambiguous for q in queries]
    
    # 1. Prediction Metrics
    f1_results = clarification_need_f1(gold_ambiguous, predicted_ambiguous)
    
    # 2. Question Quality (only for items where we generated a question)
    judge_scores = []
    ss_scores = []
    coverage_count = 0
    
    # Pre-map topic groups for multi-ref evaluation
    topic_gold_questions = {}
    if dataset_name in ("qulac", "clariq"):
        from collections import defaultdict
        groups = defaultdict(list)
        for q in queries:
            groups[q.topic_id].append(q.gold_clarifying_question)
        topic_gold_questions = dict(groups)

    results_per_example = []

    for q, gen in zip(queries, generated_questions):
        if not q.is_ambiguous:
            continue
            
        per_ex = {"query": q.query, "generated": gen, "gold": q.gold_clarifying_question}
        
        # Semantic Similarity
        if dataset_name in ("qulac", "clariq") and q.topic_id:
            golds = topic_gold_questions.get(q.topic_id, [q.gold_clarifying_question])
            ss = semantic_similarity_multi_ref(gen, golds)
        else:
            ss = semantic_similarity(gen, q.gold_clarifying_question)
        per_ex["semantic_similarity"] = ss
        if ss is not None:
            ss_scores.append(ss)

        # LLM judge
        if not skip_judge and llm is not None:
            if dataset_name in ("qulac", "clariq") and q.topic_id:
                golds = topic_gold_questions.get(q.topic_id, [q.gold_clarifying_question])
                judge = llm_judge_quality_multi_ref(q.query, gen, golds, llm)
            else:
                judge = llm_judge_quality(q.query, gen, q.gold_clarifying_question, llm)
            
            score = judge.get("score", 1)
            judge_scores.append(score)
            if judge.get("covers_interpretations"):
                coverage_count += 1
            per_ex["judge"] = judge

        results_per_example.append(per_ex)

    result = {
        "f1": f1_results,
        "semantic_similarity": {
            "mean": statistics.mean(ss_scores) if ss_scores else 0.0,
            "std": statistics.stdev(ss_scores) if len(ss_scores) > 1 else 0.0,
        },
        "examples": results_per_example
    }

    if judge_scores:
        dist = Counter(judge_scores)
        result["judge_quality"] = {
            "mean": statistics.mean(judge_scores),
            "std": statistics.stdev(judge_scores) if len(judge_scores) > 1 else 0.0,
            "median": statistics.median(judge_scores),
            "distribution": {k: dist.get(k, 0) for k in range(1, 6)},
        }
        result["coverage_rate"] = coverage_count / len(judge_scores)

    return result
