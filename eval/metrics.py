"""Evaluation metrics for the D2C system.

Metrics implemented:
- semantic_similarity         : cosine similarity via sentence-transformers
- interpretation_recall       : fraction of gold interpretations covered by system (AmbigQA)
- interpretation_precision    : fraction of system interpretations matching a gold (AmbigQA)
- clarifying_question_quality : semantic sim + LLM judge score (ClarifyMT-Bench)
- rouge_l                     : ROUGE-L F1 for answer quality (AmbigQA/ASQA)
- disambiguation_f1           : token-level F1 averaged over gold interpretations (ASQA protocol)
- retrieval_metrics           : MRR and nDCG@20 (ClariQ)
"""

from __future__ import annotations

import logging
import json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------------

_sbert_model = None


def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


# ---------------------------------------------------------------------------
# 1. Semantic similarity (backbone for metrics 1, 2, 3)
# ---------------------------------------------------------------------------

def semantic_similarity(generated: str, reference: str) -> float:
    """Cosine similarity between two strings using sentence-transformers.

    Returns a float in [-1, 1]; typically [0, 1] for natural language.
    """
    from sentence_transformers import util
    model = _get_sbert()
    embs = model.encode([generated, reference], convert_to_tensor=True)
    return float(util.cos_sim(embs[0], embs[1]))


# ---------------------------------------------------------------------------
# 2 & 3. Interpretation Coverage (Recall) and Precision  — AmbigQA
# ---------------------------------------------------------------------------

def interpretation_recall(
    system_interpretations: list[str],
    gold_interpretations: list[str],
    threshold: float = 0.6,
) -> float:
    """Fraction of gold interpretations covered by at least one system interpretation.

    A gold interpretation is "covered" if semantic_similarity with any system
    interpretation exceeds *threshold*.
    """
    if not gold_interpretations:
        return 0.0
    covered = sum(
        any(
            semantic_similarity(sys_int, gold_int) >= threshold
            for sys_int in system_interpretations
        )
        for gold_int in gold_interpretations
    )
    return covered / len(gold_interpretations)


def interpretation_precision(
    system_interpretations: list[str],
    gold_interpretations: list[str],
    threshold: float = 0.6,
) -> float:
    """Fraction of system interpretations that match at least one gold interpretation."""
    if not system_interpretations:
        return 0.0
    valid = sum(
        any(
            semantic_similarity(sys_int, gold_int) >= threshold
            for gold_int in gold_interpretations
        )
        for sys_int in system_interpretations
    )
    return valid / len(system_interpretations)


# ---------------------------------------------------------------------------
# 4. Clarifying Question Quality — ClarifyMT-Bench
# ---------------------------------------------------------------------------

def clarifying_question_similarity(generated: str, gold: str) -> float:
    """Semantic similarity between a generated clarifying question and the gold."""
    return semantic_similarity(generated, gold)


def llm_judge_score(
    query: str,
    interpretations: list[str],
    clarifying_question: str,
    llm,  # LLMClient
) -> dict:
    """LLM-as-judge score for clarifying question quality.

    Returns a dict with keys: score (1-5), reasoning (str), covers_interpretations (bool).
    """
    from d2c.prompts import JUDGE_SYSTEM, JUDGE_USER

    formatted_ints = "\n".join(f"- {i}" for i in interpretations)
    user_prompt = JUDGE_USER.format(
        query=query,
        interpretations=formatted_ints,
        clarifying_question=clarifying_question,
    )

    raw = ""
    try:
        raw = llm.chat(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        # Strip markdown code fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("LLM judge failed: %s. Raw: %s", e, raw)
        return {"score": 1, "reasoning": str(e), "covers_interpretations": False}


# ---------------------------------------------------------------------------
# 5. Answer Quality — AmbigQA / ASQA
# ---------------------------------------------------------------------------

def rouge_l(generated: str, reference: str) -> float:
    """ROUGE-L F1 between a generated answer and a reference answer."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, generated)
    return result["rougeL"].fmeasure


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings (ASQA protocol)."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def disambiguation_f1(generated_answer: str, gold_answers: list[str]) -> float:
    """Disambiguation F1: max token-level F1 against each gold answer, then averaged.

    Follows the ASQA evaluation protocol (Stelmakh et al., 2022):
    for each gold answer, take the max token-F1 with the generated answer,
    then average across gold answers.
    """
    if not gold_answers:
        return 0.0
    scores = [_token_f1(generated_answer, gold) for gold in gold_answers]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 6. Downstream Retrieval — ClariQ
# ---------------------------------------------------------------------------

def retrieval_metrics(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 20,
) -> dict:
    """Compute MRR and nDCG@k given a ranked list of doc IDs and a relevance set.

    Args:
        ranked_doc_ids: Ordered list of retrieved document IDs (most relevant first).
        relevant_doc_ids: Set of ground-truth relevant document IDs.
        k: Cutoff for nDCG (default 20, matching ClariQ protocol).

    Returns:
        {"mrr": float, "ndcg": float}
    """
    import math

    # MRR — reciprocal rank of the first relevant doc
    mrr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / rank
            break

    # nDCG@k
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant docs ranked at the top (up to k)
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"mrr": mrr, "ndcg": ndcg}
