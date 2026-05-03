"""Unified evaluation metrics for D2C.

Three metrics:
1. Clarification Need Prediction (F1)
2. Clarifying Question Quality (LLM-as-Judge)
3. Semantic Similarity to Gold (Embedding Cosine)
"""

from __future__ import annotations

import json as _json
import logging
import statistics
from collections import Counter, defaultdict
from typing import Any, TYPE_CHECKING

from eval.judge_prompts import JUDGE_SYSTEM, JUDGE_USER

if TYPE_CHECKING:
    from d2c.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric 1: Ambiguity Detection (F1)
# ---------------------------------------------------------------------------

def clarification_need_f1(preds: list[bool], golds: list[bool]) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 for 'Is this query ambiguous?'."""
    tp = sum(1 for p, g in zip(preds, golds) if p and g)
    fp = sum(1 for p, g in zip(preds, golds) if p and not g)
    fn = sum(1 for p, g in zip(preds, golds) if not p and g)
    tn = sum(1 for p, g in zip(preds, golds) if not p and not g)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": (tp + tn) / len(golds) if golds else 0.0,
    }

def internal_divergence_score(interpretations: list[str]) -> float:
    """Calculate the semantic spread between agents (1 - avg cosine sim)."""
    if len(interpretations) < 2:
        return 0.0
    
    sims = []
    for i in range(len(interpretations)):
        for j in range(i + 1, len(interpretations)):
            sims.append(semantic_similarity(interpretations[i], interpretations[j]))
    
    return 1.0 - statistics.mean(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# Metric 2: Clarifying Question Quality (LLM-as-Judge)
# ---------------------------------------------------------------------------

def llm_judge_quality(
    query: str,
    generated_question: str,
    gold_question: str,
    llm: LLMClient,
    context: str | None = None,
) -> dict[str, Any]:
    """Score a generated clarifying question using an optimized single-pass judge."""
    context_block = f"CONTEXT (The story/background):\n{context}\n\n" if context else ""
    user_prompt = JUDGE_USER.format(
        query=query,
        gold_question=gold_question,
        candidate_question=generated_question,
        context_block=context_block
    )
    
    judge_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "covers_interpretations": {"type": "boolean"},
            "reasoning_summary": {"type": "string"},
        },
        "required": ["score", "covers_interpretations"]
    }
    
    raw = llm.chat(
        system_prompt=JUDGE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=200,
        format_schema=judge_schema
    )
    
    result: dict[str, Any] = {"raw": raw, "score": 0, "reasoning": "", "covers": False}
    try:
        data = _json.loads(raw)
        result["score"] = max(1, min(5, int(data.get("score", 0))))
        result["covers"] = bool(data.get("covers_interpretations", False))
        result["reasoning"] = data.get("reasoning_summary", "")
    except (_json.JSONDecodeError, ValueError, KeyError):
        pass
        
    return result


def llm_judge_quality_multi_ref(
    query: str,
    generated_question: str,
    gold_questions: list[str],
    llm: LLMClient,
    context: str | None = None,
) -> dict[str, Any]:
    """Score against all gold references and return the best result."""
    if not gold_questions:
        return {"score": 0, "reasoning": "No gold questions", "raw": ""}

    best: dict[str, Any] = {"score": 0}
    for gold_q in gold_questions:
        res = llm_judge_quality(query, generated_question, gold_q, llm, context=context)
        if res["score"] > best["score"]:
            best = res
        if best["score"] == 5:
            break
    return best


# ---------------------------------------------------------------------------
# Metric 3: Semantic Similarity (Embedding-based)
# ---------------------------------------------------------------------------

_ST_MODEL = None

def semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between sentence embeddings."""
    global _ST_MODEL
    from sentence_transformers import SentenceTransformer, util
    
    if _ST_MODEL is None:
        print("Loading SentenceTransformer (one-time cost)...")
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        
    emb1 = _ST_MODEL.encode(text1, convert_to_tensor=True)
    emb2 = _ST_MODEL.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def semantic_similarity_multi_ref(generated: str, gold_list: list[str]) -> float:
    """Return the max semantic similarity against a list of references."""
    if not gold_list:
        return 0.0
    return max(semantic_similarity(generated, gold) for gold in gold_list)
