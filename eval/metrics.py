"""Evaluation metrics stubs — Akash will implement these."""


def semantic_similarity(generated: str, reference: str) -> float:
    """Compute semantic similarity between generated and reference clarifying question."""
    raise NotImplementedError("Akash will implement this")


def llm_judge_score(query: str, clarifying_question: str, gold_question: str) -> dict:
    """Use LLM-as-judge to score clarifying question quality."""
    raise NotImplementedError("Akash will implement this")
