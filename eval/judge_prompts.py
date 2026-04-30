"""LLM-as-judge prompt templates for clarifying question evaluation."""

JUDGE_SYSTEM = """You are an expert evaluator.

Output your response in exactly this order:
1. JSON: Output the objective result inside a single JSON block.
2. REASONING: After the JSON, provide a detailed comparison and explanation.

JSON FORMAT:
{
  "reasoning_summary": "1-sentence summary",
  "score": [1-5],
  "covers_interpretations": [true/false]
}

SCORING:
- 5: Perfect match/resolution.
- 4: Strong resolution.
- 3: Partial resolution.
- 2: Tangential.
- 1: Irrelevant.
"""

JUDGE_USER = """Query: {query}
Gold clarifying question: {gold_question}
Candidate clarifying question: {candidate_question}

Score the candidate question."""
