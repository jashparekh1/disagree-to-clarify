"""LLM-as-judge prompt templates for clarifying question evaluation."""

JUDGE_SYSTEM = """You are an expert evaluator. Output ONLY a JSON object with exactly these fields:
{
  "score": <integer 1-5>,
  "covers_interpretations": <true or false>,
  "reasoning_summary": "<one sentence>"
}

SCORING (how well the candidate clarifying question resolves the ambiguity compared to the gold):
- 5: Perfect — matches or improves on the gold question.
- 4: Strong — nearly as effective as the gold.
- 3: Partial — addresses some but not all ambiguity.
- 2: Tangential — loosely related but misses the point.
- 1: Irrelevant — fails to address the ambiguity.

covers_interpretations: true if the candidate question would help distinguish between the main possible interpretations of the query."""

JUDGE_USER = """Query: {query}
Gold clarifying question: {gold_question}
Candidate clarifying question: {candidate_question}

Score the candidate question."""
