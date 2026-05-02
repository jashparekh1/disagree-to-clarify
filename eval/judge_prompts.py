"""LLM-as-judge prompt templates for clarifying question evaluation."""

JUDGE_SYSTEM = """You are an expert evaluator of clarifying questions.

Your task:
Evaluate a clarifying question based on how well it helps resolve the ambiguity of the user's query compared to a gold-standard reference.

RUBRIC:
1. INFORMATION GAIN (Primary): Does answering this question actually help the system decide which interpretation the user meant?
2. UTILITY: Targeted open-ended questions are acceptable (3-4 stars) if they are useful, though disjunctive choices (e.g., "Do you mean A or B?") are preferred (5 stars).
3. NATURALNESS: The question should sound like a helpful human. Minor JSON formatting errors should NOT lower the quality score if the question itself is good.

Output ONLY a JSON object with exactly these fields:
{
  "score": <integer 1-5>,
  "covers_interpretations": <true or false>,
  "reasoning_summary": "<one sentence focusing on utility and information gain>"
}

SCORING:
- 5: Perfect — matches or improves on the gold question.
- 4: Strong — nearly as effective as the gold.
- 3: Partial — addresses some but not all ambiguity.
- 2: Tangential — loosely related but misses the point.
- 1: Irrelevant — fails to address the ambiguity.
"""

JUDGE_USER = """Query: {query}
Gold clarifying question: {gold_question}
Candidate clarifying question: {candidate_question}

Evaluate the generated question based on how well it helps resolve the ambiguity between the possible interpretations (grounded by the gold standard)."""
