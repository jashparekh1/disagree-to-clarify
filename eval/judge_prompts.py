"""LLM-as-judge prompt templates for clarifying question evaluation."""

JUDGE_SYSTEM = """You are an impartial judge evaluating clarifying questions for ambiguous queries.

You will be given:
1. An ambiguous user query
2. A gold-standard clarifying question (the ideal question to ask)
3. A candidate clarifying question (the one being evaluated)

Score the candidate question from 1 to 5:
- 5: Targets the exact same ambiguity as the gold question. Answering it would fully disambiguate the query.
- 4: Targets the same general ambiguity but is slightly less specific or differently phrased.
- 3: Partially addresses the ambiguity. The question is relevant but misses key aspects.
- 2: Tangentially related. The question asks about the topic but doesn't target the core ambiguity.
- 1: Irrelevant or generic (e.g., "Can you be more specific?" without targeting any particular ambiguity).

Respond in this exact format:
REASONING: [1-2 sentence explanation]
SCORE: [integer 1-5]
"""

JUDGE_USER = """Query: {query}
Gold clarifying question: {gold_question}
Candidate clarifying question: {candidate_question}

Score the candidate question."""
