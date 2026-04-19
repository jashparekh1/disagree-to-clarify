"""Run D2C on ClarifyMT-Bench and evaluate.

Usage:
    # Run on a small sample (10 items)
    python -m scripts.run_clarifymt --limit 10 --output outputs/clarifymt_results.jsonl

    # Run on a specific category
    python -m scripts.run_clarifymt --category "Lexical Ambiguity" --output outputs/clarifymt_lexical.jsonl

    # Run on specific user types only
    python -m scripts.run_clarifymt --user-type Precise --user-type Partial-Vague --limit 20

    # Evaluate only (skip D2C, just score existing results)
    python -m scripts.run_clarifymt --eval-only --input outputs/clarifymt_results.jsonl

    # Resume an interrupted run
    python -m scripts.run_clarifymt --output outputs/clarifymt_results.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from d2c.data import load_clarifymt_bench
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation logic (follows ClarifyMT-Bench protocol)
# ---------------------------------------------------------------------------

def judge_clarifying_question(
    query: str,
    generated_cq: str,
    gold_cq: str,
    user_response: str,
    category: str,
    explanation: str,
    llm: LLMClient,
) -> dict:
    """Use LLM-as-judge to score the generated clarifying question.

    Returns a dict with:
        - relevance (1-5): Does the question target the actual ambiguity?
        - specificity (1-5): Is the question specific (not generic)?
        - coverage (1-5): Does it cover the key ambiguity identified in the gold?
        - overall (1-5): Overall quality.
        - reasoning: Free-text explanation.
    """
    system = """You are an expert evaluator of clarifying questions for ambiguous queries.

Score the generated clarifying question on these dimensions (1-5 each):
- RELEVANCE: Does the question address the actual ambiguity in the original query?
- SPECIFICITY: Is the question specific and targeted (not generic like "can you clarify?")?
- COVERAGE: Does it cover the same ambiguity that the reference question addresses?
- OVERALL: Overall quality as a clarifying question.

Respond in this EXACT format:
RELEVANCE: [1-5]
SPECIFICITY: [1-5]
COVERAGE: [1-5]
OVERALL: [1-5]
REASONING: [1-2 sentence explanation]"""

    user_prompt = f"""Original ambiguous query: {query}
Ambiguity category: {category}
Ambiguity explanation: {explanation}

Reference clarifying question: {gold_cq}
User's clarified response (to reference): {user_response}

Generated clarifying question: {generated_cq}

Score the generated clarifying question."""

    raw = llm.chat(system_prompt=system, user_prompt=user_prompt, temperature=0.1)
    return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> dict:
    """Parse the judge's structured output."""
    result: dict = {"raw": raw}
    for field in ["RELEVANCE", "SPECIFICITY", "COVERAGE", "OVERALL"]:
        marker = f"{field}:"
        idx = raw.find(marker)
        if idx != -1:
            # Grab everything after the marker up to the next newline
            rest = raw[idx + len(marker):].strip()
            # Take the first token as the score
            score_str = rest.split()[0] if rest else "0"
            try:
                result[field.lower()] = int(score_str)
            except ValueError:
                result[field.lower()] = 0
        else:
            result[field.lower()] = 0

    # Reasoning
    r_idx = raw.find("REASONING:")
    if r_idx != -1:
        result["reasoning"] = raw[r_idx + len("REASONING:"):].strip()
    else:
        result["reasoning"] = ""

    return result


# ---------------------------------------------------------------------------
# ClarifyMT-Bench accuracy metric (from their eval.py)
# ---------------------------------------------------------------------------

def clarifymt_accuracy(user_type: str, response: str) -> bool:
    """ClarifyMT-Bench's own accuracy metric.

    For Precise/Refusal user types: correct if the model answers directly.
    For other types (Partial-Vague, Off-Focus, etc.): correct if model asks for clarification.

    In our case, D2C *always* produces a clarifying question, so we're mainly
    interested in the quality of that question (judged above). This metric is
    included for compatibility with their evaluation protocol.
    """
    response = response.strip().lower()
    if user_type in ("Precise", "Refusal"):
        return "answer" in response
    return "clarify" in response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_and_evaluate(
    data_path: str,
    output_path: str,
    model: str,
    num_rounds: int,
    categories: list[str] | None,
    user_types: list[str] | None,
    limit: int | None,
    resume: bool,
    judge: bool,
) -> None:
    """Run D2C on ClarifyMT-Bench items and optionally judge results."""
    items = load_clarifymt_bench(
        path=data_path,
        categories=categories,
        user_types=user_types,
        limit=limit,
    )
    print(f"Loaded {len(items)} items from {data_path}")

    # Resume support: load already-processed queries
    done_queries: set[str] = set()
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                done_queries.add(obj.get("query", ""))
        print(f"Resuming: {len(done_queries)} already processed")

    llm = LLMClient(model=model) if judge else None

    with open(output_path, "a") as out:
        for item in tqdm(items, desc="ClarifyMT-Bench"):
            if item.query in done_queries:
                continue

            try:
                result = run_d2c(item.query, model=model, num_rounds=num_rounds)
                record = {
                    **result.to_dict(),
                    "gold_clarifying_question": item.gold_clarifying_question,
                    "user_response": item.user_response,
                    "category": item.category,
                    "user_type": item.user_type,
                    "explanation": item.explanation,
                }

                if judge and llm:
                    scores = judge_clarifying_question(
                        query=item.query,
                        generated_cq=result.synthesizer_result.clarifying_question,
                        gold_cq=item.gold_clarifying_question,
                        user_response=item.user_response,
                        category=item.category,
                        explanation=item.explanation,
                        llm=llm,
                    )
                    record["judge_scores"] = scores

                out.write(json.dumps(record) + "\n")
                out.flush()
            except Exception:
                logger.exception("Failed on query: %s", item.query)


def evaluate_results(input_path: str) -> None:
    """Evaluate an existing results file and print aggregate metrics."""
    results: list[dict] = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        print("No results to evaluate.")
        return

    # Aggregate judge scores if present
    has_judges = "judge_scores" in results[0]
    cat_scores: dict[str, list[dict]] = defaultdict(list)
    utype_scores: dict[str, list[dict]] = defaultdict(list)
    all_scores: list[dict] = []

    for r in results:
        cat = r.get("category", "unknown")
        ut = r.get("user_type", "unknown")

        if has_judges:
            scores = r["judge_scores"]
            cat_scores[cat].append(scores)
            utype_scores[ut].append(scores)
            all_scores.append(scores)

    print(f"\nResults: {len(results)} items")
    print("=" * 70)

    if has_judges:
        dims = ["relevance", "specificity", "coverage", "overall"]

        print("\n--- Scores by Category ---")
        print(f"{'Category':<30} " + " ".join(f"{d:>12}" for d in dims))
        print("-" * 80)
        for cat in sorted(cat_scores):
            scores_list = cat_scores[cat]
            avgs = {d: sum(s.get(d, 0) for s in scores_list) / len(scores_list) for d in dims}
            print(f"{cat:<30} " + " ".join(f"{avgs[d]:>12.2f}" for d in dims))

        print("\n--- Scores by User Type ---")
        print(f"{'User Type':<30} " + " ".join(f"{d:>12}" for d in dims))
        print("-" * 80)
        for ut in sorted(utype_scores):
            scores_list = utype_scores[ut]
            avgs = {d: sum(s.get(d, 0) for s in scores_list) / len(scores_list) for d in dims}
            print(f"{ut:<30} " + " ".join(f"{avgs[d]:>12.2f}" for d in dims))

        print("\n--- Overall ---")
        avgs = {d: sum(s.get(d, 0) for s in all_scores) / len(all_scores) for d in dims}
        print(" ".join(f"{d}: {avgs[d]:.2f}" for d in dims))
    else:
        print("No judge scores found. Re-run with --judge to score results.")

    # Count by category and user type
    cat_counts = defaultdict(int)
    ut_counts = defaultdict(int)
    for r in results:
        cat_counts[r.get("category", "unknown")] += 1
        ut_counts[r.get("user_type", "unknown")] += 1

    print("\n--- Distribution ---")
    print("Categories:", dict(sorted(cat_counts.items())))
    print("User types:", dict(sorted(ut_counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run D2C on ClarifyMT-Bench and evaluate"
    )
    parser.add_argument(
        "--data", default="data/clarifymt_bench.jsonl",
        help="Path to ClarifyMT-Bench JSONL",
    )
    parser.add_argument(
        "--output", default="outputs/clarifymt_results.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model")
    parser.add_argument("--rounds", type=int, default=3, help="Dialogue rounds")
    parser.add_argument(
        "--category", action="append", dest="categories",
        help="Filter by category (repeatable)",
    )
    parser.add_argument(
        "--user-type", action="append", dest="user_types",
        help="Filter by user type (repeatable)",
    )
    parser.add_argument("--limit", type=int, help="Max items to process")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed queries")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-judge scoring")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip D2C run, just evaluate existing results",
    )
    parser.add_argument(
        "--input", dest="eval_input",
        help="Input file for --eval-only mode",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.eval_only:
        path = args.eval_input or args.output
        evaluate_results(path)
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_and_evaluate(
        data_path=args.data,
        output_path=args.output,
        model=args.model,
        num_rounds=args.rounds,
        categories=args.categories,
        user_types=args.user_types,
        limit=args.limit,
        resume=args.resume,
        judge=args.judge,
    )
    print(f"\nResults saved to {args.output}")
    print("Run with --eval-only to see aggregate metrics.")


if __name__ == "__main__":
    main()
