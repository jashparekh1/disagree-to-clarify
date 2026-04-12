"""Run evaluation metrics on D2C output.

Usage:
    # Evaluate a single query against gold data inline:
    python -m scripts.run_eval --query "How do I deal with a Python crash?" \
        --gold-interpretations "fixing a runtime error" "handling a syntax error" \
        --gold-cq "Are you asking about a runtime crash or a syntax error?"

    # Evaluate a batch results file against a gold JSONL file:
    python -m scripts.run_eval \
        --results outputs/results.jsonl \
        --gold data/gold.jsonl \
        --output outputs/eval_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from d2c.pipeline import run_d2c
from d2c.llm import LLMClient
from eval.metrics import (
    interpretation_recall,
    interpretation_precision,
    clarifying_question_similarity,
    llm_judge_score,
    rouge_l,
    disambiguation_f1,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_system_interpretations(result) -> list[str]:
    """Pull the interpretation field from each agent in the final round."""
    return [r.interpretation for r in result.dialogue.rounds[-1]]


def evaluate_single(
    query: str,
    gold_interpretations: list[str],
    gold_cq: str | None = None,
    gold_answers: list[str] | None = None,
    model: str = "qwen3:1.7b",
    threshold: float = 0.6,
    use_llm_judge: bool = False,
) -> dict:
    """Run D2C on a single query and compute all available metrics.

    Args:
        query: The ambiguous query.
        gold_interpretations: List of gold disambiguated interpretations (AmbigQA).
        gold_cq: Gold clarifying question (ClarifyMT-Bench). Optional.
        gold_answers: Gold answers per interpretation (ASQA). Optional.
        model: Ollama model name.
        threshold: Semantic similarity threshold for recall/precision.
        use_llm_judge: Whether to run the LLM judge (slower).
    """
    logger.info("Running D2C pipeline...")
    result = run_d2c(query, model=model)
    system_ints = _extract_system_interpretations(result)
    cq = result.synthesizer_result.clarifying_question

    scores: dict = {
        "query": query,
        "clarifying_question": cq,
        "system_interpretations": system_ints,
    }

    # --- Interpretation metrics ---
    scores["interpretation_recall"] = interpretation_recall(
        system_ints, gold_interpretations, threshold=threshold
    )
    scores["interpretation_precision"] = interpretation_precision(
        system_ints, gold_interpretations, threshold=threshold
    )

    # --- Clarifying question metrics ---
    if gold_cq:
        scores["cq_similarity"] = clarifying_question_similarity(cq, gold_cq)

    if use_llm_judge:
        llm = LLMClient(model=model)
        scores["llm_judge"] = llm_judge_score(query, system_ints, cq, llm)

    # --- Answer quality metrics ---
    if gold_answers:
        # Approximate: use the clarifying question as a proxy for the "answer"
        # In a full setup you'd generate an answer after clarification
        scores["rouge_l"] = rouge_l(cq, gold_answers[0])
        scores["disambiguation_f1"] = disambiguation_f1(cq, gold_answers)

    return scores


def evaluate_batch(
    results_path: str,
    gold_path: str,
    output_path: str,
    threshold: float = 0.6,
    use_llm_judge: bool = False,
    model: str = "qwen3:1.7b",
) -> None:
    """Evaluate a batch results JSONL against a gold JSONL.

    Gold JSONL format per line:
    {
        "query": "...",
        "gold_interpretations": ["...", "..."],
        "gold_cq": "...",          # optional
        "gold_answers": ["...", "..."]  # optional
    }

    Results JSONL format: output of run_d2c_batch (D2CResult.to_dict()).
    """
    # Load gold by query
    gold_by_query: dict[str, dict] = {}
    with open(gold_path) as f:
        for line in f:
            obj = json.loads(line)
            gold_by_query[obj["query"]] = obj

    all_scores = []

    with open(results_path) as f:
        for line in f:
            result_obj = json.loads(line)
            query = result_obj["query"]

            gold = gold_by_query.get(query)
            if not gold:
                logger.warning("No gold found for query: %s", query)
                continue

            # Extract system interpretations from saved result
            final_round = result_obj["dialogue"]["rounds"][-1]
            system_ints = [r["interpretation"] for r in final_round]
            cq = result_obj["synthesizer_result"]["clarifying_question"]
            gold_ints = gold.get("gold_interpretations", [])
            gold_cq = gold.get("gold_cq")
            gold_answers = gold.get("gold_answers")

            scores: dict = {
                "query": query,
                "clarifying_question": cq,
                "interpretation_recall": interpretation_recall(system_ints, gold_ints, threshold),
                "interpretation_precision": interpretation_precision(system_ints, gold_ints, threshold),
            }

            if gold_cq:
                scores["cq_similarity"] = clarifying_question_similarity(cq, gold_cq)

            if use_llm_judge and gold_ints:
                llm = LLMClient(model=model)
                scores["llm_judge"] = llm_judge_score(query, system_ints, cq, llm)

            if gold_answers:
                scores["rouge_l"] = rouge_l(cq, gold_answers[0])
                scores["disambiguation_f1"] = disambiguation_f1(cq, gold_answers)

            all_scores.append(scores)
            logger.info("Scored: %s | recall=%.2f precision=%.2f",
                        query[:60], scores["interpretation_recall"], scores["interpretation_precision"])

    # Aggregate
    def _avg(key):
        vals = [s[key] for s in all_scores if key in s]
        return sum(vals) / len(vals) if vals else None

    aggregate = {
        "num_queries": len(all_scores),
        "avg_interpretation_recall": _avg("interpretation_recall"),
        "avg_interpretation_precision": _avg("interpretation_precision"),
        "avg_cq_similarity": _avg("cq_similarity"),
        "avg_rouge_l": _avg("rouge_l"),
        "avg_disambiguation_f1": _avg("disambiguation_f1"),
    }

    with open(output_path, "w") as out:
        for s in all_scores:
            out.write(json.dumps(s) + "\n")

    print("\n=== Aggregate Results ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nPer-query scores written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run D2C evaluation metrics")
    subparsers = parser.add_subparsers(dest="mode")

    # Single query mode
    single = subparsers.add_parser("single", help="Evaluate a single query")
    single.add_argument("--query", required=True)
    single.add_argument("--gold-interpretations", nargs="+", required=True)
    single.add_argument("--gold-cq", default=None)
    single.add_argument("--gold-answers", nargs="+", default=None)
    single.add_argument("--model", default="qwen3:1.7b")
    single.add_argument("--threshold", type=float, default=0.6)
    single.add_argument("--llm-judge", action="store_true")

    # Batch mode
    batch = subparsers.add_parser("batch", help="Evaluate a batch results file")
    batch.add_argument("--results", required=True)
    batch.add_argument("--gold", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--model", default="qwen3:1.7b")
    batch.add_argument("--threshold", type=float, default=0.6)
    batch.add_argument("--llm-judge", action="store_true")

    args = parser.parse_args()

    if args.mode == "single":
        scores = evaluate_single(
            query=args.query,
            gold_interpretations=args.gold_interpretations,
            gold_cq=args.gold_cq,
            gold_answers=args.gold_answers,
            model=args.model,
            threshold=args.threshold,
            use_llm_judge=args.llm_judge,
        )
        print("\n=== Evaluation Results ===")
        print(f"  Query              : {scores['query']}")
        print(f"  Clarifying Question: {scores['clarifying_question']}")
        print(f"  Recall             : {scores['interpretation_recall']:.4f}")
        print(f"  Precision          : {scores['interpretation_precision']:.4f}")
        if "cq_similarity" in scores:
            print(f"  CQ Similarity      : {scores['cq_similarity']:.4f}")
        if "llm_judge" in scores:
            print(f"  LLM Judge Score    : {scores['llm_judge']}")
        if "rouge_l" in scores:
            print(f"  ROUGE-L            : {scores['rouge_l']:.4f}")
        if "disambiguation_f1" in scores:
            print(f"  Disambiguation F1  : {scores['disambiguation_f1']:.4f}")

    elif args.mode == "batch":
        evaluate_batch(
            results_path=args.results,
            gold_path=args.gold,
            output_path=args.output,
            threshold=args.threshold,
            use_llm_judge=args.llm_judge,
            model=args.model,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
