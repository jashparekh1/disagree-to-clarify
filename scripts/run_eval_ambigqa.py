"""AmbigQA Evaluation Script.

Runs D2C on AmbigQA queries and evaluates using LLM-as-a-judge.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.metrics import llm_judge_score

logger = logging.getLogger(__name__)


def parse_ambigqa_item(item: dict) -> tuple[str, list[str]]:
    """Parse query and disambiguations from AmbigQA light JSON format."""
    query = item.get("question")
    annotations = item.get("annotations", [])
    interpretations = []
    
    for ann in annotations:
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    
    # If no multipleQAs, it might be a single-answer question or have a different structure
    # For evaluation of clarifying questions, we only care about cases with multiple interpretations
    return query, interpretations


def process_query(item: dict, model: str, rounds: int, max_tokens: int, judge_model: str, variant: str) -> dict:
    """Run D2C and then judge the result."""
    query, interpretations = parse_ambigqa_item(item)
    if not query:
        return {"error": "Invalid AmbigQA item structure"}
    
    if len(interpretations) < 2:
        return {"query": query, "status": "skipped_not_ambiguous"}

    try:
        # 1. Run D2C
        result = run_d2c(query, model=model, num_rounds=rounds, max_tokens=max_tokens, variant=variant)
        clarifying_question = result.synthesizer_result.clarifying_question
        
        # 2. Judge
        judge_llm = LLMClient(model=judge_model)
        eval_result = llm_judge_score(query, interpretations, clarifying_question, judge_llm)
        
        # 3. Combine
        record = {
            "query": query,
            "interpretations": interpretations,
            "generated_question": clarifying_question,
            "eval": eval_result,
            "model": model,
            "timestamp": result.timestamp
        }
        return record
    except Exception as e:
        logger.exception(f"Failed to process query: {query}")
        return {"query": query, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D2C on AmbigQA dataset")
    parser.add_argument("--input", required=True, help="Input JSONL file (AmbigQA format)")
    parser.add_argument("--output", required=True, help="Output JSONL file for results")
    parser.add_argument("--model", default="qwen3:1.7b", help="D2C model name")
    parser.add_argument("--judge-model", default="llama3.1:8b", help="LLM Judge model name")
    parser.add_argument("--rounds", type=int, default=3, help="Number of dialogue rounds")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per D2C call")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--variant", default="speech_act", choices=["original", "speech_act"], help="D2C agent variant (default: SAT-grounded; use 'original' for the pre-theory ablation)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load items
    items = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} items from {args.input}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_query, item, args.model, args.rounds, args.max_tokens, args.judge_model, args.variant)
            for item in items
        ]
        
        with open(args.output, "w") as out:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                res = future.result()
                results.append(res)
                out.write(json.dumps(res) + "\n")
                out.flush()

    # Calculate aggregate metrics
    valid_results = [r for r in results if "eval" in r]
    if valid_results:
        avg_score = sum(r["eval"]["score"] for r in valid_results) / len(valid_results)
        coverage_pct = sum(1 for r in valid_results if r["eval"]["covers_interpretations"]) / len(valid_results) * 100
        
        print("\n" + "="*40)
        print("EVALUATION SUMMARY")
        print("="*40)
        print(f"Total processed: {len(valid_results)} / {len(items)}")
        print(f"Average Score: {avg_score:.2f} / 5.0")
        print(f"Interpretation Coverage: {coverage_pct:.1f}%")
        print(f"Results saved to: {args.output}")
        print("="*40)


if __name__ == "__main__":
    main()
