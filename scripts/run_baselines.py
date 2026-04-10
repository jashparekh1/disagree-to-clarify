"""Comparison Baselines for AmbigQA.

Implemented:
1. Vanilla: Single-turn LLM call.
2. Parallel: Agents generate 1 round in parallel, then synthesize (No Dialogue).
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from d2c.agents import Agent, AgentRole
from d2c.dialogue import DialogueResult
from d2c.llm import LLMClient
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER
from d2c.synthesizer import synthesize
from eval.metrics import llm_judge_score

logger = logging.getLogger(__name__)


def parse_ambigqa_item(item: dict) -> tuple[str, list[str]]:
    """Reuse the AmbigQA light parser."""
    query = item.get("question")
    annotations = item.get("annotations", [])
    interpretations = []
    for ann in annotations:
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    return query, interpretations


def run_vanilla(query: str, model: str, max_tokens: int) -> str:
    """Baseline 1: Single-turn LLM call."""
    llm = LLMClient(model=model)
    user_prompt = VANILLA_CQG_USER.format(query=query)
    return llm.chat(
        system_prompt=VANILLA_CQG_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )


def run_parallel_only(query: str, model: str, max_tokens: int) -> str:
    """Baseline 2: Agents generate initial interpretation in parallel, then synthesize."""
    llm = LLMClient(model=model)
    agents = [Agent(role, llm, max_tokens=max_tokens) for role in AgentRole]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.respond_initial, query) for agent in agents]
        round_0 = [f.result() for f in futures]

    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1)
    synth = synthesize(query, dialogue, llm, max_tokens=max_tokens)
    return synth.clarifying_question


def process_baseline(item: dict, baseline_type: str, model: str, judge_model: str, max_tokens: int) -> dict:
    query, interpretations = parse_ambigqa_item(item)
    if not query or len(interpretations) < 2:
        return {"query": query, "status": "skipped"}

    try:
        # 1. Generate question
        if baseline_type == "vanilla":
            question = run_vanilla(query, model, max_tokens)
        else:
            question = run_parallel_only(query, model, max_tokens)

        # 2. Judge
        judge_llm = LLMClient(model=judge_model)
        eval_result = llm_judge_score(query, interpretations, question, judge_llm)

        return {
            "query": query,
            "baseline": baseline_type,
            "generated_question": question,
            "eval": eval_result,
        }
    except Exception as e:
        logger.exception(f"Failed to process baseline {baseline_type} for query: {query}")
        return {"query": query, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baselines on AmbigQA")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output JSONL files")
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Model name")
    parser.add_argument("--judge-model", default="qwen2.5:0.5b", help="Judge model name")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    items = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    for b_type in ["vanilla", "parallel"]:
        print(f"\n--- Running Baseline: {b_type.upper()} ---")
        output_file = f"{args.output_prefix}_{b_type}.jsonl"
        results = []

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_baseline, item, b_type, args.model, args.judge_model, 300)
                for item in items
            ]
            with open(output_file, "w") as out:
                for future in tqdm(as_completed(futures), total=len(futures), desc=b_type):
                    res = future.result()
                    results.append(res)
                    out.write(json.dumps(res) + "\n")
                    out.flush()

        # Aggregate
        valid = [r for r in results if "eval" in r]
        if valid:
            avg_score = sum(r["eval"]["score"] for r in valid) / len(valid)
            coverage = sum(1 for r in valid if r["eval"]["covers_interpretations"]) / len(valid) * 100
            print(f"[{b_type.upper()}] Avg Score: {avg_score:.2f} | Coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
