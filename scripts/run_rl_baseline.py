"""RL-Inspired Baseline: Modeling Future Turns (Search at Inference Time).

Workflow for each query:
1. Generate N candidate clarifying questions.
2. For each candidate:
   a. Pick a 'gold' interpretation from AmbigQA.
   b. Simulate a user response based on that intent.
   c. Evaluate how well the (Question + Answer) resolves the ambiguity.
3. Select the candidate with the highest resolution score.
4. Finally, evaluate the selected question using the standard LLM Judge.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from d2c.llm import LLMClient
from d2c.prompts import (
    VANILLA_CQG_SYSTEM, VANILLA_CQG_USER,
    SIMULATED_USER_SYSTEM, SIMULATED_USER_USER,
    RESOLUTION_JUDGE_SYSTEM, RESOLUTION_JUDGE_USER
)
from eval.metrics import llm_judge_score

logger = logging.getLogger(__name__)

def parse_ambigqa_item(item: dict) -> tuple[str, list[str]]:
    query = item.get("question")
    annotations = item.get("annotations", [])
    interpretations = []
    for ann in annotations:
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    return query, interpretations

def generate_candidates(query: str, n: int, model: str, llm: LLMClient) -> list[str]:
    """Generate N different clarifying questions."""
    candidates = []
    for i in range(n):
        # We vary temperature to get diversity
        q = llm.chat(
            system_prompt=VANILLA_CQG_SYSTEM,
            user_prompt=VANILLA_CQG_USER.format(query=query),
            temperature=0.7 + (i * 0.1),
            max_tokens=100
        )
        candidates.append(q)
    return list(set(candidates)) # Deduplicate

def simulate_and_score(query: str, question: str, target_intent: str, all_intents: list[str], sim_llm: LLMClient) -> float:
    """Simulate a turn and return the resolution score."""
    # 1. Simulate Answer
    answer = sim_llm.chat(
        system_prompt=SIMULATED_USER_SYSTEM,
        user_prompt=SIMULATED_USER_USER.format(
            query=query, 
            interpretation=target_intent,
            clarifying_question=question
        ),
        temperature=0.1
    )

    # 2. Judge Resolution
    formatted_intents = "\n".join(f"- {i}" for i in all_intents)
    raw_eval = sim_llm.chat(
        system_prompt=RESOLUTION_JUDGE_SYSTEM,
        user_prompt=RESOLUTION_JUDGE_USER.format(
            query=query,
            clarifying_question=question,
            user_answer=answer,
            all_interpretations=formatted_intents
        ),
        temperature=0.0
    )

    try:
        # Simple JSON extraction
        if "```json" in raw_eval:
            raw_eval = raw_eval.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_eval:
            raw_eval = raw_eval.split("```")[1].split("```")[0].strip()
        
        eval_data = json.loads(raw_eval)
        return float(eval_data.get("resolution_score", 1.0))
    except:
        return 1.0

def process_query_rl(item: dict, n_candidates: int, gen_model: str, sim_model: str, judge_model: str) -> dict:
    query, interpretations = parse_ambigqa_item(item)
    if not query or len(interpretations) < 2:
        return {"query": query, "status": "skipped"}

    gen_llm = LLMClient(model=gen_model)
    sim_llm = LLMClient(model=sim_model)
    judge_llm = LLMClient(model=judge_model)

    try:
        # 1. Generate Candidates
        candidates = generate_candidates(query, n_candidates, gen_model, gen_llm)
        
        # 2. Simulation Loop (Modeling Future Turns)
        # We test against the first gold interpretation as the 'hidden intent'
        target_intent = interpretations[0]
        
        best_q = candidates[0]
        best_score = -1.0
        
        for q in candidates:
            score = simulate_and_score(query, q, target_intent, interpretations, sim_llm)
            if score > best_score:
                best_score = score
                best_q = q
        
        # 3. Final standard evaluation (for comparison)
        # This is the "Score (1-5)" we use in other baselines
        eval_result = llm_judge_score(query, interpretations, best_q, judge_llm)

        return {
            "query": query,
            "baseline": "rl_future_search",
            "generated_question": best_q,
            "candidates_count": len(candidates),
            "best_resolution_score": best_score,
            "eval": eval_result
        }
    except Exception as e:
        logger.exception(f"RL baseline failed for query: {query}")
        return {"query": query, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Run RL-Inspired 'Future Turn' Baseline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gen-model", default="qwen2.5:0.5b")
    parser.add_argument("--sim-model", default="qwen3.5:4b")
    parser.add_argument("--judge-model", default="qwen2.5:0.5b")
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2) # Lower because each item does many calls
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    items = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_query_rl, item, args.n_candidates, args.gen_model, args.sim_model, args.judge_model)
            for item in items
        ]
        
        with open(args.output, "w") as out:
            for future in tqdm(as_completed(futures), total=len(futures), desc="RL Baseline"):
                res = future.result()
                results.append(res)
                out.write(json.dumps(res) + "\n")
                out.flush()

    valid = [r for r in results if "eval" in r]
    if valid:
        avg_score = sum(r["eval"]["score"] for r in valid) / len(valid)
        coverage = sum(1 for r in valid if r["eval"]["covers_interpretations"]) / len(valid) * 100
        print(f"\n[RL-Future-Search] Avg Score: {avg_score:.2f} | Coverage: {coverage:.1f}%")

if __name__ == "__main__":
    main()
