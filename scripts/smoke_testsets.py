"""Comprehensive D2C Benchmarking Script.
Compares:
1. Vanilla (Single-turn)
2. Single-Round (Multi-agent synthesis, no debate)
3. D2C Original (Debate, Heuristic roles)
4. D2C Speech Act (Debate, Linguistic roles)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

from d2c.agents import _ROLE_DISPLAY, Agent, AgentRole
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from d2c.synthesizer import synthesize
from d2c.dialogue import run_dialogue
from eval.judge import binary_judge, pairwise_judge
from eval.metrics import bert_score_compute, semantic_similarity_batch
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER

logger = logging.getLogger(__name__)

DATASETS = ["clariq", "qulac", "clamber"]

def run_vanilla(query: str, llm: LLMClient) -> str:
    user_prompt = VANILLA_CQG_USER.format(query=query)
    raw = llm.chat(
        system_prompt=VANILLA_CQG_SYSTEM,
        user_prompt=user_prompt,
        format_schema={"type": "object", "properties": {"clarifying_question": {"type": "string"}}, "required": ["clarifying_question"]}
    )
    try:
        return json.loads(raw)["clarifying_question"]
    except:
        return raw

def load_test_set(dataset: str, n: int) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
            if len(records) >= n: break
    return records

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--backend", default="ollama")
    args = parser.parse_args()

    llm = LLMClient(model=args.model, backend=args.backend, think=not args.no_think)
    judge_llm = LLMClient(model=args.judge_model, backend=args.backend, think=False)

    print(f"\n{'='*80}\n  D2C COMPREHENSIVE BENCHMARK (n={args.n})\n{'='*80}")
    
    overall_report = {}

    for dataset in DATASETS:
        items = load_test_set(dataset, args.n)
        print(f"\nProcessing {dataset.upper()}...")
        
        results = {
            "vanilla": {"qs": [], "matches": [], "rounds": []},
            "single_round": {"qs": [], "matches": [], "rounds": []},
            "original": {"qs": [], "matches": [], "rounds": []},
            "speech_act": {"qs": [], "matches": [], "rounds": []},
            "golds": []
        }

        for item in items:
            query = item["query"]
            golds = item.get("gold_clarifying_questions", [])
            results["golds"].append(golds)

            # 1. Vanilla
            q_v = run_vanilla(query, llm)
            results["vanilla"]["qs"].append(q_v)
            results["vanilla"]["matches"].append(binary_judge(query, q_v, golds, judge_llm).match)
            results["vanilla"]["rounds"].append(0)

            # 2. Single-Round (Speech Act roles)
            res_sr = run_d2c(query, model=args.model, num_rounds=1, variant="speech_act", backend=args.backend, think=not args.no_think)
            q_sr = res_sr.synthesizer_result.clarifying_question
            results["single_round"]["qs"].append(q_sr)
            results["single_round"]["matches"].append(binary_judge(query, q_sr, golds, judge_llm).match)
            results["single_round"]["rounds"].append(1)

            # 3. D2C Original
            res_orig = run_d2c(query, model=args.model, num_rounds=args.rounds, variant="original", backend=args.backend, think=not args.no_think)
            q_orig = res_orig.synthesizer_result.clarifying_question
            results["original"]["qs"].append(q_orig)
            results["original"]["matches"].append(binary_judge(query, q_orig, golds, judge_llm).match)
            results["original"]["rounds"].append(res_orig.dialogue.converged_at_round or args.rounds)

            # 4. D2C Speech Act
            res_sa = run_d2c(query, model=args.model, num_rounds=args.rounds, variant="speech_act", backend=args.backend, think=not args.no_think)
            q_sa = res_sa.synthesizer_result.clarifying_question
            results["speech_act"]["qs"].append(q_sa)
            results["speech_act"]["matches"].append(binary_judge(query, q_sa, golds, judge_llm).match)
            results["speech_act"]["rounds"].append(res_sa.dialogue.converged_at_round or args.rounds)

        # Dataset Summary Table
        print(f"\nResults for {dataset.upper()}:")
        print(f"{'Method':<15} | {'Match':<6} | {'Sim':<6} | {'B-S':<6} | {'Rnds':<4}")
        print("-" * 45)
        
        for name in ["vanilla", "single_round", "original", "speech_act"]:
            m = sum(results[name]["matches"]) / len(items)
            sim = semantic_similarity_batch(results[name]["qs"], results["golds"])["mean"]
            bs = bert_score_compute(results[name]["qs"], results["golds"])["f1"]
            rnd = statistics.mean(results[name]["rounds"])
            print(f"{name:<15} | {m*100:>5.1f}% | {sim:>6.3f} | {bs:>6.3f} | {rnd:>4.1f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
