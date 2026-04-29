"""The Ultimate D2C Evaluation Script for the Paper.
Generates a comprehensive Master Table comparing 4 distinct strategies across all metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from d2c.baseline import run_vanilla_cqg
from d2c.agents import Agent, AgentRole
from d2c.dialogue import DialogueResult
from d2c.synthesizer import synthesize
from eval.metrics import (
    semantic_similarity, 
    llm_judge_quality, 
    clarification_need_f1
)

logger = logging.getLogger(__name__)

DATASETS = ["clamber", "clariq", "qulac", "abgcoqa"]

def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def run_parallel_baseline(query: str, model: str, llm: LLMClient) -> str:
    """Agents generate 1 round in parallel, then synthesize (No Dialogue)."""
    roles = [AgentRole.LOCUTIONARY, AgentRole.ILLOCUTIONARY, AgentRole.PERLOCUTIONARY]
    agents = [Agent(role, llm, max_tokens=300) for role in roles]
    
    # 1 Round only
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.respond_initial, query) for agent in agents]
        round_0 = [f.result() for f in futures]

    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1)
    synth = synthesize(query, dialogue, llm, max_tokens=300, variant="speech_act")
    return synth.clarifying_question

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    args = parser.parse_args()

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    results_file = Path("paper_evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"PAPER-READY MASTER EVALUATION\n")
        f.write(f"Generator Model: {args.model} | Judge Model: {args.judge_model}\n")
        f.write("="*120 + "\n\n")

    methods = ["vanilla", "parallel", "madisse", "speech_act"]
    all_results = {m: {
        "scores": [], "sims": [], "preds": [], "golds": [], "rounds": [], "covers": []
    } for m in methods}
    
    total_samples = 0

    print(f"\n{'='*120}")
    print(f"  PAPER-READY MASTER EVALUATION (Gen: {args.model} | Judge: {args.judge_model})")
    print(f"{'='*120}")

    for dataset in DATASETS:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
            
        items = all_items[:args.n_per_dataset]
        msg = f"\n>>> Processing {dataset.upper()} ({len(items)} items)..."
        print(msg)
        with open(results_file, "a") as f: f.write(msg + "\n")
        
        for idx, item in enumerate(items):
            total_samples += 1
            query = item["query"]
            is_ambig = item["is_ambiguous"]
            gold_qs = item.get("gold_clarifying_questions", [])
            gold_q = gold_qs[0] if gold_qs else "N/A"
            
            msg = f"  [{total_samples}] Query: {query[:60]}..."
            print(msg)
            with open(results_file, "a") as f: f.write(msg + "\n")
            
            for name in methods:
                try:
                    # 1. GENERATE
                    if name == "vanilla":
                        res = run_vanilla_cqg(query, llm, max_tokens=300)
                        q_text = res.clarifying_question
                        rnd = 0
                    elif name == "parallel":
                        q_text = run_parallel_baseline(query, args.model, llm)
                        rnd = 1
                    else: # madisse or speech_act
                        res = run_d2c(query, variant=name, model=args.model, num_rounds=args.num_rounds)
                        q_text = res.synthesizer_result.clarifying_question
                        rnd = res.dialogue.rounds_completed

                    # 2. DETECT (Detection-Aware logic)
                    pred_ambig = "CLEAR" not in q_text.upper()

                    # 3. JUDGE
                    j_res = llm_judge_quality(query, q_text, gold_q, judge_llm)
                    
                    # 4. STORE
                    all_results[name]["scores"].append(j_res["score"])
                    all_results[name]["sims"].append(semantic_similarity(q_text, gold_q))
                    all_results[name]["preds"].append(pred_ambig)
                    all_results[name]["golds"].append(is_ambig)
                    all_results[name]["rounds"].append(rnd)
                    all_results[name]["covers"].append(1 if j_res.get("covers") else 0)
                    
                    msg = f"    - {name:<10}: Score={j_res['score']}, Round={rnd}"
                    print(msg)
                    with open(results_file, "a") as f: f.write(msg + "\n")
                except Exception as e:
                    msg = f"    - {name:<10}: ERROR {str(e)}"
                    print(msg)
                    with open(results_file, "a") as f: f.write(msg + "\n")

            # --- REAL-TIME TABLE UPDATE ---
            table_lines = []
            table_lines.append(f"\n--- RUNNING RESULTS (N={total_samples}) ---")
            header = f"{'Method':<12} | {'F1':<5} | {'Qual':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
            table_lines.append(header)
            table_lines.append("-" * len(header))
            for name in methods:
                if not all_results[name]["scores"]: continue
                det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
                q_mean = statistics.mean(all_results[name]["scores"])
                sim_mean = statistics.mean(all_results[name]["sims"])
                cov_pct = statistics.mean(all_results[name]["covers"]) * 100
                rnd_avg = statistics.mean(all_results[name]["rounds"])
                table_lines.append(f"{name:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {sim_mean:>6.3f} | {cov_pct:>5.1f} | {rnd_avg:>4.1f}")
            table_lines.append("-------------------------------------------\n")
            
            full_table = "\n".join(table_lines)
            print(full_table)
            with open(results_file, "a") as f: f.write(full_table)

    # FINAL MASTER TABLE
    final_output = []
    final_output.append(f"\n{'='*130}")
    final_output.append(f"  FINAL MASTER RESULTS TABLE (N={total_samples})")
    final_output.append(f"{'='*130}")
    header = f"{'Method':<12} | {'F1':<5} | {'Prec':<5} | {'Rec':<5} | {'Qual':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
    final_output.append(header)
    final_output.append("-" * len(header))
    
    for name in methods:
        valid_scores = all_results[name]["scores"]
        if not valid_scores: continue
        
        det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
        q_mean = statistics.mean(valid_scores)
        sim_mean = statistics.mean(all_results[name]["sims"])
        cov_pct = statistics.mean(all_results[name]["covers"]) * 100
        rnd_avg = statistics.mean(all_results[name]["rounds"])
        
        final_output.append(f"{name:<12} | {det['f1']:>5.2f} | {det['precision']:>5.2f} | {det['recall']:>5.2f} | {q_mean:>4.1f} | {sim_mean:>6.3f} | {cov_pct:>5.1f} | {rnd_avg:>4.1f}")
    final_output.append(f"{'='*130}\n")
    
    final_full = "\n".join(final_output)
    print(final_full)
    with open(results_file, "a") as f: f.write(final_full)
    print(f"Full log saved to: {results_file}")

if __name__ == "__main__":
    main()
