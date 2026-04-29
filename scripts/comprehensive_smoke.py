"""Comprehensive D2C Smoke Test with Homogeneity Analysis.
Metrics: Detection F1, Quality, BERT-equivalent Sim, Coverage, and Average Rounds.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from d2c.baseline import run_vanilla_cqg
from eval.metrics import (
    semantic_similarity, 
    semantic_similarity_batch, 
    ndcg_score, 
    llm_judge_quality, 
    clarification_need_f1
)

logger = logging.getLogger(__name__)

DATASETS = ["clamber", "clariq", "qulac"]

def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=7)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    args = parser.parse_args()

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    print(f"\n{'='*120}")
    print(f"  COMPREHENSIVE SMOKE TEST & HOMOGENEITY ANALYSIS")
    print(f"  Models: {args.model} | Judge: {args.judge_model}")
    print(f"{'='*120}")

    methods = ["vanilla", "speech_act"]
    all_results = {m: {
        "scores": [], "sims": [], "preds": [], "golds": [], "rounds": [], "covers": []
    } for m in methods}
    
    total_samples = 0
    homogeneous_count = 0

    for dataset in DATASETS:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
            
        print(f"\n>>> Processing {dataset.upper()} ({args.n_per_dataset} items)...")
        
        # Take a slice
        items = all_items[:args.n_per_dataset]
        
        for idx, item in enumerate(items):
            total_samples += 1
            query = item["query"]
            print(f"  [{idx+1}/{len(items)}] Query: {query[:50]}...")
            is_ambig = item["is_ambiguous"]
            golds = item.get("gold_clarifying_questions", [])
            gold_q = golds[0] if golds else "N/A"
            
            sample_results = {}
            for name in methods:
                print(f"    Running {name}...")
                if name == "vanilla":
                    res = run_vanilla_cqg(query, llm, max_tokens=300)
                    q_text = res.clarifying_question
                    pred_ambig = "CLEAR" not in q_text.upper()
                    rnd = 0
                else:
                    res = run_d2c(query, variant=name, model=args.model)
                    q_text = res.synthesizer_result.clarifying_question
                    pred_ambig = "CLEAR" not in q_text.upper()
                    rnd = res.dialogue.rounds_completed

                print(f"    Judging {name}...")
                j_res = llm_judge_quality(query, q_text, gold_q, judge_llm)
                
                sample_results[name] = {
                    "score": j_res["score"],
                    "pred": pred_ambig,
                    "text": q_text,
                    "covers": j_res.get("covers", False),
                    "sim": semantic_similarity(q_text, gold_q),
                    "rnd": rnd
                }
                
                all_results[name]["scores"].append(j_res["score"])
                all_results[name]["sims"].append(sample_results[name]["sim"])
                all_results[name]["preds"].append(pred_ambig)
                all_results[name]["golds"].append(is_ambig)
                all_results[name]["rounds"].append(rnd)
                all_results[name]["covers"].append(1 if j_res.get("covers") else 0)

            # INTERMEDIATE TABLE
            print(f"\n--- INTERMEDIATE SUMMARY (N={total_samples}) ---")
            for name in methods:
                q_avg = statistics.mean(all_results[name]["scores"])
                r_avg = statistics.mean(all_results[name]["rounds"])
                print(f"{name:<12} | Score: {q_avg:>4.1f} | Rnd: {r_avg:>4.1f}")
            print("-------------------------------------------\n")

            scores = [v["score"] for v in sample_results.values()]
            preds = [v["pred"] for v in sample_results.values()]
            if len(set(scores)) == 1 and len(set(preds)) == 1:
                homogeneous_count += 1

    # Final Master Table
    print(f"\n{'='*120}")
    print(f"  FINAL MASTER TABLE (N={total_samples})")
    print(f"{'='*120}")
    header = f"{'Method':<12} | {'F1':<5} | {'Prec':<5} | {'Rec':<5} | {'Qual':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
    print(header)
    print("-" * len(header))
    
    for name in methods:
        det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
        q_mean = statistics.mean(all_results[name]["scores"])
        sim_mean = statistics.mean(all_results[name]["sims"])
        cov_pct = statistics.mean(all_results[name]["covers"]) * 100
        rnd_avg = statistics.mean(all_results[name]["rounds"])
        print(f"{name:<12} | {det['f1']:>5.2f} | {det['precision']:>5.2f} | {det['recall']:>5.2f} | {q_mean:>4.1f} | {sim_mean:>6.3f} | {cov_pct:>5.1f} | {rnd_avg:>4.1f}")

    print(f"\n{'='*120}")
    print(f"HOMOGENEITY ANALYSIS:")
    print(f"Homogeneous Samples: {homogeneous_count} / {total_samples} ({homogeneous_count/total_samples*100:.1f}%)")
    print(f"Varied Samples: {total_samples - homogeneous_count} / {total_samples} ({(total_samples - homogeneous_count)/total_samples*100:.1f}%)")
    
    if homogeneous_count / total_samples > 0.5:
        print("\nWARNING: High homogeneity detected. Consider:")
        print("1. Updating the Judge Prompt to be more 'strict' (e.g., penalyzing generic questions).")
        print("2. Increasing temperature or top_p for variety.")
        print("3. Using a larger model for the judge (e.g. 7B+ instead of 4B).")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
