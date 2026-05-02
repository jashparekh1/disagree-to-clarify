"""Exhaustive D2C Benchmarking with Informative Sampling.
Target: 5 informative queries per dataset (where models disagree).
Metrics: Detection F1, Quality, Semantic Sim, nDCG, and Average Rounds.
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
    mrr_score, 
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
    parser.add_argument("--target-n", type=int, default=5, help="Find this many informative queries")
    parser.add_argument("--max-look", type=int, default=30, help="Max queries to check per dataset")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--judge-model", default="qwen2.5:1.5b")
    args = parser.parse_args()

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    print(f"\n{'='*110}")
    print(f"  D2C INFORMATIVE BENCHMARK (Target: {args.target_n} informative results per dataset)")
    print(f"  Models: {args.model} | Judge: {args.judge_model}")
    print(f"{'='*110}")

    for dataset in DATASETS:
        all_items = load_full_test_set(dataset)
        if not all_items:
            print(f"\nSkipping {dataset.upper()} (no test set found)")
            continue
            
        print(f"\n>>> Searching {dataset.upper()} for informative examples...")
        
        methods = ["vanilla", "speech_act", "d2c", "taxonomy"]
        results = {m: {
            "qs": [], "matches": [], "scores": [], "sim_lists": [], 
            "preds_ambig": [], "gold_ambig": [], "rounds": []
        } for m in methods}
        golds_batch = []
        
        found_count = 0
        look_count = 0
        
        for item in all_items:
            if found_count >= args.target_n or look_count >= args.max_look:
                break
            
            look_count += 1
            query = item["query"]
            is_ambig = item["is_ambiguous"]
            golds = item.get("gold_clarifying_questions", [])
            
            q_data = {}
            for name in methods:
                if name == "vanilla":
                    res = run_vanilla_cqg(query, llm, max_tokens=300)
                    q_text = res.clarifying_question
                    pred_ambig = True
                    rnd = 0
                else:
                    # Let run_d2c handle the default num_rounds for the variant
                    res = run_d2c(query, variant=name, model=args.model)
                    q_text = res.synthesizer_result.clarifying_question
                    # Decision: If any agent held or updated, it's ambiguous
                    pred_ambig = any(resp.stance.value != "CONCEDE" for rnd_data in res.dialogue.rounds for resp in rnd_data)
                    rnd = res.dialogue.rounds_completed

                j_res = llm_judge_quality(query, q_text, golds[0] if golds else "N/A", judge_llm)
                q_data[name] = {
                    "score": j_res["score"],
                    "pred_ambig": pred_ambig,
                    "text": q_text,
                    "sims": [semantic_similarity(q_text, g) for g in golds],
                    "rnd": rnd
                }

            # INFORMATIVE CHECK: Do the models actually differ?
            scores = [v["score"] for v in q_data.values()]
            preds = [v["pred_ambig"] for v in q_data.values()]
            
            # Informative if scores or predictions vary
            is_informative = (len(set(scores)) > 1) or (len(set(preds)) > 1)
            
            if is_informative:
                found_count += 1
                golds_batch.append(golds)
                print(f"  [{found_count}/{args.target_n}] Found informative query: {query[:60]}...")
                for name in methods:
                    results[name]["qs"].append(q_data[name]["text"])
                    results[name]["matches"].append(1 if q_data[name]["score"] >= 3 else 0)
                    results[name]["scores"].append(q_data[name]["score"])
                    results[name]["sim_lists"].append(q_data[name]["sims"])
                    results[name]["preds_ambig"].append(q_data[name]["pred_ambig"])
                    results[name]["gold_ambig"].append(is_ambig)
                    results[name]["rounds"].append(q_data[name]["rnd"])

        if found_count == 0:
            print(f"  No informative queries found in first {look_count} items. Skipping summary.")
            continue

        # Dataset Summary Table
        print(f"\nSUMMARY FOR {dataset.upper()} ({found_count} informative queries):")
        header = f"{'Method':<12} | {'F1':<5} | {'Prec':<5} | {'Rec':<5} | {'Qual':<4} | {'Sim':<6} | {'nDCG':<5} | {'Rnd':<4}"
        print(header)
        print("-" * len(header))
        
        for name in methods:
            det = clarification_need_f1(results[name]["preds_ambig"], results[name]["gold_ambig"])
            q_mean = statistics.mean(results[name]["scores"])
            sim_mean = semantic_similarity_batch(results[name]["qs"], golds_batch)["mean"]
            ndcg_val = statistics.mean([ndcg_score(s, k=5) for s in results[name]["sim_lists"]])
            rnd_avg = statistics.mean(results[name]["rounds"])
            print(f"{name:<12} | {det['f1']:>5.2f} | {det['precision']:>5.2f} | {det['recall']:>5.2f} | {q_mean:>4.1f} | {sim_mean:>6.3f} | {ndcg_val:>5.3f} | {rnd_avg:>4.1f}")

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    main()
