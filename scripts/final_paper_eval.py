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
    clarification_need_f1,
    internal_divergence_score
)

logger = logging.getLogger(__name__)

DATASETS = ["clamber", "clariq", "qulac", "abgcoqa"]

def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def run_parallel_baseline_full(query: str, model: str, llm: LLMClient, context: str | None = None) -> tuple[str, list[str]]:
    """Agents generate 1 round in parallel, then synthesize (No Dialogue)."""
    roles = [AgentRole.LOCUTIONARY, AgentRole.ILLOCUTIONARY, AgentRole.PERLOCUTIONARY]
    agents = [Agent(role, llm, max_tokens=300) for role in roles]
    
    # 1 Round only
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.respond_initial, query, context) for agent in agents]
        round_0 = [f.result() for f in futures]

    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
    synth = synthesize(query, dialogue, llm, max_tokens=300, variant="speech_act")
    return synth.clarifying_question, [r.interpretation for r in round_0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="Specific dataset to run (clamber, clariq, qulac, abgcoqa)")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    args = parser.parse_args()

    # Set seed for reproducibility
    import random
    import time
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    random.seed(args.seed)

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    results_file = Path("paper_evaluation_results.txt")
    inspection_file = Path("variant_inspections.txt")
    
    with open(results_file, "w") as f:
        f.write(f"PAPER-READY MASTER EVALUATION\n")
        f.write(f"Seed: {args.seed} | Generator Model: {args.model} | Judge Model: {args.judge_model}\n")
        f.write("="*120 + "\n\n")
        
    with open(inspection_file, "w") as f:
        f.write("DETAILED VARIANT INSPECTION LOG\n")
        f.write(f"Seed: {args.seed} | Generator: {args.model} | Judge: {args.judge_model}\n")
        f.write("="*120 + "\n\n")

    methods = ["vanilla", "parallel", "madisse", "speech_act"]
    all_results = {m: {
        "scores": [], "sims": [], "preds": [], "golds": [], "rounds": [], "covers": [], "divs": []
    } for m in methods}
    
    total_samples = 0

    print(f"\n{'='*120}")
    print(f"  PAPER-READY MASTER EVALUATION (Gen: {args.model} | Judge: {args.judge_model})")
    print(f"{'='*120}")

    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    for dataset in datasets_to_run:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
        
        # Shuffle for random sampling
        import random
        random.shuffle(all_items)
            
        items = all_items[:args.n_per_dataset]
        msg = f"\n>>> Processing {dataset.upper()} ({len(items)} items)..."
        print(msg)
        with open(results_file, "a") as f: f.write(msg + "\n")
        
        for idx, item in enumerate(items):
            total_samples += 1
            query = item["query"]
            is_ambig = item["is_ambiguous"]
            context = item.get("context")
            gold_qs = item.get("gold_clarifying_questions", [])
            gold_q = gold_qs[0] if gold_qs else "N/A"
            
            msg = f"  [{total_samples}] Query: {query[:60]}..."
            print(msg)
            with open(results_file, "a") as f: f.write(msg + "\n")
            
            with open(inspection_file, "a") as f:
                f.write(f"\n{'#'*80}\n")
                f.write(f"SAMPLE {total_samples} | Dataset: {dataset} | Ambiguous: {is_ambig}\n")
                f.write(f"QUERY: {query}\n")
                if context:
                    f.write(f"CONTEXT: {context[:500]}...\n")
                f.write(f"GOLD CQ: {gold_q}\n")
                f.write(f"{'#'*80}\n")
            
            for name in methods:
                try:
                    # 1. GENERATE
                    full_dialogue_log = ""
                    if name == "vanilla":
                        res = run_vanilla_cqg(query, llm, max_tokens=300, context=context)
                        q_text = res.clarifying_question
                        rnd = 0
                        div_score = 0.0
                        full_dialogue_log = f"Vanilla direct output: {q_text}"
                    elif name == "parallel":
                        q_text, interps = run_parallel_baseline_full(query, args.model, llm, context=context)
                        rnd = 1
                        div_score = internal_divergence_score(interps)
                        for r_interp in interps:
                            full_dialogue_log += f"[AGENT]: {r_interp}\n"
                    else: # madisse or speech_act
                        res = run_d2c(query, variant=name, model=args.model, num_rounds=args.num_rounds, context=context)
                        q_text = res.synthesizer_result.clarifying_question
                        rnd = res.dialogue.rounds_completed
                        final_interps = [r.interpretation for r in res.dialogue.rounds[-1]]
                        div_score = internal_divergence_score(final_interps)
                        
                        for r_idx, rnd_data in enumerate(res.dialogue.rounds):
                            full_dialogue_log += f"--- Round {r_idx} ---\n"
                            for r in rnd_data:
                                stance_info = f" ({r.stance.value})" if r_idx > 0 else ""
                                full_dialogue_log += f"[{r.role.value.upper()}]{stance_info}: {r.interpretation}\n"

                    # 2. DETECT (Detection-Aware logic)
                    pred_ambig = "CLEAR" not in q_text.upper()

                    # 3. JUDGE
                    j_res = llm_judge_quality(query, q_text, gold_q, judge_llm)
                    
                    # 4. STORE
                    # Always store detection, rounds, and divergence
                    all_results[name]["preds"].append(pred_ambig)
                    all_results[name]["golds"].append(is_ambig)
                    all_results[name]["rounds"].append(rnd)
                    all_results[name]["divs"].append(div_score)

                    # ONLY store generation quality metrics for truly ambiguous samples
                    if is_ambig:
                        all_results[name]["scores"].append(j_res["score"])
                        all_results[name]["sims"].append(semantic_similarity(q_text, gold_q))
                        all_results[name]["covers"].append(1 if j_res.get("covers") else 0)
                    
                    msg = f"    - {name:<10}: Score={j_res['score'] if is_ambig else 'N/A'}, Div={div_score:.2f}, Round={rnd}"
                    print(msg)
                    with open(results_file, "a") as f: f.write(msg + "\n")
                    
                    with open(inspection_file, "a") as f:
                        f.write(f"\nMETHOD: {name}\n")
                        f.write(f"GENERATED CQ: {q_text}\n")
                        f.write(f"\n--- METRICS SUMMARY ---\n")
                        f.write(f"Quality Score: {j_res['score'] if is_ambig else 'N/A'}/5\n")
                        f.write(f"Internal Divergence: {div_score:.4f}\n")
                        sim_val = all_results[name]["sims"][-1] if is_ambig and all_results[name]["sims"] else 0.0
                        f.write(f"Semantic Similarity: {sim_val:.4f}\n")
                        f.write(f"Covers Interpretations: {j_res.get('covers', False) if is_ambig else 'N/A'}\n")
                        f.write(f"Detection: {'AMBIGUOUS' if pred_ambig else 'CLEAR'} (Gold: {'AMBIGUOUS' if is_ambig else 'CLEAR'})\n")
                        f.write(f"Rounds: {rnd}\n")
                        f.write(f"-----------------------\n")
                        f.write(f"\nINTERNAL DIALOGUE:\n{full_dialogue_log}\n")
                        f.write(f"\nJUDGE REASONING:\n{j_res['raw']}\n")
                        f.write(f"{'#'*80}\n")

                except Exception as e:
                    msg = f"    - {name:<10}: ERROR {str(e)}"
                    print(msg)
                    with open(results_file, "a") as f: f.write(msg + "\n")

            # --- REAL-TIME TABLE UPDATE ---
            table_lines = []
            table_lines.append(f"\n--- RUNNING RESULTS (N={total_samples}) ---")
            header = f"{'Method':<12} | {'F1':<5} | {'Qual':<4} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
            table_lines.append(header)
            table_lines.append("-" * len(header))
            for name in methods:
                if not all_results[name]["preds"]: continue # Check preds instead of scores
                det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
                
                # Metrics that might be empty if all samples so far are clear
                q_list = all_results[name]["scores"]
                sim_list = all_results[name]["sims"]
                cov_list = all_results[name]["covers"]
                
                q_mean = statistics.mean(q_list) if q_list else 0.0
                sim_mean = statistics.mean(sim_list) if sim_list else 0.0
                cov_pct = (statistics.mean(cov_list) * 100) if cov_list else 0.0
                
                rnd_avg = statistics.mean(all_results[name]["rounds"])
                div_avg = statistics.mean(all_results[name]["divs"])
                table_lines.append(f"{name:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_pct:>5.1f} | {rnd_avg:>4.1f}")
            table_lines.append("-------------------------------------------\n")
            
            full_table = "\n".join(table_lines)
            print(full_table)
            with open(results_file, "a") as f: f.write(full_table)

    # FINAL MASTER TABLE
    final_output = []
    final_output.append(f"\n{'='*130}")
    final_output.append(f"  FINAL MASTER RESULTS TABLE (N={total_samples})")
    final_output.append(f"{'='*130}")
    header = f"{'Method':<12} | {'F1':<5} | {'Prec':<5} | {'Rec':<5} | {'Qual':<4} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
    final_output.append(header)
    final_output.append("-" * len(header))
    
    for name in methods:
        preds = all_results[name]["preds"]
        if not preds: continue
        
        det = clarification_need_f1(preds, all_results[name]["golds"])
        
        q_list = all_results[name]["scores"]
        sim_list = all_results[name]["sims"]
        cov_list = all_results[name]["covers"]
        
        q_mean = statistics.mean(q_list) if q_list else 0.0
        sim_mean = statistics.mean(sim_list) if sim_list else 0.0
        cov_pct = (statistics.mean(cov_list) * 100) if cov_list else 0.0
        
        rnd_avg = statistics.mean(all_results[name]["rounds"])
        div_avg = statistics.mean(all_results[name]["divs"])
        
        final_output.append(f"{name:<12} | {det['f1']:>5.2f} | {det['precision']:>5.2f} | {det['recall']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_pct:>5.1f} | {rnd_avg:>4.1f}")
    final_output.append(f"{'='*130}\n")
    
    final_full = "\n".join(final_output)
    print(final_full)
    with open(results_file, "a") as f: f.write(final_full)
    print(f"Full log saved to: {results_file}")

if __name__ == "__main__":
    main()
