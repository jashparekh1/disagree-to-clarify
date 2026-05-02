"""The Ultimate D2C Evaluation Script for the Paper.
Generates a comprehensive Master Table comparing 5 distinct strategies across all metrics.
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

# Cache for models to avoid reloading every sample
_MLX_SFT_MODEL = None
_MLX_SFT_TOKENIZER = None
_MLX_RL_MODEL = None
_MLX_RL_TOKENIZER = None

def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def run_mlx_model(query: str, adapter_path: str, model_type: str = "sft") -> str:
    """Runs a fine-tuned model using MLX (SFT or RL/DPO)."""
    global _MLX_SFT_MODEL, _MLX_SFT_TOKENIZER, _MLX_RL_MODEL, _MLX_RL_TOKENIZER
    
    try:
        from mlx_lm import load, generate
    except ImportError:
        return "ERROR: mlx-lm not installed"

    # Select the right cache slots
    if model_type == "rl":
        model_ref, tokenizer_ref = _MLX_RL_MODEL, _MLX_RL_TOKENIZER
    else:
        model_ref, tokenizer_ref = _MLX_SFT_MODEL, _MLX_SFT_TOKENIZER

    if model_ref is None:
        base_model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        if not Path(adapter_path).exists():
            return f"ERROR: Adapter path {adapter_path} not found"
        model_ref, tokenizer_ref = load(base_model, adapter_path=adapter_path)
        
        # Update globals
        if model_type == "rl":
            _MLX_RL_MODEL, _MLX_RL_TOKENIZER = model_ref, tokenizer_ref
        else:
            _MLX_SFT_MODEL, _MLX_SFT_TOKENIZER = model_ref, tokenizer_ref

    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    output = generate(model_ref, tokenizer_ref, prompt=prompt, max_tokens=150, verbose=False)
    return output.split("<|im_end|>")[0].strip()

def run_parallel_baseline_full(query: str, model: str, llm: LLMClient, context: str | None = None) -> tuple[str, list[str]]:
    """Agents generate 1 round in parallel, then synthesize (No Dialogue).
    Uses the same agents as D2C for a fair comparison.
    """
    roles = [AgentRole.FACT_FINDER, AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER]
    agents = [Agent(role, llm, max_tokens=300) for role in roles]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.respond_initial, query, context) for agent in agents]
        round_0 = [f.result() for f in futures]

    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
    synth = synthesize(query, dialogue, llm, max_tokens=300, variant="d2c")
    return synth.clarifying_question, [r.interpretation for r in round_0]

from d2c.prompts import (
    VANILLA_CQG_SYSTEM, VANILLA_CQG_USER,
    SIMULATED_USER_SYSTEM, SIMULATED_USER_USER,
    RESOLUTION_JUDGE_SYSTEM, RESOLUTION_JUDGE_USER
)

def run_rl_inference(query: str, interpretations: list[str], llm: LLMClient, sim_llm: LLMClient, n_candidates: int = 3) -> str:
    """RL-Inspired 'Search at Inference' Baseline."""
    if not interpretations:
        return llm.chat(VANILLA_CQG_SYSTEM, VANILLA_CQG_USER.format(query=query))

    # 1. Generate N candidates
    candidates = []
    for i in range(n_candidates):
        raw = llm.chat(
            system_prompt=VANILLA_CQG_SYSTEM,
            user_prompt=VANILLA_CQG_USER.format(query=query),
            temperature=0.7 + (i * 0.1),
            max_tokens=150
        )
        # Parse JSON if needed
        if "{" in raw and "}" in raw:
            try:
                import json
                start, end = raw.find("{"), raw.rfind("}")
                data = json.loads(raw[start:end+1])
                q = data.get("clarifying_question", "CLEAR")
                if q != "CLEAR": candidates.append(q)
            except: pass
        elif "CLEAR" not in raw.upper():
            candidates.append(raw.strip())
    
    if not candidates: return "CLEAR"
    candidates = list(set(candidates))

    # 2. Simulate and Score
    target_intent = interpretations[0]
    best_q = candidates[0]
    best_score = -1.0

    for q in candidates:
        # Simulate Answer
        answer = sim_llm.chat(
            system_prompt=SIMULATED_USER_SYSTEM,
            user_prompt=SIMULATED_USER_USER.format(
                query=query, interpretation=target_intent, clarifying_question=q
            ),
            temperature=0.1
        )
        # Judge Resolution
        formatted_intents = "\n".join(f"- {i}" for i in interpretations)
        raw_eval = sim_llm.chat(
            system_prompt=RESOLUTION_JUDGE_SYSTEM,
            user_prompt=RESOLUTION_JUDGE_USER.format(
                query=query, clarifying_question=q, user_answer=answer, all_interpretations=formatted_intents
            ),
            temperature=0.0
        )
        try:
            import json
            start, end = raw_eval.find("{"), raw_eval.rfind("}")
            eval_data = json.loads(raw_eval[start:end+1])
            score = float(eval_data.get("resolution_score", 1.0))
            if score > best_score:
                best_score = score
                best_q = q
        except: continue
        
    return best_q

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="Specific dataset to run (clamber, clariq, qulac, abgcoqa)")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--sim-model", default="qwen3:4b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    parser.add_argument("--adapter-path", default="adapters", help="Path to MLX SFT adapters")
    parser.add_argument("--rl-adapter-path", default="adapters_rl", help="Path to MLX RL/DPO adapters")
    parser.add_argument("--methods", nargs="+", help="Specific methods to run (vanilla, sft, rl, rl_search, etc.)")
    args = parser.parse_args()

    import random
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    random.seed(args.seed)

    llm = LLMClient(model=args.model, think=False)
    sim_llm = LLMClient(model=args.sim_model, think=False)
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

    available_methods = ["vanilla", "sft", "rl", "parallel", "d2c"]
    methods = args.methods if args.methods else available_methods
    all_results = {m: {
        "scores": [], "sims": [], "preds": [], "golds": [], "rounds": [], "covers": [], "divs": []
    } for m in methods}
    
    total_samples = 0
    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    for dataset in datasets_to_run:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
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
            # Extract interpretations for RL (DPO format often needs them)
            interpretations = item.get("interpretations", [])
            if not interpretations and "annotations" in item:
                # AmbigQA style
                for ann in item.get("annotations", []):
                    if ann.get("type") == "multipleQAs":
                        for qa in ann.get("qaPairs", []):
                            if "question" in qa: interpretations.append(qa["question"])
            
            msg = f"  [{total_samples}] Query: {query[:60]}..."
            print(msg)
            with open(results_file, "a") as f: f.write(msg + "\n")
            
            for name in methods:
                try:
                    full_dialogue_log = ""
                    if name == "vanilla":
                        res = run_vanilla_cqg(query, llm, max_tokens=300, context=context)
                        q_text = res.clarifying_question
                        rnd, div_score = 0, 0.0
                    elif name == "sft":
                        q_text = run_mlx_model(query, adapter_path=args.adapter_path, model_type="sft")
                        rnd, div_score = 0, 0.0
                    elif name == "rl":
                        # This is the DPO-tuned model
                        q_text = run_mlx_model(query, adapter_path=args.rl_adapter_path, model_type="rl")
                        rnd, div_score = 0, 0.0
                    elif name == "parallel":
                        q_text, interps = run_parallel_baseline_full(query, args.model, llm, context=context)
                        rnd, div_score = 1, internal_divergence_score(interps)
                    elif name == "d2c":
                        res = run_d2c(query, variant="d2c", model=args.model, num_rounds=args.num_rounds, context=context)
                        q_text = res.synthesizer_result.clarifying_question
                        rnd = res.dialogue.rounds_completed
                        final_interps = [r.interpretation for r in res.dialogue.rounds[-1]]
                        div_score = internal_divergence_score(final_interps)
                    else:
                        continue # Should not happen with current list

                    pred_ambig = "CLEAR" not in q_text.upper()
                    j_res = llm_judge_quality(query, q_text, gold_q, judge_llm)
                    sim_score = semantic_similarity(q_text, gold_q) if is_ambig else 0.0
                    
                    all_results[name]["preds"].append(pred_ambig)
                    all_results[name]["golds"].append(is_ambig)
                    all_results[name]["rounds"].append(rnd)
                    all_results[name]["divs"].append(div_score)

                    if is_ambig:
                        all_results[name]["scores"].append(j_res["score"])
                        all_results[name]["sims"].append(sim_score)
                        all_results[name]["covers"].append(1 if j_res.get("covers") else 0)
                    
                    # Write to Inspection Log
                    with open(inspection_file, "a") as f_insp:
                        f_insp.write(f"Query: {query}\n")
                        f_insp.write(f"Method: {name}\n")
                        f_insp.write(f"Output: {q_text}\n")
                        f_insp.write(f"Gold:   {gold_q}\n")
                        f_insp.write(f"Judge Score: {j_res.get('score', 'N/A')}\n")
                        f_insp.write("-" * 40 + "\n")

                    msg = f"    - {name:<10}: Score={j_res['score'] if is_ambig else 'N/A'}, Sim={sim_score:.3f}, Round={rnd}"
                    print(msg)
                    with open(results_file, "a") as f: f.write(msg + "\n")

                except Exception as e:
                    print(f"    - {name:<10}: ERROR {str(e)}")

            # Real-time Table
            if total_samples % 1 == 0:
                table_lines = [f"\n--- RUNNING RESULTS (N={total_samples}) ---"]
                header = f"{'Method':<12} | {'F1':<5} | {'Qual':<4} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
                table_lines.append(header)
                table_lines.append("-" * len(header))
                for m in methods:
                    if not all_results[m]["preds"]: continue
                    det = clarification_need_f1(all_results[m]["preds"], all_results[m]["golds"])
                    q_mean = statistics.mean(all_results[m]["scores"]) if all_results[m]["scores"] else 0.0
                    sim_mean = statistics.mean(all_results[m]["sims"]) if all_results[m]["sims"] else 0.0
                    cov_mean = (sum(all_results[m]["covers"]) / len(all_results[m]["covers"]) * 100) if all_results[m]["covers"] else 0.0
                    div_avg = statistics.mean(all_results[m]["divs"])
                    table_lines.append(f"{m:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f} | {statistics.mean(all_results[m]['rounds']):>4.1f}")
                table_lines.append("-" * len(header))
                print("\n".join(table_lines))

    # Final Master Table
    master_header = f"\n{'='*130}\n  FINAL MASTER EVALUATION RESULTS (N={total_samples})\n{'='*130}"
    header_row = f"{'Method':<12} | {'F1':<5} | {'Qual':<4} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
    
    print(master_header)
    print(header_row)
    print("-" * 130)
    
    with open(results_file, "a") as f:
        f.write(master_header + "\n")
        f.write(header_row + "\n")
        f.write("-" * 130 + "\n")

    for name in methods:
        if not all_results[name]["preds"]:
            msg = f"{name:<12} | No data"
            print(msg)
            with open(results_file, "a") as f: f.write(msg + "\n")
            continue
            
        det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
        q_mean = statistics.mean(all_results[name]["scores"]) if all_results[name]["scores"] else 0.0
        sim_mean = statistics.mean(all_results[name]["sims"]) if all_results[name]["sims"] else 0.0
        cov_mean = (sum(all_results[name]["covers"]) / len(all_results[name]["covers"]) * 100) if all_results[name]["covers"] else 0.0
        div_avg = statistics.mean(all_results[name]["divs"])
        rnd_avg = statistics.mean(all_results[name]["rounds"])
        
        row = f"{name:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f} | {rnd_avg:>4.1f}"
        print(row)
        with open(results_file, "a") as f: f.write(row + "\n")
        
    print("="*130)
    with open(results_file, "a") as f: f.write("="*130 + "\n")

if __name__ == "__main__":
    main()
