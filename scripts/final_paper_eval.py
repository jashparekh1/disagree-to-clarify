"""The Ultimate D2C Evaluation Script for the Paper.
Generates a comprehensive Master Table comparing 5 distinct strategies across all metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import random
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
    semantic_similarity_multi_ref,
    llm_judge_quality,
    llm_judge_quality_multi_ref,
    clarification_need_f1,
    internal_divergence_score
)
from d2c.prompts import (
    VANILLA_CQG_SYSTEM, VANILLA_CQG_USER
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
    """Agents generate 1 round in parallel, then synthesize (No Dialogue)."""
    roles = [AgentRole.FACT_FINDER, AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER]
    agents = [Agent(role, llm, max_tokens=300) for role in roles]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.respond_initial, query, context) for agent in agents]
        round_0 = [f.result() for f in futures]

    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
    synth = synthesize(query, dialogue, llm, max_tokens=300, variant="d2c")
    return synth.clarifying_question, [r.interpretation for r in round_0]

def run_self_consistency(query: str, llm: LLMClient, n: int = 3, context: str | None = None) -> str:
    """Self-consistency baseline: run vanilla CQG multiple times and pick most common/stable answer."""
    candidates = []
    for i in range(n):
        res = run_vanilla_cqg(query, llm, max_tokens=300, context=context, temperature=0.7 + (i*0.1))
        candidates.append(res.clarifying_question)
    
    if not candidates: return "CLEAR"
    # Centroid selection
    best_q, best_score = candidates[0], -1.0
    for q in candidates:
        avg_sim = sum(semantic_similarity(q, other) for other in candidates if other != q) / max(1, len(candidates)-1)
        if avg_sim > best_score:
            best_score = avg_sim
            best_q = q
    return best_q

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="Specific dataset to run")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--adapter-path", default="adapters", help="Path to MLX SFT adapters")
    parser.add_argument("--rl-adapter-path", default="adapters_rl", help="Path to MLX RL/DPO adapters")
    parser.add_argument("--methods", nargs="+", help="Specific methods to run")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    random.seed(args.seed)

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    results_file = Path("paper_evaluation_results.txt")
    inspection_file = Path("variant_inspections.txt")
    
    with open(results_file, "w") as f:
        f.write(f"PAPER-READY MASTER EVALUATION (IMPROVED)\n")
        f.write(f"Seed: {args.seed} | Generator: {args.model} | Judge: {args.judge_model}\n")
        f.write("="*120 + "\n\n")

    available_methods = ["vanilla", "self_consistency", "sft", "rl", "parallel", "d2c"]
    methods = args.methods if args.methods else available_methods
    all_results = {m: {
        "scores": [], "sims": [], "preds": [], "golds": [], "rounds": [], "covers": [], "divs": []
    } for m in methods}
    
    total_samples = 0
    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    for dataset in datasets_to_run:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
        
        # BALANCED SAMPLING (from Jash's branch)
        ambig = [x for x in all_items if x.get("is_ambiguous", True)]
        non_ambig = [x for x in all_items if not x.get("is_ambiguous", True)]
        random.shuffle(ambig)
        random.shuffle(non_ambig)
        
        n = args.n_per_dataset
        n_non = min(len(non_ambig), n // 2)
        n_amb = min(len(ambig), n - n_non)
        items = ambig[:n_amb] + non_ambig[:n_non]
        random.shuffle(items)
        
        msg = f"\n>>> Processing {dataset.upper()} ({len(items)} items, {n_amb} ambiguous)..."
        print(msg)
        with open(results_file, "a") as f: f.write(msg + "\n")
        
        for idx, item in enumerate(items):
            total_samples += 1
            query = item["query"]
            is_ambig = item.get("is_ambiguous", True)
            context = item.get("context")
            gold_qs = item.get("gold_clarifying_questions", [])
            gold_q = gold_qs[0] if gold_qs else "N/A"
            
            msg = f"  [{total_samples}] Query: {query[:60]}..."
            print(msg)
            
            for name in methods:
                try:
                    if name == "vanilla":
                        res = run_vanilla_cqg(query, llm, max_tokens=300, context=context)
                        q_text = res.clarifying_question
                        rnd, div_score = 0, 0.0
                    elif name == "self_consistency":
                        q_text = run_self_consistency(query, llm, context=context)
                        rnd, div_score = 0, 0.0
                    elif name == "sft":
                        q_text = run_mlx_model(query, adapter_path=args.adapter_path, model_type="sft")
                        rnd, div_score = 0, 0.0
                    elif name == "rl":
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

                    pred_ambig = "CLEAR" not in q_text.upper()
                    
                    # MULTI-REFERENCE EVALUATION
                    if is_ambig:
                        j_res = llm_judge_quality_multi_ref(query, q_text, gold_qs, judge_llm, context=context)
                        sim_score = semantic_similarity_multi_ref(q_text, gold_qs)
                    else:
                        j_res = {"score": 5 if not pred_ambig else 1, "covers": False}
                        sim_score = 0.0
                    
                    all_results[name]["preds"].append(pred_ambig)
                    all_results[name]["golds"].append(is_ambig)
                    all_results[name]["rounds"].append(rnd)
                    all_results[name]["divs"].append(div_score)

                    if is_ambig:
                        all_results[name]["scores"].append(j_res["score"])
                        all_results[name]["sims"].append(sim_score)
                        all_results[name]["covers"].append(1 if j_res.get("covers") else 0)
                    
                    with open(inspection_file, "a") as f_insp:
                        f_insp.write(f"Query: {query}\nMethod: {name}\nOutput: {q_text}\nScore: {j_res.get('score')}\nReason: {j_res.get('reasoning')}\n---\n")

                    msg = f"    - {name:<10}: Score={j_res['score'] if is_ambig else 'N/A'}, Sim={sim_score:.3f}, Round={rnd}"
                    print(msg)
                    with open(results_file, "a") as f: f.write(msg + "\n")

                except Exception as e:
                    print(f"    - {name:<10}: ERROR {str(e)}")

    # Final Master Table
    master_header = f"\n{'='*130}\n  FINAL MASTER EVALUATION RESULTS (N={total_samples})\n{'='*130}"
    header_row = f"{'Method':<12} | {'F1':<5} | {'Qual':<4} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
    print(master_header)
    print(header_row)
    for name in methods:
        det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
        q_mean = statistics.mean(all_results[name]["scores"]) if all_results[name]["scores"] else 0.0
        sim_mean = statistics.mean(all_results[name]["sims"]) if all_results[name]["sims"] else 0.0
        cov_mean = (sum(all_results[name]["covers"]) / len(all_results[name]["covers"]) * 100) if all_results[name]["covers"] else 0.0
        div_avg = statistics.mean(all_results[name]["divs"])
        print(f"{name:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f} | {statistics.mean(all_results[name]['rounds']):>4.1f}")

if __name__ == "__main__":
    main()
