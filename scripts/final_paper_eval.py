"""The Ultimate D2C Evaluation Script for the Paper.
Generates a comprehensive Master Table comparing 5 distinct strategies across all metrics.
"""

import argparse
import json
import logging
import statistics
import random
import torch
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

import threading

logger = logging.getLogger(__name__)

DATASETS = ["clamber", "clariq", "qulac", "abgcoqa"]

# CUDA Model Cache and Lock
_CUDA_MODELS = {} # path -> (model, tokenizer)
_MODEL_LOCK = threading.Lock()

def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def run_cuda_model(query: str, adapter_path: str) -> str:
    """Runs a fine-tuned model using Transformers (CUDA) with PEFT support."""
    global _CUDA_MODELS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    with _MODEL_LOCK:
        if adapter_path not in _CUDA_MODELS:
            try:
                # Check if directory is empty
                if not any(Path(adapter_path).iterdir()):
                    return "ERROR: Adapter directory is empty"

                print(f"Loading base model and adapter for {adapter_path}...")
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                # Load LoRA adapter
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.to(torch.bfloat16) # Force consistency to avoid mat1/mat2 dtype error
                _CUDA_MODELS[adapter_path] = (model, tokenizer)
            except Exception as e:
                return f"ERROR loading CUDA model: {e}"

    model, tokenizer = _CUDA_MODELS[adapter_path]
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generated_text

def run_parallel_baseline_full(query: str, model: str, llm: LLMClient, context: str | None = None) -> tuple[str, list[str]]:
    """Agents generate 1 round in parallel, then synthesize (No Dialogue)."""
    roles = [AgentRole.FACT_FINDER, AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER]
    agents = [Agent(role, llm, max_tokens=300) for role in roles]
    
    # Nested parallelism for agents
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.respond_initial, query, context) for agent in agents]
        round_0 = [f.result() for f in futures]

    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
    synth = synthesize(query, dialogue, llm, max_tokens=300, variant="d2c")
    return synth.clarifying_question, [r.interpretation for r in round_0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="Specific dataset to run")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--judge-model", default="llama3.1:8b")
    parser.add_argument("--adapter-path", default="adapters", help="Path to CUDA SFT adapters")
    parser.add_argument("--max-workers", type=int, default=16, help="Global parallel workers")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    random.seed(args.seed)

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    # Pre-load embedding model once to avoid thread deadlocks during lazy loading
    print("Pre-loading embedding model for metrics...")
    from eval.metrics import semantic_similarity
    semantic_similarity("warmup", "warmup") 

    results_file = Path("paper_evaluation_results.txt")
    inspection_file = Path("variant_inspections.txt")
    
    with open(results_file, "w") as f:
        f.write(f"PAPER-READY MASTER EVALUATION (CUDA OPTIMIZED)\n")
        f.write(f"Seed: {args.seed} | Generator: {args.model} | Judge: {args.judge_model}\n")
        f.write("="*120 + "\n\n")

    # Clear and initialize inspection log
    with open(inspection_file, "w") as f:
        f.write(f"VARIANT INSPECTION LOG\nSeed: {args.seed}\n" + "="*80 + "\n")

    methods = ["vanilla", "parallel", "sft", "d2c"]
    all_results = {m: {
        "scores": [], "sims": [], "preds": [], "golds": [], "rounds": [], "covers": [], "divs": []
    } for m in methods}
    
    datasets_to_run = [args.dataset] if args.dataset else DATASETS
    sample = []

    for dataset in datasets_to_run:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
        
        ambig = [x for x in all_items if x.get("is_ambiguous", True)]
        non_ambig = [x for x in all_items if not x.get("is_ambiguous", True)]
        random.shuffle(ambig)
        random.shuffle(non_ambig)
        
        n = args.n_per_dataset
        n_non = min(len(non_ambig), n // 2)
        n_amb = min(len(ambig), n - n_non)
        items = ambig[:n_amb] + non_ambig[:n_non]
        for it in items:
            it["_dataset"] = dataset
        sample.extend(items)

    total_tasks = len(sample) * len(methods)
    print(f"Starting Paper Evaluation on {len(sample)} items ({total_tasks} tasks) using {args.max_workers} workers...")

    pbar = tqdm(total=total_tasks, desc="Evaluating")

    def process_task(item, method_name):
        query = item["query"]
        is_ambig = item.get("is_ambiguous", True)
        context = item.get("context")
        gold_qs = item.get("gold_clarifying_questions", [])
        
        try:
            if method_name == "vanilla":
                res = run_vanilla_cqg(query, llm, max_tokens=300, context=context)
                q_text = res.clarifying_question
                rnd, div_score = 0, 0.0
            elif method_name == "sft":
                if not Path(args.adapter_path).exists():
                    q_text = "ADAPTER_MISSING"
                else:
                    q_text = run_cuda_model(query, args.adapter_path)
                rnd, div_score = 0, 0.0
            elif method_name == "rl":
                if not Path(args.rl_adapter_path).exists():
                    q_text = "RL_ADAPTER_MISSING"
                else:
                    q_text = run_cuda_model(query, args.rl_adapter_path)
                rnd, div_score = 0, 0.0
            elif method_name == "parallel":
                q_text, interps = run_parallel_baseline_full(query, args.model, llm, context=context)
                rnd, div_score = 1, internal_divergence_score(interps)
            elif method_name == "d2c":
                res = run_d2c(query, variant="d2c", model=args.model, num_rounds=args.num_rounds, context=context)
                q_text = res.synthesizer_result.clarifying_question
                rnd = res.dialogue.rounds_completed
                final_interps = [r.interpretation for r in res.dialogue.rounds[-1]]
                div_score = internal_divergence_score(final_interps)

            pred_ambig = "CLEAR" not in q_text.upper()
            
            if is_ambig:
                j_res = llm_judge_quality_multi_ref(query, q_text, gold_qs, judge_llm, context=context)
                sim_score = semantic_similarity_multi_ref(q_text, gold_qs)
            else:
                j_res = {"score": 5 if not pred_ambig else 1, "covers": False, "reasoning": "Correctly identified clear query"}
                sim_score = 0.0
            
            res_dict = {
                "method": method_name,
                "q_text": q_text,
                "score": j_res.get("score", 0),
                "sim": sim_score,
                "round": rnd,
                "div": div_score,
                "pred": pred_ambig,
                "covers": j_res.get("covers", False),
                "reasoning": j_res.get("reasoning", ""),
                "is_ambig": is_ambig
            }
            return query, res_dict
        except Exception as e:
            logger.error(f"Error in method {method_name} for query {query[:50]}: {e}")
            return query, None

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for it in sample:
            for m in methods:
                futures.append(executor.submit(process_task, it, m))
        
        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)
            if not result: continue
            
            query, m_res = result
            name = m_res["method"]
            
            all_results[name]["preds"].append(m_res["pred"])
            all_results[name]["golds"].append(m_res["is_ambig"])
            all_results[name]["rounds"].append(m_res["round"])
            all_results[name]["divs"].append(m_res["div"])

            if m_res["is_ambig"]:
                all_results[name]["scores"].append(m_res["score"])
                all_results[name]["sims"].append(m_res["sim"])
                all_results[name]["covers"].append(1 if m_res["covers"] else 0)
            
            with open(inspection_file, "a") as f_insp:
                f_insp.write(f"\nQUERY: {query}\nMethod: {name}\nScore: {m_res['score']}\nOutput: {m_res['q_text']}\n---\n")

    pbar.close()

    # Final Master Table
    print(f"\n{'='*130}\n  FINAL MASTER EVALUATION RESULTS (N={len(sample)})\n{'='*130}")
    header_row = f"{'Method':<12} | {'F1':<5} | {'Qual':<4} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5} | {'Rnd':<4}"
    print(header_row)
    
    with open(results_file, "a") as f:
        f.write(header_row + "\n" + "-"*130 + "\n")
        for name in methods:
            det = clarification_need_f1(all_results[name]["preds"], all_results[name]["golds"])
            q_mean = statistics.mean(all_results[name]["scores"]) if all_results[name]["scores"] else 0.0
            sim_mean = statistics.mean(all_results[name]["sims"]) if all_results[name]["sims"] else 0.0
            cov_mean = (sum(all_results[name]["covers"]) / len(all_results[name]["covers"]) * 100) if all_results[name]["covers"] else 0.0
            div_avg = statistics.mean(all_results[name]["divs"]) if all_results[name]["divs"] else 0.0
            rnd_avg = statistics.mean(all_results[name]["rounds"]) if all_results[name]["rounds"] else 0.0
            
            row = f"{name:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f} | {rnd_avg:>4.1f}"
            print(row)
            f.write(row + "\n")

if __name__ == "__main__":
    main()
] else 0.0
            
            row = f"{name:<12} | {det['f1']:>5.2f} | {q_mean:>4.1f} | {div_avg:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f} | {rnd_avg:>4.1f}"
            print(row)
            f.write(row + "\n")

if __name__ == "__main__":
    main()
