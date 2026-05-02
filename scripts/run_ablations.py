"""Comprehensive Ablation Study for D2C.
Tests three key hypotheses:
1. Role Necessity: Does removing a specific agent (e.g. Fact-Finder) hurt F1 or Quality?
2. Taxonomy Value: Do specialized roles perform better than generic 'Assistant' roles?
3. Adversarial Pressure: Does 'forcing' a stance improve coverage over 'collaborative' consensus?
"""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from d2c.agents import Agent, AgentRole
from d2c.dialogue import DialogueResult
from d2c.synthesizer import synthesize
from eval.metrics import clarification_need_f1, llm_judge_quality, semantic_similarity

logger = logging.getLogger(__name__)

DATASETS = ["clamber", "clariq", "qulac", "abgcoqa"]

# --- Generic prompts for Ablation 2 ---
GENERIC_AGENT_SYSTEM = "You are an AI assistant. Analyze the user's query and identify any potential ambiguities. If it is clear, say CLEAR. If not, explain one specific interpretation. 1-2 sentences only."

# --- Non-forced prompts for Ablation 3 ---
COLLAB_FACT_SYSTEM = "You are a Fact-Finder. Your goal is to see if the query is clear. If you find real ambiguity, you may admit it, but your default lens is to look for a clear reading. 1-2 sentences only."
COLLAB_FACET_SYSTEM = "You are a Facet-Finder. Your goal is to see if subtopics are missing. If the query is already specific, you may admit it is clear. 1-2 sentences only."
COLLAB_INTENT_SYSTEM = "You are an Intent-Finder. Your goal is to see if the action is missing. If the intent is obvious, you may admit it is clear. 1-2 sentences only."

def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def run_ablation_variant(query, variant_name, llm, context=None):
    """
    Runs a modified version of the D2C pipeline based on the ablation type.
    """
    # Define Agents based on ablation
    if variant_name == "no_fact_finder":
        roles = [AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER]
    elif variant_name == "no_facet_finder":
        roles = [AgentRole.FACT_FINDER, AgentRole.INTENT_FINDER]
    elif variant_name == "no_intent_finder":
        roles = [AgentRole.FACT_FINDER, AgentRole.FACET_FINDER]
    elif variant_name == "generic_agents":
        # We use 3 agents but with generic prompts
        agents = [Agent(AgentRole.FACT_FINDER, llm) for _ in range(3)]
        for a in agents: a.system_prompt = GENERIC_AGENT_SYSTEM
        # Manually run one round
        with ThreadPoolExecutor(max_workers=3) as ex:
            round_0 = [ex.submit(a.respond_initial, query, context).result() for a in agents]
        dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
        res = synthesize(query, dialogue, llm, variant="d2c")
        return res.clarifying_question
    elif variant_name == "collaborative":
        agents = [
            Agent(AgentRole.FACT_FINDER, llm),
            Agent(AgentRole.FACET_FINDER, llm),
            Agent(AgentRole.INTENT_FINDER, llm)
        ]
        agents[0].system_prompt = COLLAB_FACT_SYSTEM
        agents[1].system_prompt = COLLAB_FACET_SYSTEM
        agents[2].system_prompt = COLLAB_INTENT_SYSTEM
        # Run standard 1-round flow
        with ThreadPoolExecutor(max_workers=3) as ex:
            round_0 = [ex.submit(a.respond_initial, query, context).result() for a in agents]
        dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
        res = synthesize(query, dialogue, llm, variant="d2c")
        return res.clarifying_question
    else:
        # Full D2C (1 round for fair comparison)
        res = run_d2c(query, variant="d2c", model=llm.model, num_rounds=1, context=context)
        return res.synthesizer_result.clarifying_question

    # Default logic for "no_X_finder"
    agents = [Agent(r, llm) for r in roles]
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        round_0 = [ex.submit(a.respond_initial, query, context).result() for a in agents]
    dialogue = DialogueResult(query=query, rounds=[round_0], num_rounds=1, context=context)
    res = synthesize(query, dialogue, llm, variant="d2c")
    return res.clarifying_question

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    random.seed(args.seed)

    llm = LLMClient(model=args.model, think=False)
    judge_llm = LLMClient(model=args.judge_model, think=False)

    sample = []
    for dataset in DATASETS:
        all_items = load_full_test_set(dataset)
        if not all_items: continue
        random.shuffle(all_items)
        sample.extend(all_items[:args.n_per_dataset])

    results_file = Path("ablation_results.txt")
    inspection_file = Path("ablation_inspections.txt")
    
    with open(results_file, "w") as f:
        f.write(f"D2C ABLATION STUDY\nSeed: {args.seed}\n" + "="*80 + "\n")
    with open(inspection_file, "w") as f:
        f.write(f"ABLATION INSPECTION LOG\nSeed: {args.seed}\n" + "="*80 + "\n")

    variants = [
        "full_d2c",
        "no_fact_finder", 
        "no_facet_finder", 
        "no_intent_finder",
        "generic_agents",
        "collaborative"
    ]

    results = {v: {"preds": [], "golds": [], "scores": [], "sims": []} for v in variants}

    print(f"Starting Ablation Study on {len(sample)} items (Seed: {args.seed})...")

    for item in tqdm(sample):
        query = item["query"]
        is_ambig = item["is_ambiguous"]
        gold_qs = item.get("gold_clarifying_questions", [])
        gold_q = gold_qs[0] if gold_qs else "N/A"

        with open(inspection_file, "a") as f_insp:
            f_insp.write(f"\nQuery: {query}\n")

        for v in variants:
            try:
                q_text = run_ablation_variant(query, v, llm, item.get("context"))
                pred_ambig = "CLEAR" not in q_text.upper()
                
                results[v]["preds"].append(pred_ambig)
                results[v]["golds"].append(is_ambig)
                
                if is_ambig:
                    j_res = llm_judge_quality(query, q_text, gold_q, judge_llm)
                    sim_score = semantic_similarity(q_text, gold_q)
                    results[v]["scores"].append(j_res["score"])
                    results[v]["sims"].append(sim_score)
                
                with open(inspection_file, "a") as f_insp:
                    f_insp.write(f"  [{v:<18}] -> {q_text[:100]}\n")
            except Exception as e:
                print(f"Error in variant {v}: {e}")

    # Final Summary Table
    print("\n" + "="*80)
    print(f"{'Variant':<20} | {'F1':<5} | {'Prec':<5} | {'Rec':<5} | {'Qual':<5} | {'Sim'}")
    print("-" * 80)
    
    with open(results_file, "a") as f:
        f.write(f"{'Variant':<20} | {'F1':<5} | {'Prec':<5} | {'Rec':<5} | {'Qual':<5} | {'Sim'}\n")
        f.write("-" * 80 + "\n")

    for v in variants:
        f1_m = clarification_need_f1(results[v]["preds"], results[v]["golds"])
        avg_qual = statistics.mean(results[v]["scores"]) if results[v]["scores"] else 0.0
        avg_sim = statistics.mean(results[v]["sims"]) if results[v]["sims"] else 0.0
        row = f"{v:<20} | {f1_m['f1']:>5.2f} | {f1_m['precision']:>5.2f} | {f1_m['recall']:>5.2f} | {avg_qual:>5.2f} | {avg_sim:>5.3f}"
        print(row)
        with open(results_file, "a") as f: f.write(row + "\n")

if __name__ == "__main__":
    main()
