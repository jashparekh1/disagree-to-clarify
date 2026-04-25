import json
import os
import sys
import pandas as pd
import random
import time
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.datasets import load_dataset
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER
from d2c.synthesizer import _format_transcript

# High-quality Utility Rubric
UTILITY_JUDGE_SYSTEM = """You are an expert evaluator of clarifying questions. 
Assign a Utility Score (0-100) and provide detailed REASONING.

Rubric:
- 0-20: Useless. Hallucinated or irrelevant.
- 21-50: Fair. Generic or slightly off-target.
- 51-80: Good. Targets a specific detail or sub-topic.
- 81-100: Excellent. Perfect surgical question.

Output EXACT JSON:
{
  "score": (0-100),
  "reasoning": "detailed explanation"
}"""

UTILITY_JUDGE_USER = """Query: {query}
Interpretations: {gold}

Question: {candidate}"""

def main():
    gen_model = "qwen2.5:1.5b"
    judge_models = ["qwen3.5:4b", "gemma2:9b"]
    
    llm_gen = LLMClient(model=gen_model)
    judges = {m: LLMClient(model=m, think=False) for m in judge_models}
    
    datasets = ["clamber", "qulac", "clariq"]
    n_total = 50
    variants = ["vanilla", "speech_act", "speech_act_surgical"]
    
    # Load and sample
    all_items = []
    for ds in datasets:
        items = [q for q in load_dataset(ds) if q.is_ambiguous]
        all_items.extend([(ds, item) for item in items])
    
    sample = random.sample(all_items, min(n_total, len(all_items)))
    
    results = []
    csv_file = "colab_detailed_results.csv"
    audit_file = "evaluation_audit.txt"
    
    with open(audit_file, "w") as f_audit:
        f_audit.write("="*80 + "\n")
        f_audit.write("D2C DETAILED EVALUATION AUDIT\n")
        f_audit.write(f"Generator: {gen_model} | Judges: {', '.join(judge_models)}\n")
        f_audit.write("="*80 + "\n\n")

        for i, (ds_name, item) in enumerate(sample):
            query = item.query
            gold = item.gold_clarifying_question
            
            f_audit.write(f"\n[{i+1}/{len(sample)}] QUERY: {query}\n")
            f_audit.write(f"     GOLD CQ: {gold}\n")
            f_audit.write("-" * 40 + "\n")
            
            print(f"\n[{i+1}/{len(sample)}] Query: {query[:50]}...")

            for var in variants:
                # 1. Generate Question
                if var == "vanilla":
                    gen_q = llm_gen.chat(VANILLA_CQG_SYSTEM, VANILLA_CQG_USER.format(query=query))
                    trace = "Trace: Single-step prompt."
                else:
                    d2c_res = run_d2c(query, model=gen_model, num_rounds=1, variant=var, think=False)
                    gen_q = d2c_res.synthesizer_result.clarifying_question
                    trace = _format_transcript(d2c_res.dialogue)
                
                f_audit.write(f"\nVARIANT: {var}\n")
                f_audit.write(f"GENERATED: {gen_q}\n")
                f_audit.write(f"DIALOGUE TRACE:\n{trace}\n")
                
                # 2. Evaluate with both judges
                for j_name, llm_judge in judges.items():
                    j_prompt = UTILITY_JUDGE_USER.format(query=query, gold=gold, candidate=gen_q)
                    schema = {
                        "type": "object", 
                        "properties": {
                            "score": {"type": "integer"},
                            "reasoning": {"type": "string"}
                        }, 
                        "required": ["score", "reasoning"]
                    }
                    
                    try:
                        res_raw = llm_judge.chat(UTILITY_JUDGE_SYSTEM, j_prompt, format_schema=schema)
                        res = json.loads(res_raw)
                        score = res["score"]
                        reasoning = res["reasoning"]
                    except:
                        score, reasoning = 0, "Judge failed to respond."
                    
                    results.append({
                        "ID": i+1,
                        "Variant": var,
                        "Judge": j_name,
                        "Score": score,
                        "Question": gen_q
                    })
                    
                    f_audit.write(f"JUDGE ({j_name}) SCORE: {score}/100\n")
                    f_audit.write(f"JUDGE ({j_name}) REASON: {reasoning}\n")

            # Rolling averages print
            df_temp = pd.DataFrame(results)
            pivot = df_temp.pivot_table(index="Variant", columns="Judge", values="Score", aggfunc="mean")
            print(f"Rolling Avgs (after {i+1} samples):")
            print(pivot.to_string())
            sys.stdout.flush()
            f_audit.flush()

        # Final Summary
        df_final = pd.DataFrame(results)
        df_final.to_csv(csv_file, index=False)
        
        f_audit.write("\n\n" + "="*80 + "\n")
        f_audit.write("FINAL SUMMARY TABLE\n")
        f_audit.write("="*80 + "\n")
        f_audit.write(df_final.pivot_table(index="Variant", columns="Judge", values="Score", aggfunc="mean").to_string())
        f_audit.write("\n" + "="*80 + "\n")

    print(f"\nEvaluation complete. Summary: {csv_file}, Audit: {audit_file}")

if __name__ == "__main__":
    main()
