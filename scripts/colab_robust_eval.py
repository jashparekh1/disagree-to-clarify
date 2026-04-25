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

# High-quality Utility Rubric
UTILITY_JUDGE_SYSTEM = """You are an expert evaluator of clarifying questions. 
Assign a Utility Score (0-100) based on how effectively the question resolves query ambiguity.

Rubric:
- 0-20: Useless. Hallucinated or irrelevant.
- 21-50: Fair. Generic or slightly off-target.
- 51-80: Good. Targets a specific detail or sub-topic.
- 81-100: Excellent. Perfect surgical question.

Output EXACT JSON:
{
  "score": (0-100),
  "reasoning": "brief explanation"
}"""

UTILITY_JUDGE_USER = """Query: {query}
Interpretations: {gold}

Question: {candidate}"""

def main():
    gen_model = "qwen3:1.7b"
    judge_models = ["qwen3.5:4b", "qwen2.5:7b"]
    
    llm_gen = LLMClient(model=gen_model)
    judges = {m: LLMClient(model=m, think=False) for m in judge_models}
    
    datasets = ["clamber", "qulac", "clariq"]
    n_total = 50
    variants = ["vanilla", "speech_act", "speech_act_surgical"]
    
    # Load and sample 50 items total
    all_items = []
    for ds in datasets:
        items = [q for q in load_dataset(ds) if q.is_ambiguous]
        all_items.extend([(ds, item) for item in items])
    
    sample = random.sample(all_items, min(n_total, len(all_items)))
    
    results = []
    output_file = "colab_robust_results.csv"
    
    print(f"{'='*60}")
    print(f"COLAB ROBUST EVALUATION")
    print(f"Total Samples: {len(sample)} | Variants: {variants}")
    print(f"{'='*60}")

    for i, (ds_name, item) in enumerate(sample):
        query = item.query
        gold = item.gold_clarifying_question
        
        print(f"\n[{i+1}/{len(sample)}] Query: {query[:70]}...")
        
        for var in variants:
            # 1. Generate Question
            if var == "vanilla":
                gen_q = llm_gen.chat(VANILLA_CQG_SYSTEM, VANILLA_CQG_USER.format(query=query))
            else:
                d2c_res = run_d2c(query, model=gen_model, num_rounds=1, variant=var, think=False)
                gen_q = d2c_res.synthesizer_result.clarifying_question
            
            # 2. Evaluate with both judges
            for j_name, llm_judge in judges.items():
                j_prompt = UTILITY_JUDGE_USER.format(query=query, gold=gold, candidate=gen_q)
                schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
                
                try:
                    res = json.loads(llm_judge.chat(UTILITY_JUDGE_SYSTEM, j_prompt, format_schema=schema))
                    score = res["score"]
                except:
                    score = 0
                
                results.append({
                    "ID": i+1,
                    "Dataset": ds_name,
                    "Variant": var,
                    "Judge": j_name,
                    "Score": score,
                    "Question": gen_q
                })
        
        # Calculate and print rolling averages
        df_temp = pd.DataFrame(results)
        pivot = df_temp.pivot_table(index="Variant", columns="Judge", values="Score", aggfunc="mean")
        print("Rolling Averages:")
        print(pivot.to_string())
        sys.stdout.flush()

    # Save to file
    df_final = pd.DataFrame(results)
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print(df_final.pivot_table(index="Variant", columns="Judge", values="Score", aggfunc="mean"))
    print("="*60)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
