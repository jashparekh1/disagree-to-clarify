import json
import os
import sys
import pandas as pd
import random
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.datasets import load_dataset
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER

UTILITY_JUDGE_SYSTEM = """You are an expert evaluator of clarifying questions. 
Assign a Utility Score (0-100) based on how effectively the question resolves query ambiguity.

Rubric:
- 0-20: Useless. Hallucinated, irrelevant, or robotic.
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
    judge_model = "qwen3.5:4b"
    llm_gen = LLMClient(model=gen_model)
    llm_judge = LLMClient(model=judge_model, think=False)
    
    datasets = ["clamber", "qulac", "clariq"]
    n_per_ds = 10
    variants = ["vanilla", "speech_act", "info_gap"]
    
    print(f"{'='*60}")
    print(f"ITERATIVE COMPARISON: Searching for the 'Better' Variant")
    print(f"Generator: {gen_model} | Judge: {judge_model}")
    print(f"{'='*60}")
    sys.stdout.flush()

    results = []
    
    # Pool items for random sampling
    pool = []
    for ds in datasets:
        items = [q for q in load_dataset(ds) if q.is_ambiguous]
        pool.extend(random.sample(items, min(n_per_ds, len(items))))

    for item in pool:
        query = item.query
        gold = item.gold_clarifying_question
        
        print(f"\nQuery: {query[:70]}...")
        for var in variants:
            if var == "vanilla":
                prompt = VANILLA_CQG_USER.format(query=query)
                gen_q = llm_gen.chat(system_prompt=VANILLA_CQG_SYSTEM, user_prompt=prompt)
            else:
                d2c_res = run_d2c(query, model=gen_model, num_rounds=1, variant=var, think=False)
                gen_q = d2c_res.synthesizer_result.clarifying_question
            
            # Judge
            j_prompt = UTILITY_JUDGE_USER.format(query=query, gold=gold, candidate=gen_q)
            schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
            try:
                res = json.loads(llm_judge.chat(UTILITY_JUDGE_SYSTEM, j_prompt, format_schema=schema))
                score = res["score"]
            except:
                score = 0
            
            results.append({"Variant": var, "Score": score})
            print(f"  [{var:20}] Score: {score:3} | Q: {gen_q[:50]}...")
            sys.stdout.flush()

        # Rolling averages
        df_temp = pd.DataFrame(results)
        means = df_temp.groupby("Variant")["Score"].mean().to_dict()
        summary = " | ".join([f"{k}: {v:.1f}" for k, v in means.items()])
        print(f"ROLLING AVG: {summary}")
        sys.stdout.flush()

    print("\n" + "="*60)
    print("FINAL ITERATIVE RESULTS")
    print(df_temp.groupby("Variant")["Score"].mean().to_string())
    print("="*60)

if __name__ == "__main__":
    main()
