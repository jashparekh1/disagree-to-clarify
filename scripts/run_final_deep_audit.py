import json
import os
import sys
import pandas as pd
import random
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.datasets import load_dataset
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER
from d2c.synthesizer import _format_transcript

# Rubric for the 4B Judge
UTILITY_JUDGE_SYSTEM = """You are an expert evaluator of clarifying questions. 
Assign a Utility Score (0-100) and provide detailed REASONING.

Rubric:
- 0-20: Useless/Hallucinated.
- 21-50: Fair. Generic or slightly off-target.
- 51-80: Good. Specific and natural.
- 81-100: Excellent. Perfect surgical question.

Output EXACT JSON:
{
  "score": (0-100),
  "reasoning": "detailed explanation"
}"""

def main():
    gen_model = "qwen2.5:1.5b"
    judge_model = "qwen3.5:4b"
    llm_gen = LLMClient(model=gen_model)
    llm_judge = LLMClient(model=judge_model, think=False)
    
    datasets = ["clamber", "qulac", "clariq"]
    n_per_ds = 5
    variants = ["vanilla", "speech_act", "speech_act_surgical"]
    
    output_file = "final_deep_audit.txt"
    results = []
    
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("FINAL DEEP AUDIT: D2C PERFORMANCE EVALUATION\n")
        f.write(f"Generator: {gen_model} | Judge: {judge_model}\n")
        f.write("="*80 + "\n\n")

        for ds_name in datasets:
            f.write(f"\n\n{'#'*80}\n")
            f.write(f"  DATASET: {ds_name.upper()}\n")
            f.write(f"{'#'*80}\n")
            
            items = [q for q in load_dataset(ds_name) if q.is_ambiguous]
            sample = random.sample(items, min(n_per_ds, len(items)))
            
            for idx, item in enumerate(sample):
                query = item.query
                gold = item.gold_clarifying_question
                
                f.write(f"\n[{idx+1}] QUERY: {query}\n")
                f.write(f"    GOLD CQ: {gold}\n")
                f.write("-" * 40 + "\n")

                for var in variants:
                    f.write(f"\n--- VARIANT: {var} ---\n")
                    
                    if var == "vanilla":
                        prompt = VANILLA_CQG_USER.format(query=query)
                        gen_q = llm_gen.chat(system_prompt=VANILLA_CQG_SYSTEM, user_prompt=prompt)
                        f.write(f"Trace: (Single-step baseline, no dialogue)\n")
                    else:
                        d2c_res = run_d2c(query, model=gen_model, num_rounds=1, variant=var, think=False)
                        gen_q = d2c_res.synthesizer_result.clarifying_question
                        f.write("Dialogue Trace:\n")
                        f.write(_format_transcript(d2c_res.dialogue))
                        f.write("\n")

                    # Judge
                    j_prompt = f"Query: {query}\nInterpretations: {gold}\n\nQuestion: {gen_q}"
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
                    
                    f.write(f"GENERATED Q: {gen_q}\n")
                    f.write(f"JUDGE SCORE: {score}/100\n")
                    f.write(f"JUDGE REASON: {reasoning}\n")
                    
                    results.append({"Variant": var, "Score": score})
                    print(f"Processed {ds_name} Item {idx+1} | {var}")
                    sys.stdout.flush()

        # Add Final Table to bottom
        df = pd.DataFrame(results)
        summary = df.groupby("Variant")["Score"].mean().to_string()
        f.write("\n\n" + "="*80 + "\n")
        f.write("FINAL PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(summary + "\n")
        f.write("="*80 + "\n")

    print(f"\nAudit complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
