"""
Run a small batch of queries through 4 variants and save full details for manual inspection.
"""
import json
from d2c.pipeline import run_d2c
from d2c.llm import LLMClient
from d2c.baseline import run_vanilla_cqg
from eval.metrics import llm_judge_quality
from pathlib import Path

def main():
    queries = [
        "french lick resort and casino",
        "How should I use the septic system design",
        "Find sewing instructions for me",
        "I'm looking for information on the president of the united states",
        "jax chemical company"
    ]
    
    # Gold reference questions for judging (mimicking dataset format)
    golds = {
        "french lick resort and casino": "Are you looking for the history of the resort or a list of available casino games?",
        "How should I use the septic system design": "Are you looking for installation instructions or maintenance tips for your septic system?",
        "Find sewing instructions for me": "What specific sewing project or garment are you looking for instructions on?",
        "I'm looking for information on the president of the united states": "Are you interested in the current president's biography or a list of all historical presidents?",
        "jax chemical company": "Are you looking for the product catalog or the safety data sheets for JAX Chemical Company?"
    }
    
    variants = ["speech_act", "madisse", "taxonomy"]
    model = "qwen2.5:1.5b"
    judge_model = "qwen2.5:7b" # Larger judge as requested
    
    judge_llm = LLMClient(model=judge_model, think=False)
    
    output_file = "variant_inspections.txt"
    with open(output_file, "w") as f:
        f.write("D2C VARIANT INSPECTION REPORT (With LLM-as-Judge Scoring)\n")
        f.write("="*80 + "\n\n")
        
        for i, q in enumerate(queries):
            f.write(f"QUERY {i+1}: {q}\n")
            f.write("-" * 40 + "\n")
            
            gold_q = golds.get(q, "")
            
            # 1. Vanilla
            vanilla_llm = LLMClient(model=model, think=False)
            v_res = run_vanilla_cqg(q, vanilla_llm)
            v_judge = llm_judge_quality(q, v_res.clarifying_question, gold_q, judge_llm)
            f.write(f"[VANILLA]\n")
            f.write(f"Question: {v_res.clarifying_question}\n")
            f.write(f"JUDGE SCORE: {v_judge['score']}/5\n")
            f.write(f"JUDGE REASON: {v_judge['reasoning']}\n\n")
            
            # 2. Variants
            for var in variants:
                print(f"Running {var} for query {i+1}...")
                res = run_d2c(q, variant=var, model=model, num_rounds=2 if var == "speech_act" else 1)
                q_text = res.synthesizer_result.clarifying_question
                
                # Judge the variant
                judge_res = llm_judge_quality(q, q_text, gold_q, judge_llm)
                
                f.write(f"[{var.upper()}]\n")
                f.write(f"Question: {q_text}\n")
                f.write(f"JUDGE SCORE: {judge_res['score']}/5\n")
                f.write(f"JUDGE REASON: {judge_res['reasoning']}\n")
                f.write("Dialogue/Logic summary:\n")
                for rnd_idx, rnd in enumerate(res.dialogue.rounds):
                    f.write(f"  Round {rnd_idx}:\n")
                    for resp in rnd:
                        f.write(f"    - {resp.role.value}: {resp.interpretation[:150]}...\n")
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n\n")

    print(f"Done. Inspection saved to {output_file}")

if __name__ == "__main__":
    main()
