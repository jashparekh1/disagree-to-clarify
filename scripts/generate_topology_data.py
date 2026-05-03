
import json
import os
import random
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.metrics import semantic_similarity

def get_dist(text1, text2):
    return 1.0 - semantic_similarity(text1, text2)

def main():
    print("Starting Disagreement Topology Generation (Large Run)...")
    
    # 1. Load CLAMBER data for ground truth labels
    clamber_path = Path("test_sets/clamber_test.jsonl")
    items = []
    with open(clamber_path) as f:
        for line in f:
            item = json.loads(line)
            # Only ambiguous items that have a category
            if item.get("is_ambiguous") and item.get("ambiguity_type"):
                items.append(item)
    
    random.shuffle(items)
    # We'll target 100 samples total (or more if you prefer)
    sample = items[:150] 
    
    llm = LLMClient(model="qwen2.5:1.5b", think=False)
    results = []

    def process_query(item):
        query = item["query"]
        raw_cat = item["ambiguity_type"]
        category = raw_cat.split()[0].split('/')[0]
        
        try:
            # Run D2C (3 rounds)
            res = run_d2c(query, model="qwen2.5:1.5b", num_rounds=3, context=item.get("context"))
            
            # Extract interpretations from Round 0 (The "Primal" Disagreement)
            round_0 = res.dialogue.rounds[0]
            # Fact, Facet, Intent
            lit_txt = next(r.interpretation for r in round_0 if "Fact" in r.role or "Literal" in r.role)
            facet_txt = next(r.interpretation for r in round_0 if "Facet" in r.role)
            intent_txt = next(r.interpretation for r in round_0 if "Intent" in r.role)
            
            # Calculate Semantic Distances (The Axes)
            d_lit_facet = get_dist(lit_txt, facet_txt)
            d_lit_intent = get_dist(lit_txt, intent_txt)
            d_facet_intent = get_dist(facet_txt, intent_txt)
            
            # Extract Final Stances (Round 3)
            final_round = res.dialogue.rounds[-1]
            h_fact = 1 if any("Fact" in r.role and r.stance == "HOLD" for r in final_round) else 0
            h_facet = 1 if any("Facet" in r.role and r.stance == "HOLD" for r in final_round) else 0
            h_intent = 1 if any("Intent" in r.role and r.stance == "HOLD" for r in final_round) else 0
            
            return {
                "query": query[:50],
                "category": category,
                "dist_lit_facet": d_lit_facet,
                "dist_lit_intent": d_lit_intent,
                "dist_facet_intent": d_facet_intent,
                "hold_fact": h_fact,
                "hold_facet": h_facet,
                "hold_intent": h_intent
            }
        except Exception as e:
            # print(f"Error on {query[:30]}: {e}")
            return None

    print(f"Processing {len(sample)} queries to map the Disagreement Axis...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_query, item) for item in sample]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("disagreement_topology.csv", index=False)
    print(f"\nSaved topology data for {len(results)} samples to disagreement_topology.csv")

if __name__ == "__main__":
    main()
