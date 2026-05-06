"""Analysis script to prove the 'Axis of Disagreement' vs 'Ambiguity Type' correlation.
Grounds agent-pair disagreements in the ClarifyMT-Bench / CLAMBER taxonomy.
"""

import json
import logging
import pandas as pd
import numpy as np
import concurrent.futures
import argparse
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.datasets.load_clamber import load_clamber
from d2c.agents import AgentRole

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embedding model for distance calculation
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_distance(text1: str, text2: str) -> float:
    if not text1 or not text1.strip() or not text2 or not text2.strip():
        return 0.0
    embeddings = model.encode([text1, text2], normalize_embeddings=True, show_progress_bar=False)
    sim = float(embeddings[0] @ embeddings[1])
    return 1.0 - max(0.0, min(1.0, sim))

# CLAMBER to ClarifyMT-Bench Mapping
TAXONOMY_MAP = {
    "Linguistic Ambiguity/co-reference": "Linguistic",
    "Linguistic Ambiguity/polysemy": "Linguistic",
    "Aleatoric Output/what": "Intent",
    "Aleatoric Output/when": "Facet",
    "Aleatoric Output/where": "Facet",
    "Aleatoric Output/whom": "Facet",
    "Epistemic Misalignment/ICL": "Epistemic",
    "Epistemic Misalignment/NK": "Epistemic",
    "Aleatoric Output/none": "Clear"
}

def process_item(item, model_name):
    query = item.query
    context = item.context
    category = TAXONOMY_MAP.get(item.ambiguity_type, "Unknown")
    
    try:
        # Run D2C with 2 rounds to allow for Stance transitions
        res = run_d2c(query, variant="d2c", model=model_name, num_rounds=2, context=context)
        
        # 1. Semantic Distance (Initial Divergence)
        round_0 = res.dialogue.rounds[0]
        role_map_0 = {resp.role: resp.interpretation for resp in round_0}
        
        lit_0 = role_map_0.get(AgentRole.FACT_FINDER, "")
        facet_0 = role_map_0.get(AgentRole.FACET_FINDER, "")
        intent_0 = role_map_0.get(AgentRole.INTENT_FINDER, "")
        
        dist_li = get_distance(lit_0, intent_0)
        dist_lf = get_distance(lit_0, facet_0)
        dist_fi = get_distance(facet_0, intent_0)
        
        # 2. Symbolic Stance Signal (Persistence)
        # We look at Round 1 to see who refused to concede (HOLD)
        holds = {"FACT": 0, "FACET": 0, "INTENT": 0}
        if len(res.dialogue.rounds) > 1:
            round_1 = res.dialogue.rounds[1]
            for resp in round_1:
                if resp.stance.value == "HOLD":
                    if resp.role == AgentRole.FACT_FINDER: holds["FACT"] = 1
                    elif resp.role == AgentRole.FACET_FINDER: holds["FACET"] = 1
                    elif resp.role == AgentRole.INTENT_FINDER: holds["INTENT"] = 1

        return {
            "example_id": item.example_id,
            "category": category,
            "dist_lit_intent": dist_li,
            "dist_lit_facet": dist_lf,
            "dist_facet_intent": dist_fi,
            "hold_fact": holds["FACT"],
            "hold_facet": holds["FACET"],
            "hold_intent": holds["INTENT"],
            "rounds_to_converge": res.dialogue.converged_at_round if res.dialogue.converged else 3
        }
    except Exception as e:
        logger.error(f"Failed for {item.example_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument("--model", default="qwen2.5:3b", help="Model name")
    args = parser.parse_args()

    # 1. Load CLAMBER dataset using the official loader
    print("Loading CLAMBER dataset...")
    items = load_clamber()
    
    # Filter for ambiguous items with a mapped category
    valid_items = [it for it in items if it.is_ambiguous and TAXONOMY_MAP.get(it.ambiguity_type) in ["Linguistic", "Intent", "Facet", "Epistemic"]]
    
    # Sample balanced across categories
    df_items = pd.DataFrame([{"obj": it, "category": TAXONOMY_MAP.get(it.ambiguity_type)} for it in valid_items])
    
    # Target 100 per category for a "larger run" (Total 400)
    sampled_dfs = []
    for cat, group in df_items.groupby("category"):
        sampled_dfs.append(group.sample(min(len(group), 100), random_state=42))
    
    sample = pd.concat(sampled_dfs).reset_index(drop=True)
    to_process = sample["obj"].tolist()
    
    model_name = args.model
    records = []
    
    print(f"Running D2C analysis on {len(to_process)} queries with {model_name} (Parallel, workers={args.max_workers})...")
    
    # Using max_workers to match OLLAMA_NUM_PARALLEL.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, item, model_name) for item in to_process]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="D2C Queries"):
            try:
                res = f.result()
                if res:
                    records.append(res)
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    # 2. Analyze Correlations
    df = pd.DataFrame(records)
    df.to_csv("disagreement_topology.csv", index=False)
    
    # Calculate Heatmap (Mean distance per category per axis)
    heatmap = df.groupby("category")[[
        "dist_lit_intent", "dist_lit_facet", "dist_facet_intent"
    ]].mean()
    
    print("\n=== Disagreement Heatmap (IDS) ===")
    print(heatmap)
    
    # Export for paper visualization
    heatmap.to_csv("disagreement_heatmap_data.csv")
    
    print("\nCSV saved to disagreement_topology.csv and disagreement_heatmap_data.csv")
    print("You can now run 'python scripts/visualize_disagreement.py' to generate the paper plots.")

if __name__ == "__main__":
    main()
