
import json
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def parse_stances(inspection_file):
    """
    Parses stances from the detailed inspection log.
    Looking for patterns like:
    Query: ...
    Method: d2c
    ...
    [Agent Name] -> {"stance": "HOLD", ...}
    """
    if not os.path.exists(inspection_file):
        return []

    results = []
    with open(inspection_file) as f:
        content = f.read()
    
    # Split by query
    blocks = content.split("Query: ")
    for block in blocks[1:]:
        lines = block.split("\n")
        query = lines[0].strip()
        
        # We only care about the 'd2c' method for this analysis
        if "Method: d2c" not in block:
            continue
            
        # Extract stances for the three roles
        # FACT-FINDER, FACET-FINDER, INTENT-FINDER
        fact_stance = "CONCEDE"
        facet_stance = "CONCEDE"
        intent_stance = "CONCEDE"
        
        # Look for the last stance mentioned for each role
        # This is a bit heuristic based on how the log is written
        # In a real run, we'd use the structured dialogue objects
        
        # For simplicity in this script, let's assume we can find them via regex
        # as the log contains the agent outputs
        stances = re.findall(r'\[(.*?)\] -> .*?"stance":\s*"(.*?)"', block)
        
        # Map roles
        latest_stances = {}
        for role, stance in stances:
            latest_stances[role.upper()] = stance
            
        results.append({
            "query": query,
            "fact_finder": latest_stances.get("FACT-FINDER", "CONCEDE"),
            "facet_finder": latest_stances.get("FACET-FINDER", "CONCEDE"),
            "intent_finder": latest_stances.get("INTENT-FINDER", "CONCEDE")
        })
        
    return results

def get_clamber_labels():
    """Load ground truth labels from CLAMBER."""
    path = Path("test_sets/clamber_test.jsonl")
    labels = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                # CLAMBER labels are in the 'ambiguity_type' field
                # Expected values: Lexical, Aleatoric, Epistemic
                labels[item["query"]] = item.get("ambiguity_type", "Unknown")
    return labels

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("Generating Multi-Modal Disagreement Visualization...")
    
    if not os.path.exists("disagreement_topology.csv"):
        print("Error: disagreement_topology.csv not found.")
        return
        
    df = pd.read_csv("disagreement_topology.csv")
    
    # 1. HEATMAP 1: SEMANTIC DIVERGENCE (IDS)
    summary_ids = df.groupby("category")[[
        "dist_lit_intent", "dist_lit_facet", "dist_facet_intent"
    ]].mean()
    summary_ids.columns = ["L-I Axis", "L-F Axis", "F-I Axis"]
    
    # Apply column-wise normalization (Z-score) to highlight relative peaks
    summary_ids_norm = (summary_ids - summary_ids.mean()) / summary_ids.std()
    
    plt.figure(figsize=(10, 6))
    # Use Z-score for BOTH color and annotation
    sns.heatmap(summary_ids_norm, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Relative Semantic Divergence (Z-Score of Mean IDS)\n(Shows where each category 'over-performs' on an axis)")
    plt.ylabel("Ambiguity Type (CLAMBER)")
    plt.tight_layout()
    plt.savefig("paper_heatmap_semantic.png")
    
    # 2. HEATMAP 2: SYMBOLIC PERSISTENCE (HOLD Frequency)
    summary_holds = df.groupby("category")[[
        "hold_fact", "hold_facet", "hold_intent"
    ]].mean()
    summary_holds.columns = ["Fact-Finder", "Facet-Finder", "Intent-Finder"]
    
    # Normalize holds as well
    summary_holds_norm = (summary_holds - summary_holds.mean()) / summary_holds.std()
    
    plt.figure(figsize=(10, 6))
    # Use Z-score for BOTH color and annotation
    sns.heatmap(summary_holds_norm, annot=True, cmap="OrRd", fmt=".2f")
    plt.title("Relative Symbolic Persistence (Z-Score of HOLD Rate)")
    plt.ylabel("Ambiguity Type (CLAMBER)")
    plt.tight_layout()
    plt.savefig("paper_heatmap_symbolic.png")
    
    # 3. DIAGNOSTIC ANALYSIS (Using Normalized Peaks)
    print("\n=== Multi-Modal Topology Results (Normalized) ===")
    for cat in summary_ids_norm.index:
        peak_ids = summary_ids_norm.loc[cat].idxmax()
        peak_hold = summary_holds_norm.loc[cat].idxmax()
        print(f"Category: {cat:<12} | Relative Semantic Peak: {peak_ids} | Relative Symbolic Peak: {peak_hold}")

    print("\nVisualizations saved to paper_heatmap_semantic.png and paper_heatmap_symbolic.png")

if __name__ == "__main__":
    main()
