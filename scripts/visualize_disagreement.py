
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

def main():
    print("Analyzing Disagreement Dynamics...")
    
    # 1. Parse stances from the actual run logs
    stances = parse_stances("variant_inspections.txt")
    if not stances:
        print("No stances found. Ensure variant_inspections.txt exists and contains 'd2c' runs.")
        return
        
    # 2. Get ground truth
    clamber_labels = get_clamber_labels()
    
    # 3. Merge
    data = []
    for s in stances:
        label = clamber_labels.get(s["query"])
        if label and label != "Unknown":
            data.append({
                "Ambiguity Type": label,
                "Fact-Finder HOLD": 1 if s["fact_finder"] == "HOLD" else 0,
                "Facet-Finder HOLD": 1 if s["facet_finder"] == "HOLD" else 0,
                "Intent-Finder HOLD": 1 if s["intent_finder"] == "HOLD" else 0
            })
            
    df = pd.DataFrame(data)
    if df.empty:
        print("No overlap found between logs and CLAMBER labels.")
        return
        
    # 4. Generate Correlation Heatmap
    # Group by Ambiguity Type and calculate mean HOLD frequency
    analysis = df.groupby("Ambiguity Type").mean()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(analysis, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Disagreement Signature vs. Ambiguity Type\n(Mean 'HOLD' Frequency per Agent)")
    plt.ylabel("Ground Truth (CLAMBER)")
    plt.xlabel("D2C Agent Stance")
    plt.tight_layout()
    plt.savefig("disagreement_heatmap.png")
    
    # 5. Statistical Significance Test (Chi-Square)
    from scipy.stats import chi2_contingency
    print("\n--- Statistical Significance (Chi-Square Test) ---")
    for agent in ["Fact-Finder HOLD", "Facet-Finder HOLD", "Intent-Finder HOLD"]:
        contingency = pd.crosstab(df["Ambiguity Type"], df[agent])
        chi2, p, dof, ex = chi2_contingency(contingency)
        print(f"{agent:<18} | Chi2: {chi2:>6.2f} | p-value: {p:>8.4f}")
    
    # 6. Save results
    df.to_csv("disagreement_dynamics.csv", index=False)
    print("\nAnalysis complete. Saved disagreement_heatmap.png and disagreement_dynamics.csv")

if __name__ == "__main__":
    main()
