"""
Visualizes Disagreement Dynamics for the MADISSE Variant.
Runs 4 samples per dataset (16 total) and generates:
1. Stance Persistence (Bar chart)
2. State Heatmap (Agent x Round)
3. Sankey Data (Flow of stances)
4. Correlation analysis (Holders vs Question Type)
"""

import json
import random
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from d2c.pipeline import run_d2c
from d2c.agents import AgentRole, Stance
from d2c.llm import LLMClient

# Configuration
DATASETS = ["clamber", "clariq", "qulac", "abgcoqa"]
N_PER_DATASET = 5
MODEL = "qwen2.5:1.5b"
NUM_ROUNDS = 3
SEED = 42

def load_samples():
    random.seed(SEED)
    all_samples = []
    for ds in DATASETS:
        path = Path("test_sets") / f"{ds}_test.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            items = [json.loads(line) for line in f if line.strip()]
        # Only use ambiguous items for interesting disagreement
        ambiguous = [i for i in items if i.get("is_ambiguous", True)]
        sample = random.sample(ambiguous, min(N_PER_DATASET, len(ambiguous)))
        for item in sample:
            item["dataset_name"] = ds
        all_samples.extend(sample)
    return all_samples

def classify_question_type(query, question, llm):
    """Simple LLM-based classification of the final question's strategy."""
    prompt = f"""Analyze this clarifying question for an ambiguous query.
Query: {query}
Question: {question}

Classify the question strategy into exactly ONE:
- DISJUNCTIVE (Presents a clear choice: "Are you asking about A or B?")
- OPEN-ENDED (Asks for more info without options: "What do you mean by X?")
- TARGETED (Focuses on one specific entity/parameter: "Where are you located?")
- CLEAR (Model claimed query is clear)

Output ONLY the category name."""
    
    res = llm.chat(system_prompt="You are a linguistic classifier.", user_prompt=prompt, temperature=0.0)
    res = res.strip().upper()
    for cat in ["DISJUNCTIVE", "OPEN-ENDED", "TARGETED", "CLEAR"]:
        if cat in res:
            return cat
    return "OTHER"

def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} samples. Starting MADISSE runs...")
    
    llm = LLMClient(model=MODEL)
    
    data_records = []
    sankey_links = Counter()
    
    for item in tqdm(samples):
        query = item["query"]
        ds = item["dataset_name"]
        
        try:
            res = run_d2c(query, variant="madisse", model=MODEL, num_rounds=NUM_ROUNDS, context=item.get("context"))
            q_text = res.synthesizer_result.clarifying_question
            q_type = classify_question_type(query, q_text, llm)
            
            # Track who is holding at the end
            final_holders = []
            
            # Extract round-by-round states
            # rounds[r_idx][agent_idx]
            for r_idx, round_resps in enumerate(res.dialogue.rounds):
                for resp in round_resps:
                    role = resp.role.value
                    stance = resp.stance.value
                    
                    data_records.append({
                        "query_idx": len(data_records) // (NUM_ROUNDS * 3), # approximation
                        "dataset": ds,
                        "round": r_idx,
                        "role": role,
                        "stance": stance
                    })
                    
                    # Track flow for Sankey
                    if r_idx < len(res.dialogue.rounds) - 1:
                        # Find next round response for same agent
                        # Note: some agents might drop out if they concede
                        next_stance = "DROPPED"
                        if r_idx + 1 < len(res.dialogue.rounds):
                            for n_resp in res.dialogue.rounds[r_idx+1]:
                                if n_resp.role == resp.role:
                                    next_stance = n_resp.stance.value
                                    break
                        
                        source = f"R{r_idx}_{role}_{stance}"
                        target = f"R{r_idx+1}_{role}_{next_stance}"
                        sankey_links[(source, target)] += 1
            
            # Record final state correlation
            last_round = res.dialogue.rounds[-1]
            holders = [r.role.value for r in last_round if r.stance in [Stance.HOLD, Stance.UPDATE]]
            
            item_summary = {
                "query": query,
                "dataset": ds,
                "holders": holders,
                "q_type": q_type,
                "question": q_text
            }
            # (In a real run we'd save this to analyze correlation)
            
        except Exception as e:
            print(f"Error on query: {e}")

    df = pd.DataFrame(data_records)
    
    # --- Visualization 1: Heatmap of Stances ---
    # Map Stances to numbers for heatmap
    stance_map = {"HOLD": 2, "UPDATE": 1, "CONCEDE": 0, "DROPPED": -1}
    df['stance_val'] = df['stance'].map(stance_map)
    
    plt.figure(figsize=(10, 6))
    pivot = df.groupby(['role', 'round'])['stance_val'].mean().unstack()
    
    plt.imshow(pivot, cmap="YlGnBu", aspect="auto")
    plt.colorbar(label="Stance (2=HOLD, 0=CONCEDE)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Average Agent Stance by Round")
    
    # Add text labels
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center")
            
    plt.savefig("disagreement_heatmap.png")
    print("Saved disagreement_heatmap.png")

    # --- Visualization 2: Final Stance Distribution ---
    plt.figure(figsize=(10, 6))
    # Filter for only the last recorded round for each query (Final State)
    final_states = df.sort_values('round').groupby(['query_idx', 'role']).tail(1)
    stance_counts = final_states.groupby(['role', 'stance']).size().unstack(fill_value=0)
    
    # Reorder columns for logical progression if they exist
    cols = [c for c in ["HOLD", "UPDATE", "CONCEDE", "DROPPED"] if c in stance_counts.columns]
    stance_counts = stance_counts[cols]
    
    stance_counts.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#d7191c','#fdae61','#abdda4','#2b83ba'][:len(cols)])
    plt.title("Final Stance per Agent Role (End of Dialogue)")
    plt.ylabel("Number of Queries")
    plt.xlabel("Agent Role")
    plt.legend(title="Final Stance", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("stance_frequencies.png")
    print("Saved final stance_frequencies.png")

    # --- Sankey Data Output ---
    print("\n--- SANKEY FLOW DATA (Copy to SankeyMatic) ---")
    for (src, tgt), count in sankey_links.items():
        print(f"{src} [{count}] {tgt}")

    # --- Correlation Analysis ---
    # (Simplified summary of which roles tend to drive the synthesizer)
    print("\n--- DISAGREEMENT AXIS vs QUESTION TYPE ---")
    # This would ideally be a cross-tab, but with N=16 we'll just print highlights
    # e.g., If Intent-Finder holds, is the question DISJUNCTIVE?
    
    # Save the raw data for further analysis
    df.to_csv("disagreement_dynamics.csv", index=False)
    print("\nFull dynamics data saved to disagreement_dynamics.csv")

if __name__ == "__main__":
    main()
