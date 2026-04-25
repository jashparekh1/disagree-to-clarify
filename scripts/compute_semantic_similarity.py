import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

def parse_audit_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by item
    items = re.split(r'\[\d+/50\] QUERY:', content)[1:]
    
    parsed_data = []
    
    for item in items:
        lines = item.strip().split('\n')
        query = lines[0].strip()
        gold_cq = ""
        for line in lines:
            if "GOLD CQ:" in line:
                gold_cq = line.split("GOLD CQ:")[1].strip()
                break
        
        # Find variants
        variant_blocks = re.split(r'VARIANT:', item)[1:]
        for block in variant_blocks:
            lines_v = block.strip().split('\n')
            variant_name = lines_v[0].strip()
            
            gen_q = ""
            for line_v in lines_v:
                if "GENERATED:" in line_v:
                    gen_q = line_v.split("GENERATED:")[1].strip()
                    break
            
            if variant_name and gen_q and gold_cq:
                parsed_data.append({
                    "Query": query,
                    "Gold": gold_cq,
                    "Variant": variant_name,
                    "Generated": gen_q
                })
                
    return pd.DataFrame(parsed_data)

def main():
    audit_file = "evaluation_audit.txt"
    print(f"Loading audit log from {audit_file}...")
    df = parse_audit_log(audit_file)
    
    if df.empty:
        print("No data parsed. Check audit log format.")
        return

    print(f"Parsed {len(df)} samples. Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Computing similarities...")
    # Unique questions for efficiency
    all_qs = list(set(df['Gold'].tolist() + df['Generated'].tolist()))
    embeddings = model.encode(all_qs, show_progress_bar=True)
    emb_map = {q: emb for q, emb in zip(all_qs, embeddings)}
    
    similarities = []
    for _, row in df.iterrows():
        emb_g = emb_map[row['Gold']]
        emb_gen = emb_map[row['Generated']]
        # Cosine similarity
        sim = np.dot(emb_g, emb_gen) / (np.linalg.norm(emb_g) * np.linalg.norm(emb_gen))
        similarities.append(sim)
    
    df['SemanticSimilarity'] = similarities
    
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY RESULTS (Cosine Similarity to Gold)")
    print("="*60)
    summary = df.groupby("Variant")['SemanticSimilarity'].mean().sort_values(ascending=False)
    print(summary.to_string())
    print("="*60)
    
    output_csv = "semantic_similarity_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")

if __name__ == "__main__":
    main()
