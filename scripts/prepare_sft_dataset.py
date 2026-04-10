import json
import os
import concurrent.futures
from tqdm import tqdm
from d2c.llm import LLMClient

# Use a smaller model for fast testing
GOLD_MODEL = "qwen2.5:0.5b"
llm = LLMClient(model=GOLD_MODEL)

PROMPT_TEMPLATE = """You are an expert at creating clarifying questions for ambiguous queries.
Given an original query and several disambiguated interpretations, create a single, concise, and natural clarifying question that would help a user specify which of these interpretations they meant.

Original Query: {query}
Interpretations:
{interpretations}

Clarifying Question:"""

def generate_gold_question(item):
    query = item.get("question")
    interpretations = []
    for ann in item.get("annotations", []):
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    
    if len(interpretations) < 2:
        return None
    
    formatted_ints = "\n".join(f"- {i}" for i in interpretations[:5]) # limit to 5
    prompt = PROMPT_TEMPLATE.format(query=query, interpretations=formatted_ints)
    
    try:
        # We want the highest quality so we use a lower temperature
        gold_q = llm.chat(
            system_prompt="You are an expert assistant that generates precise clarifying questions.",
            user_prompt=prompt,
            temperature=0.1
        )
        # mlx-lm usually likes a "text" field or specific "instruction"/"output"
        # We'll use a simple format that works well with chat templates
        return {
            "text": f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{gold_q}<|im_end|>"
        }
    except Exception:
        return None

def main():
    print("Loading training data...")
    with open('data/train_light.json') as f:
        data = json.load(f)
    
    # Filter for ambiguous queries first to avoid wasting LLM calls
    ambiguous = []
    for item in data:
        for ann in item.get("annotations", []):
            if ann.get("type") == "multipleQAs" and len(ann.get("qaPairs", [])) >= 2:
                ambiguous.append(item)
                break
    
    print(f"Found {len(ambiguous)} ambiguous queries. Generating gold questions for 200 items...")
    target_count = 200
    sample = ambiguous[:target_count*2]
    
    os.makedirs('data/sft', exist_ok=True)
    
    with open('data/sft/raw_gold.jsonl', 'a') as out:
        dataset = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(generate_gold_question, item) for item in sample]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                res = future.result()
                if res:
                    dataset.append(res)
                    out.write(json.dumps(res) + '\n')
                    out.flush()
                    if len(dataset) >= target_count:
                        # Cancel remaining
                        for f in futures:
                            f.cancel()
                        break
    
    # Split 80/10/10
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    for name, d in [("train", train_data), ("valid", val_data), ("test", test_data)]:
        with open(f'data/sft/{name}.jsonl', 'w') as f:
            for item in d:
                f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(train_data)} train, {len(val_data)} valid, {len(test_data)} test items to data/sft/")

if __name__ == "__main__":
    main()
