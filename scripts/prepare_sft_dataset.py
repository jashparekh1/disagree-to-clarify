import json
import os
import random
import concurrent.futures
from tqdm import tqdm
from d2c.llm import LLMClient

# Use a high-quality model for distillation (Teacher)
GOLD_MODEL = "qwen3:4b" 
llm = LLMClient(model=GOLD_MODEL)

# This prompt "cheats" by showing the teacher the ground truth interpretations
PROMPT_TEMPLATE = """You are an expert at creating clarifying questions for ambiguous queries.
Given an original query and several disambiguated interpretations, create a single, concise, and natural clarifying question that would help a user specify which of these interpretations they meant.

The question should:
1. Be polite and natural.
2. Explicitly mention the different interpretations to help the user choose (e.g., "Are you asking about X or Y?").
3. Be concise and avoid conversational filler.

Original Query: {query}
Ground Truth Interpretations:
{interpretations}

Clarifying Question:"""

def generate_gold_question(item):
    query = item.get("question")
    interpretations = []
    # Extract the ground truth interpretations from AmbigQA format
    for ann in item.get("annotations", []):
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    
    if len(interpretations) < 2:
        return None
    
    # Format the interpretations for the Teacher prompt
    formatted_ints = "\n".join(f"- {i}" for i in interpretations[:5])
    prompt = PROMPT_TEMPLATE.format(query=query, interpretations=formatted_ints)
    
    try:
        # Teacher generates the "ideal" question
        gold_q = llm.chat(
            system_prompt="You are an expert assistant that generates precise clarifying questions.",
            user_prompt=prompt,
            temperature=0.1
        )
        
        # MLX-LM format: JSONL with 'messages' for chat-finetuning
        return {
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": gold_q}
            ]
        }
    except Exception:
        return None

def main():
    print("Starting Distillation Process...")
    input_path = 'data/train_light.json'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please ensure AmbigQA train data is present.")
        return

    with open(input_path) as f:
        data = json.load(f)
    
    # Filter for truly ambiguous items
    ambiguous = []
    for item in data:
        for ann in item.get("annotations", []):
            if ann.get("type") == "multipleQAs" and len(ann.get("qaPairs", [])) >= 2:
                ambiguous.append(item)
                break
    
    target_count = 1000
    print(f"Found {len(ambiguous)} ambiguous queries. Distilling top {target_count} items...")
    
    # Take a buffer for failures
    sample = ambiguous[:int(target_count * 1.5)]
    
    os.makedirs('data/sft', exist_ok=True)
    train_path = 'data/sft/train.jsonl'
    valid_path = 'data/sft/valid.jsonl'
    
    train_count = 0
    valid_count = 0
    
    # Open files for incremental writing
    with open(train_path, 'w') as f_train, open(valid_path, 'w') as f_valid:
        # Parallel generation for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_gold_question, item) for item in sample]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Distilling Gold Qs"):
                res = future.result()
                if res:
                    # Probabilistic split for incremental writing
                    if random.random() < 0.9:
                        f_train.write(json.dumps(res) + '\n')
                        f_train.flush()
                        train_count += 1
                    else:
                        f_valid.write(json.dumps(res) + '\n')
                        f_valid.flush()
                        valid_count += 1
                        
                    if (train_count + valid_count) >= target_count:
                        # Attempt to cancel remaining
                        for f in futures: f.cancel()
                        break
    
    print(f"\nDistillation Complete!")
    print(f"Saved {train_count} train and {valid_count} validation items to data/sft/")
    print("\nNext step: Run the MLX fine-tuning command provided in the plan.")

if __name__ == "__main__":
    main()
