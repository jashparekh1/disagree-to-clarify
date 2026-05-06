import json
import os
import random
import concurrent.futures
from tqdm import tqdm
from d2c.llm import LLMClient
from d2c.prompts import (
    VANILLA_CQG_SYSTEM, VANILLA_CQG_USER,
    SIMULATED_USER_SYSTEM, SIMULATED_USER_USER,
    RESOLUTION_JUDGE_SYSTEM, RESOLUTION_JUDGE_USER
)

# Use the vLLM endpoint for simulation and judging
TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct" 
llm_teacher = LLMClient(
    model=TEACHER_MODEL, 
    backend="openai", 
    base_url="http://localhost:8001"
)

def extract_json(text):
    """Robust JSON extraction from a string."""
    try:
        # Try finding the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        return json.loads(text)
    except:
        return None

CQG_SCHEMA = {
    "type": "object",
    "properties": {
        "clarifying_question": {"type": "string"}
    },
    "required": ["clarifying_question"]
}

RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "resolution_score": {"type": "number"},
        "reasoning": {"type": "string"}
    },
    "required": ["resolution_score", "reasoning"]
}

# Use a better prompt for candidate generation (don't use the 'strict' vanilla prompt here)
DPO_CANDIDATE_SYSTEM = "You are an expert at identifying ambiguity. Generate a concise clarifying question for the given query."
DPO_CANDIDATE_USER = "Query: {query}\n\nOutput ONLY this JSON: {{\"clarifying_question\": \"<your question>\"}}"

def generate_candidate(query, temp):
    """Generate a single candidate clarifying question."""
    try:
        raw = llm_teacher.chat(
            system_prompt=DPO_CANDIDATE_SYSTEM,
            user_prompt=DPO_CANDIDATE_USER.format(query=query),
            temperature=temp,
            max_tokens=100,
            format_schema=CQG_SCHEMA
        )
        data = extract_json(raw)
        if not data:
             return None
        q = data.get("clarifying_question", "CLEAR")
        return q if q != "CLEAR" else None
    except Exception as e:
        print(f"Error in generate_candidate: {e}")
        return None

def score_trajectory(query, question, intent, all_intents):
    """Simulate a turn and return the resolution score."""
    try:
        # 1. Simulate Answer
        answer = llm_teacher.chat(
            system_prompt=SIMULATED_USER_SYSTEM,
            user_prompt=SIMULATED_USER_USER.format(
                query=query, 
                interpretation=intent,
                clarifying_question=question
            ),
            temperature=0.1
        )

        # 2. Judge Resolution
        formatted_intents = "\n".join(f"- {i}" for i in all_intents)
        raw_eval = llm_teacher.chat(
            system_prompt=RESOLUTION_JUDGE_SYSTEM,
            user_prompt=RESOLUTION_JUDGE_USER.format(
                query=query,
                clarifying_question=question,
                user_answer=answer,
                all_interpretations=formatted_intents
            ),
            temperature=0.0,
            format_schema=RESOLUTION_SCHEMA
        )

        eval_data = extract_json(raw_eval)
        if not eval_data:
            return 1.0
        return float(eval_data.get("resolution_score", 1.0))
    except Exception as e:
        print(f"Error in score_trajectory: {e}")
        return 1.0

def process_item_dpo(item):
    query = item.get("question")
    interpretations = []
    for ann in item.get("annotations", []):
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    
    # CASE 1: UNAMBIGUOUS QUERY (Preference for staying quiet)
    if len(interpretations) < 2:
        # Generate a "bad" question to reject
        rejected = generate_candidate(query, 0.8)
        if not rejected: return None
        return {
            "prompt": query,
            "chosen": "CLEAR",
            "rejected": rejected
        }

    # CASE 2: AMBIGUOUS QUERY (Preference for better resolution)
    # 1. Generate two candidates
    q1 = generate_candidate(query, 0.7)
    q2 = generate_candidate(query, 0.9)
    
    if not q1 or not q2 or q1 == q2:
        return None

    # 2. Score both against a random target intent
    target_intent = random.choice(interpretations)
    score1 = score_trajectory(query, q1, target_intent, interpretations)
    score2 = score_trajectory(query, q2, target_intent, interpretations)

    if score1 == score2:
        return None # No clear preference

    chosen, rejected = (q1, q2) if score1 > score2 else (q2, q1)
    
    # MLX DPO format
    return {
        "prompt": query,
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    print("Starting DPO Dataset Preparation (Balanced DPO)...")
    input_path = 'data/train_light.json'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path) as f:
        data = json.load(f)
    
    # Split items
    ambiguous = []
    clear = []
    for item in data:
        is_ambig = False
        for ann in item.get("annotations", []):
            if ann.get("type") == "multipleQAs" and len(ann.get("qaPairs", [])) >= 2:
                is_ambig = True
                break
        if is_ambig:
            ambiguous.append(item)
        else:
            clear.append(item)
    
    # Targeted mix for DPO
    target_ambig = 150
    target_clear = 50
    
    sample = ambiguous[:target_ambig] + clear[:target_clear]
    random.shuffle(sample)
    
    print(f"Sampling {len(sample)} items ({target_ambig} ambig, {target_clear} clear)...")
    
    os.makedirs('data/dpo', exist_ok=True)
    train_path = 'data/dpo/train.jsonl'
    valid_path = 'data/dpo/valid.jsonl'
    
    train_count = 0
    valid_count = 0
    target_count = len(sample)
    
    with open(train_path, 'w') as f_train, open(valid_path, 'w') as f_valid:
        # Increased max_workers to 4 for M1 Max
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_item_dpo, item) for item in sample]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating DPO Pairs"):
                res = future.result()
                if res:
                    if random.random() < 0.9:
                        f_train.write(json.dumps(res) + '\n')
                        f_train.flush()
                        train_count += 1
                    else:
                        f_valid.write(json.dumps(res) + '\n')
                        f_valid.flush()
                        valid_count += 1
                        
                    if (train_count + valid_count) >= target_count:
                        # Cancel remaining
                        for f in futures: f.cancel()
                        break
    
    print(f"\nDPO Dataset Complete! Saved {train_count} train and {valid_count} validation pairs.")
    
    print(f"\nDPO Dataset Complete! Saved {train_count} train and {valid_count} validation pairs.")

if __name__ == "__main__":
    main()
