import json

with open('data/dev_light.json') as f:
    data = json.load(f)

ambiguous_items = []
for item in data:
    interpretations = []
    annotations = item.get("annotations", [])
    for ann in annotations:
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    
    if len(interpretations) >= 2:
        ambiguous_items.append(item)
    
    if len(ambiguous_items) >= 100:
        break

with open('data/ambigqa_100.jsonl', 'w') as out:
    for item in ambiguous_items:
        out.write(json.dumps(item) + '\n')

print(f"Sampled {len(ambiguous_items)} ambiguous items to data/ambigqa_filtered.jsonl")
