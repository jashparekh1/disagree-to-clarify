import json

with open('data/dev_light.json') as f:
    data = json.load(f)

# Sample 5 items to test the eval script
with open('data/ambigqa_sample.jsonl', 'w') as out:
    for item in data[:5]:
        out.write(json.dumps(item) + '\n')

print(f"Sampled {len(data[:5])} items to data/ambigqa_sample.jsonl")
