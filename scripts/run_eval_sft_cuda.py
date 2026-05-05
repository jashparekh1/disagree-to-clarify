"""Evaluation script for SFT-tuned model using Transformers (CUDA)."""

import argparse
import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.metrics import llm_judge_score
from d2c.llm import LLMClient

logger = logging.getLogger(__name__)

def parse_ambigqa_item(item: dict) -> tuple[str, list[str]]:
    query = item.get("question")
    annotations = item.get("annotations", [])
    interpretations = []
    for ann in annotations:
        if ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                if "question" in qa:
                    interpretations.append(qa["question"])
    return query, interpretations

def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT-tuned model on CUDA")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="adapters/", help="Path to model/adapter directory")
    parser.add_argument("--judge-model", default="qwen3:4b", help="Judge model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Loading model and tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    items = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Evaluating {len(items)} items...")
    judge_llm = LLMClient(model=args.judge_model)
    results = []

    for item in tqdm(items, desc="Evaluating SFT"):
        query, interpretations = parse_ambigqa_item(item)
        if not query or len(interpretations) < 2:
            continue

        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            eval_result = llm_judge_score(query, interpretations, generated_text, judge_llm)
            
            res = {
                "query": query,
                "generated_question": generated_text,
                "eval": eval_result,
                "baseline": "sft_cuda"
            }
            results.append(res)
        except Exception as e:
            logger.error(f"Failed on {query}: {e}")

    with open(args.output, "w") as out:
        for res in results:
            out.write(json.dumps(res) + "\n")

    if results:
        avg_score = sum(r["eval"]["score"] for r in results) / len(results)
        coverage = sum(1 for r in results if r["eval"]["covers_interpretations"]) / len(results) * 100
        print(f"\n[SFT-CUDA] Avg Score: {avg_score:.2f} | Coverage: {coverage:.1f}%")

if __name__ == "__main__":
    main()
