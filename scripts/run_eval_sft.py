"""Evaluation script for the SFT-tuned MLX model."""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import mlx.core as mx
from mlx_lm import load, generate
from eval.metrics import llm_judge_score
from d2c.llm import LLMClient

logger = logging.getLogger(__name__)

def parse_ambigqa_item(item: dict) -> tuple[str, list[str]]:
    """Reuse the AmbigQA light parser."""
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
    parser = argparse.ArgumentParser(description="Evaluate SFT-tuned MLX model")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit", help="Base model")
    parser.add_argument("--adapter", default="adapters/", help="Path to adapter directory")
    parser.add_argument("--judge-model", default="qwen3:4b", help="Judge model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Loading model {args.model} with adapter {args.adapter}...")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

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

        # Generate using the SFT model
        # Replicate the template used in training
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            generated_text = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=150,
                verbose=False
            )
            # Remove potential end-of-text markers if any
            generated_text = generated_text.split("<|im_end|>")[0].strip()

            # Judge
            eval_result = llm_judge_score(query, interpretations, generated_text, judge_llm)
            
            res = {
                "query": query,
                "generated_question": generated_text,
                "eval": eval_result,
                "baseline": "sft_mlx"
            }
            results.append(res)
        except Exception as e:
            logger.error(f"Failed on {query}: {e}")

    with open(args.output, "w") as out:
        for res in results:
            out.write(json.dumps(res) + "\n")

    # Aggregate
    if results:
        avg_score = sum(r["eval"]["score"] for r in results) / len(results)
        coverage = sum(1 for r in results if r["eval"]["covers_interpretations"]) / len(results) * 100
        print(f"\n[SFT-MLX] Avg Score: {avg_score:.2f} | Coverage: {coverage:.1f}%")

if __name__ == "__main__":
    main()
