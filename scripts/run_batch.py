"""Batch run D2C on a dataset file.

Usage:
    python -m scripts.run_batch --input data/queries.jsonl --output outputs/d2c_results.jsonl
    python -m scripts.run_batch --input data/queries.jsonl --output outputs/d2c_results.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import logging

from d2c.pipeline import run_d2c_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D2C on a batch of queries")
    parser.add_argument("--input", required=True, help="Input JSONL file (each line: {\"query\": \"...\"})")
    parser.add_argument("--output", required=True, help="Output JSONL file for results")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--rounds", type=int, default=3, help="Number of dialogue rounds")
    parser.add_argument("--resume", action="store_true", help="Skip queries already in the output file")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel query workers")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per LLM call")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                        help="LLM backend: ollama (default) or openai (vLLM / any OpenAI-compatible server)")
    parser.add_argument("--base-url", default=None,
                        help="Override LLM server URL (default: localhost:11434 for ollama, localhost:8000 for openai)")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load queries
    queries: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries from {args.input}")
    print(f"Model: {args.model} | Rounds: {args.rounds} | Backend: {args.backend} | Resume: {args.resume}")

    run_d2c_batch(
        queries=queries,
        output_path=args.output,
        model=args.model,
        num_rounds=args.rounds,
        resume=args.resume,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        backend=args.backend,
        base_url=args.base_url,
        think=args.think,
    )

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
