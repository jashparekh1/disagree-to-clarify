"""Run evaluation on D2C outputs against any dataset.

Usage:
  python -m eval.run_eval --dataset clamber --input outputs/d2c_clamber.jsonl
  python -m eval.run_eval --dataset clamber --input outputs/d2c_clamber.jsonl --judge-model qwen3:8b
  python -m eval.run_eval --dataset all --input-dir outputs/ --skip-judge
  python -m eval.run_eval --dataset qulac --input outputs/d2c_qulac.jsonl --sample 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from d2c.llm import LLMClient
from eval.datasets import load_dataset
from eval.datasets.base import AmbiguousQuery
from eval.metrics import evaluate_all

logger = logging.getLogger(__name__)


def _load_d2c_outputs(path: str) -> dict[str, dict]:
    """Load D2C output JSONL, indexed by example_id."""
    outputs: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            eid = obj.get("example_id", obj.get("query", ""))
            outputs[eid] = obj
    return outputs


def _match_outputs(
    queries: list[AmbiguousQuery],
    outputs: dict[str, dict],
) -> tuple[list[AmbiguousQuery], list[str], list[bool]]:
    """Match dataset queries to D2C outputs by example_id (or query text fallback).

    Returns matched (queries, generated_questions, predicted_ambiguous).
    """
    matched_queries: list[AmbiguousQuery] = []
    generated: list[str] = []
    predicted: list[bool] = []

    # Build fallback index by query text
    by_query = {v.get("query", ""): v for v in outputs.values()}

    for q in queries:
        out = outputs.get(q.example_id) or by_query.get(q.query)
        if out is None:
            continue

        gen_q = out.get("generated_question", "")
        # Also try extracting from nested D2C result format
        if not gen_q:
            synth = out.get("synthesizer_result", {})
            gen_q = synth.get("clarifying_question", "")

        pred_amb = out.get("predicted_ambiguous", True)

        matched_queries.append(q)
        generated.append(gen_q)
        predicted.append(pred_amb)

    return matched_queries, generated, predicted


def _print_summary(results: dict) -> None:
    """Print evaluation results to stdout."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {results['dataset']}  ({results['n_examples']} examples)")
    print(f"{'='*60}")

    cn = results.get("clarification_need")
    if cn:
        print(f"\n  Clarification Need F1:")
        print(f"    Precision: {cn['precision']:.3f}")
        print(f"    Recall:    {cn['recall']:.3f}")
        print(f"    F1:        {cn['f1']:.3f}")
        print(f"    Accuracy:  {cn['accuracy']:.3f}")
    else:
        print(f"\n  Clarification Need F1: N/A (all queries ambiguous)")

    ss = results.get("semantic_similarity")
    if ss:
        print(f"\n  Semantic Similarity:")
        print(f"    Mean:   {ss['mean']:.3f}")
        print(f"    Median: {ss['median']:.3f}")
        print(f"    Std:    {ss['std']:.3f}")

    jq = results.get("judge_quality")
    if jq:
        print(f"\n  Judge Quality (1-5):")
        print(f"    Mean:   {jq['mean']:.2f}")
        print(f"    Median: {jq['median']:.1f}")
        print(f"    Std:    {jq['std']:.2f}")
        dist = jq.get("distribution", {})
        print(f"    Distribution: " + " ".join(f"{k}:{dist.get(k,0)}" for k in range(1, 6)))
    elif not results.get("_skip_judge"):
        print(f"\n  Judge Quality: not computed (use --judge-model to enable)")

    print()


def run_single_dataset(
    dataset_name: str,
    input_path: str,
    judge_model: str | None,
    output_path: str | None,
    skip_judge: bool,
    sample: int | None,
    seed: int,
) -> dict:
    """Run evaluation on a single dataset."""
    print(f"\nLoading dataset '{dataset_name}'...")
    queries = load_dataset(dataset_name)

    print(f"Loading D2C outputs from {input_path}...")
    outputs = _load_d2c_outputs(input_path)
    print(f"  {len(outputs)} outputs loaded")

    matched_queries, generated, predicted = _match_outputs(queries, outputs)
    print(f"  {len(matched_queries)} matched to dataset")

    if not matched_queries:
        print("ERROR: No matches found. Check example_id or query fields.")
        return {}

    # Sample if requested
    if sample and sample < len(matched_queries):
        random.seed(seed)
        indices = sorted(random.sample(range(len(matched_queries)), sample))
        matched_queries = [matched_queries[i] for i in indices]
        generated = [generated[i] for i in indices]
        predicted = [predicted[i] for i in indices]
        print(f"  Sampled {sample} examples (seed={seed})")

    # Set up judge LLM
    llm = None
    if not skip_judge and judge_model:
        llm = LLMClient(model=judge_model)

    results = evaluate_all(
        dataset_name=dataset_name,
        queries=matched_queries,
        generated_questions=generated,
        predicted_ambiguous=predicted,
        llm=llm,
        skip_judge=skip_judge,
    )
    results["_skip_judge"] = skip_judge

    _print_summary(results)

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable bits for saving
        save_results = {k: v for k, v in results.items() if not k.startswith("_")}
        save_results["judge_model"] = judge_model
        save_results["seed"] = seed
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D2C outputs")
    parser.add_argument(
        "--dataset", required=True,
        choices=["clamber", "qulac", "clariq", "all"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--input", help="D2C output JSONL file")
    parser.add_argument(
        "--input-dir", default="outputs",
        help="Directory with d2c_{dataset}.jsonl files (used with --dataset all)",
    )
    parser.add_argument("--judge-model", default="qwen3:8b", help="Model for LLM-as-judge")
    parser.add_argument("--output", help="Output results JSON path")
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for results (used with --dataset all)",
    )
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge metric")
    parser.add_argument("--sample", type=int, help="Random sample N examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.dataset == "all":
        for ds_name in ["clamber", "qulac", "clariq"]:
            input_path = f"{args.input_dir}/d2c_{ds_name}.jsonl"
            output_path = f"{args.output_dir}/{ds_name}_eval.json"
            if not Path(input_path).exists():
                print(f"\nSkipping {ds_name}: {input_path} not found")
                continue
            run_single_dataset(
                ds_name, input_path, args.judge_model,
                output_path, args.skip_judge, args.sample, args.seed,
            )
    else:
        input_path = args.input
        if not input_path:
            input_path = f"{args.input_dir}/d2c_{args.dataset}.jsonl"
        output_path = args.output or f"{args.output_dir}/{args.dataset}_eval.json"
        run_single_dataset(
            args.dataset, input_path, args.judge_model,
            output_path, args.skip_judge, args.sample, args.seed,
        )


if __name__ == "__main__":
    main()
