"""Run D2C on N examples from each test set, print full traces, and judge.

Usage:
    python -m scripts.smoke_testsets
    python -m scripts.smoke_testsets --n 3 --model qwen3:4b --no-think
    python -m scripts.smoke_testsets --backend openai --model Qwen/Qwen3-4B --no-think
    python -m scripts.smoke_testsets --dataset qulac --n 10
    python -m scripts.smoke_testsets --no-judge   # skip LLM judge, just print traces
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from d2c.agents import _ROLE_DISPLAY
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.judge import JudgeResult, binary_judge

logger = logging.getLogger(__name__)

DATASETS = ["clariq", "qulac", "clamber"]


def load_test_set(dataset: str, n: int) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No test set at {path}. Run `python -m scripts.build_test_sets`.")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


def print_trace(dataset: str, idx: int, item: dict, result, judge_result: JudgeResult | None) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  [{dataset.upper()}] EXAMPLE {idx + 1}")
    print(bar)
    print(f"QUERY: \"{item['query']}\"")

    golds = item.get("gold_clarifying_questions", [])
    if golds:
        print(f"\nGOLD CLARIFYING QUESTIONS ({len(golds)}):")
        for g in golds[:8]:
            print(f"  - {g}")
        if len(golds) > 8:
            print(f"  ... and {len(golds) - 8} more")

    for round_idx, rnd in enumerate(result.dialogue.rounds):
        print(f"\n--- Round {round_idx} ---")
        for resp in rnd:
            flag = " [FORMAT FAIL]" if resp.format_failed else ""
            stance = f" [{resp.stance.value}]" if round_idx > 0 else ""
            if resp.stance.value == "CONCEDE" and round_idx > 0:
                stance = " [CONCEDE — dropped out]"
            print(f"  [{_ROLE_DISPLAY[resp.role]}]{stance}{flag}")
            print(f"    {resp.interpretation}")
            if round_idx > 0 and resp.stance_reason:
                print(f"    Reason: {resp.stance_reason}")

    sr = result.synthesizer_result
    sfail = " [FORMAT FAIL]" if sr.format_failed else ""
    print(f"\n--- Synthesizer ---")
    print(f"  CLARIFYING QUESTION{sfail}: {sr.clarifying_question}")

    if judge_result is not None:
        verdict = "MATCH" if judge_result.match else "FAIL"
        print(f"\n--- Judge ---")
        print(f"  Result:  {verdict}")
        print(f"  Reason:  {judge_result.reason}")

    print(f"\n--- Meta ---")
    d = result.dialogue
    print(
        f"  converged={d.converged}  "
        f"converged_at={d.converged_at_round}  "
        f"format_failure_rate={d.format_failure_rate:.2f}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Examples per dataset")
    parser.add_argument(
        "--dataset", choices=DATASETS + ["all"], default="all",
        help="Which dataset(s) to sample from (default: all)",
    )
    parser.add_argument("--model", default="qwen3:4b", help="Model name")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model (default: same as --model)")
    parser.add_argument("--rounds", type=int, default=3, help="Dialogue rounds")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per LLM call")
    parser.add_argument(
        "--variant", default="speech_act", choices=["original", "speech_act"],
    )
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge scoring")
    parser.add_argument(
        "--backend", default="ollama", choices=["ollama", "openai"],
        help="LLM backend",
    )
    parser.add_argument("--base-url", default=None, help="Override LLM server URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    judge_model = args.judge_model or args.model
    judge_llm = None if args.no_judge else LLMClient(
        model=judge_model,
        base_url=args.base_url,
        backend=args.backend,
        think=False,
    )

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    print(f"\nD2C SMOKE — {args.n} examples × {len(datasets)} dataset(s)")
    print(f"model={args.model}  judge={judge_model if not args.no_judge else 'OFF'}  "
          f"backend={args.backend}  variant={args.variant}  "
          f"rounds={args.rounds}  think={'OFF' if args.no_think else 'on'}\n")

    totals: dict[str, list[int]] = {}

    for dataset in datasets:
        try:
            items = load_test_set(dataset, args.n)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        print(f"\n{'#' * 72}")
        print(f"  DATASET: {dataset.upper()}  ({len(items)} items)")
        print(f"{'#' * 72}")

        matches: list[int] = []

        for idx, item in enumerate(items):
            try:
                result = run_d2c(
                    item["query"],
                    model=args.model,
                    num_rounds=args.rounds,
                    max_tokens=args.max_tokens,
                    variant=args.variant,
                    think=False if args.no_think else None,
                    base_url=args.base_url,
                    backend=args.backend,
                )

                judge_result = None
                if judge_llm is not None:
                    golds = item.get("gold_clarifying_questions", [])
                    pred = result.synthesizer_result.clarifying_question
                    judge_result = binary_judge(item["query"], pred, golds, judge_llm)
                    matches.append(judge_result.match)

                print_trace(dataset, idx, item, result, judge_result)
            except Exception:
                logger.exception("Failed on %s example %d: %s", dataset, idx + 1, item.get("query"))

        if matches:
            rate = sum(matches) / len(matches)
            print(f"  >> {dataset.upper()} match rate: {sum(matches)}/{len(matches)} = {rate*100:.0f}%\n")
            totals[dataset] = matches

    if len(totals) > 1:
        all_matches = [m for ms in totals.values() for m in ms]
        rate = sum(all_matches) / len(all_matches)
        print(f"\n{'=' * 72}")
        print(f"  OVERALL: {sum(all_matches)}/{len(all_matches)} = {rate*100:.0f}%")
        print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
