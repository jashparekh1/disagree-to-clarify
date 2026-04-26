"""End-to-end eval: run D2C + vanilla-CQG on a test set, judge both with a
DIFFERENT LLM, report match-accuracy metrics.

Metrics reported per dataset:
  - D2C match rate (fraction of queries judged to match gold ambiguity)
  - Vanilla match rate
  - Delta (D2C - vanilla)
  - Per-category breakdown (CLAMBER only, on ambiguity_type)
  - Format failure rates
  - Convergence rate (D2C)

Saves per-query rows to outputs/eval_{dataset}_{model}.jsonl so predictions
+ judge verdicts can be inspected after the fact.

Usage:
  # Pull the judge first (one-time):
  ollama pull gemma:7b

  # Small smoke (5 queries) to verify everything wires up:
  python -m scripts.eval --dataset clamber --n 5

  # Full run on a dataset:
  python -m scripts.eval --dataset clamber

  # Different judge:
  python -m scripts.eval --dataset clamber --judge-model gemma2:9b
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from d2c.baseline import run_vanilla_cqg
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from eval.judge import binary_judge

logger = logging.getLogger(__name__)


def load_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No test set at {path}. Run `python -m scripts.build_test_sets`."
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _summary_line(label: str, matches: list[int]) -> str:
    n = len(matches)
    if n == 0:
        return f"  {label}: n=0"
    s = sum(matches)
    rate = s / n
    # Approximate Wald 95% CI.
    import math

    se = math.sqrt(rate * (1 - rate) / n) if n > 0 else 0.0
    ci = 1.96 * se
    return f"  {label}: {s}/{n} = {rate*100:.1f}% (±{ci*100:.1f}pp, 95% CI)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=["clariq", "qulac", "clamber"]
    )
    parser.add_argument("--n", type=int, default=None, help="Cap queries")
    parser.add_argument("--model", default="qwen3:4b", help="D2C + vanilla model")
    parser.add_argument(
        "--judge-model",
        default="gemma:7b",
        help="Judge LLM — must be DIFFERENT from --model to avoid self-judging bias.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--variant", default="speech_act", choices=["original", "speech_act"]
    )
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                        help="LLM backend: ollama (default) or openai (vLLM / any OpenAI-compatible server)")
    parser.add_argument("--base-url", default=None,
                        help="Override LLM server URL (default: localhost:11434 for ollama, localhost:8000 for openai)")
    parser.add_argument(
        "--out-dir", default="outputs", help="Where to write per-query rows."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip queries already in the output file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.model == args.judge_model:
        logger.warning(
            "--model and --judge-model are the same (%s). Judge-vs-system "
            "bias is a known issue; consider a different judge.",
            args.model,
        )

    test_set = load_test_set(args.dataset)
    if args.n is not None:
        test_set = test_set[: args.n]

    out_path = Path(args.out_dir) / f"eval_{args.dataset}_{args.model.replace(':','_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("example_id", ""))
                except Exception:
                    pass
        logger.info("Resume: skipping %d already-scored rows", len(done_ids))

    judge_llm = LLMClient(model=args.judge_model, think=False, base_url=args.base_url, backend=args.backend)

    print(
        f"Eval: dataset={args.dataset} n={len(test_set)} "
        f"model={args.model} judge={args.judge_model} "
        f"variant={args.variant} think={'on' if args.think else 'OFF'}"
    )
    print(f"Writing per-query rows to {out_path}")
    print()

    # We stream rows to disk as we go so a crash doesn't lose progress.
    f_out = open(out_path, "a", encoding="utf-8")

    d2c_matches: list[int] = []
    vanilla_matches: list[int] = []
    per_category: dict[str, list[tuple[int, int]]] = defaultdict(list)
    convergence = 0
    d2c_fail_rates: list[float] = []

    try:
        for idx, item in enumerate(test_set):
            if item.get("example_id") in done_ids:
                continue

            query = item["query"]
            golds = item.get("gold_clarifying_questions", [])
            category = item.get("ambiguity_type") or "uncategorized"

            # D2C prediction.
            d2c_res = run_d2c(
                query,
                model=args.model,
                num_rounds=args.rounds,
                max_tokens=args.max_tokens,
                variant=args.variant,
                think=True if args.think else False,
                base_url=args.base_url,
                backend=args.backend,
            )
            d2c_pred = d2c_res.synthesizer_result.clarifying_question
            d2c_fail = d2c_res.dialogue.format_failure_rate
            d2c_fail_rates.append(d2c_fail)
            if d2c_res.dialogue.converged:
                convergence += 1

            # Vanilla baseline.
            vanilla_llm = LLMClient(model=args.model, think=False, base_url=args.base_url, backend=args.backend)
            vanilla_res = run_vanilla_cqg(query, vanilla_llm, max_tokens=300)
            vanilla_pred = vanilla_res.clarifying_question

            # Judge both.
            d2c_judge = binary_judge(query, d2c_pred, golds, judge_llm)
            vanilla_judge = binary_judge(query, vanilla_pred, golds, judge_llm)

            d2c_matches.append(d2c_judge.match)
            vanilla_matches.append(vanilla_judge.match)
            per_category[category].append((d2c_judge.match, vanilla_judge.match))

            row = {
                "example_id": item.get("example_id"),
                "dataset": args.dataset,
                "query": query,
                "ambiguity_type": category,
                "gold_clarifying_questions": golds,
                "d2c_question": d2c_pred,
                "vanilla_question": vanilla_pred,
                "d2c_match": d2c_judge.match,
                "d2c_judge_reason": d2c_judge.reason,
                "vanilla_match": vanilla_judge.match,
                "vanilla_judge_reason": vanilla_judge.reason,
                "d2c_converged": d2c_res.dialogue.converged,
                "d2c_converged_at_round": d2c_res.dialogue.converged_at_round,
                "d2c_format_failure_rate": d2c_fail,
                "vanilla_format_failed": vanilla_res.format_failed,
                "model": args.model,
                "judge_model": args.judge_model,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            print(
                f"[{idx+1}/{len(test_set)}] d2c={d2c_judge.match} vanilla={vanilla_judge.match}  "
                f"q: {query[:60]!r}  d2c_pred: {d2c_pred[:80]!r}"
            )
    finally:
        f_out.close()

    # ---- Summary -------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"RESULTS  dataset={args.dataset}  n={len(d2c_matches)}  "
          f"model={args.model}  judge={args.judge_model}")
    print("=" * 72)
    print(_summary_line("D2C     match rate", d2c_matches))
    print(_summary_line("Vanilla match rate", vanilla_matches))
    if d2c_matches and vanilla_matches:
        delta = (sum(d2c_matches) / len(d2c_matches)) - (
            sum(vanilla_matches) / len(vanilla_matches)
        )
        print(f"  Delta (D2C - Vanilla): {delta*100:+.1f}pp")

        # 2x2 matrix
        both = sum(1 for a, b in zip(d2c_matches, vanilla_matches) if a and b)
        d2c_only = sum(1 for a, b in zip(d2c_matches, vanilla_matches) if a and not b)
        van_only = sum(1 for a, b in zip(d2c_matches, vanilla_matches) if not a and b)
        neither = sum(1 for a, b in zip(d2c_matches, vanilla_matches) if not a and not b)
        print(f"  both hit:    {both}")
        print(f"  D2C only:    {d2c_only}")
        print(f"  vanilla only: {van_only}")
        print(f"  neither:     {neither}")

    if per_category and args.dataset == "clamber":
        print()
        print("  Per-category (CLAMBER):")
        for cat in sorted(per_category):
            pairs = per_category[cat]
            n = len(pairs)
            d2c_rate = sum(p[0] for p in pairs) / n
            van_rate = sum(p[1] for p in pairs) / n
            print(
                f"    {cat:40s} n={n:3d}  d2c={d2c_rate*100:5.1f}%  "
                f"vanilla={van_rate*100:5.1f}%  Δ={((d2c_rate-van_rate)*100):+.1f}pp"
            )

    # Diagnostics.
    print()
    n_d2c = len(d2c_matches) or 1
    print(f"  D2C convergence rate: {convergence}/{n_d2c} = {convergence/n_d2c*100:.1f}%")
    if d2c_fail_rates:
        avg_fail = sum(d2c_fail_rates) / len(d2c_fail_rates)
        print(f"  D2C avg format-failure rate: {avg_fail*100:.1f}%")

    print()
    print(f"Per-query rows saved: {out_path}")


if __name__ == "__main__":
    main()
