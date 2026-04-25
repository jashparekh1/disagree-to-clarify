"""Run D2C on a sample from any of the three datasets (ClariQ / Qulac /
CLAMBER) and print each agent's reading + the synthesizer's clarifying
question next to the gold question for eyeball comparison.

This is a qualitative smoke, not an eval. Use it to decide whether the
system is producing coherent clarifications across datasets before
running the full eval sweep.

Usage:
    python -m scripts.smoke --dataset clariq --n 5 --model qwen3:4b
    python -m scripts.smoke --dataset clamber --n 5 --model qwen3:4b
    python -m scripts.smoke --dataset qulac   --n 5 --model qwen3:4b
    python -m scripts.smoke --dataset clariq --n 5 --model qwen3:4b --debug
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict

from d2c.agents import _ROLE_DISPLAY
from d2c.pipeline import run_d2c
from eval.datasets import load_dataset
from eval.datasets.base import AmbiguousQuery


def sample_queries(
    dataset_name: str, n: int, seed: int
) -> list[tuple[AmbiguousQuery, list[str]]]:
    """Sample N ambiguous queries from the dataset.

    For Qulac/ClariQ, items are grouped by topic_id (one row per facet-
    question); we deduplicate by topic and collect all gold questions per
    topic. For CLAMBER, each item stands alone.

    Returns a list of (example, gold_questions_for_example) tuples.
    """
    items = load_dataset(dataset_name)
    ambiguous = [q for q in items if q.is_ambiguous]

    if dataset_name in ("clariq", "qulac"):
        by_topic: dict[str, list[AmbiguousQuery]] = defaultdict(list)
        for q in ambiguous:
            by_topic[q.topic_id or q.query].append(q)
        topic_ids = list(by_topic.keys())
        random.Random(seed).shuffle(topic_ids)
        sampled = []
        for tid in topic_ids[:n]:
            entries = by_topic[tid]
            # First entry represents the topic; collect gold questions from all.
            golds = [
                e.gold_clarifying_question
                for e in entries
                if e.gold_clarifying_question
            ]
            sampled.append((entries[0], golds))
        return sampled

    # CLAMBER (one gold per item).
    picked = random.Random(seed).sample(ambiguous, min(n, len(ambiguous)))
    return [(q, [q.gold_clarifying_question]) for q in picked]


def print_result(result, item: AmbiguousQuery, golds: list[str]) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    header = f"[{item.dataset}]"
    if item.topic_id:
        header += f" topic {item.topic_id}"
    if item.example_id:
        header += f" id {item.example_id}"
    if item.ambiguity_type:
        header += f" ({item.ambiguity_type})"
    print(header)
    print(f"QUERY: {item.query}")
    print(bar)

    for round_idx, rnd in enumerate(result.dialogue.rounds):
        print(f"\n--- Round {round_idx} ---")
        for resp in rnd:
            flag = " [FORMAT FAIL]" if resp.format_failed else ""
            stance = f" [{resp.stance.value}]" if round_idx > 0 else ""
            print(f"\n  [{_ROLE_DISPLAY[resp.role]}]{stance}{flag}")
            interp = (resp.interpretation or "").strip()
            # Keep printout readable — truncate runaway essays.
            if len(interp) > 400:
                interp = interp[:400] + f" ... [truncated, full len={len(resp.interpretation)}]"
            print(f"    interpretation : {interp}")
            if round_idx > 0 and resp.stance_reason:
                reason = resp.stance_reason.strip()
                if len(reason) > 300:
                    reason = reason[:300] + " ..."
                print(f"    stance_reason  : {reason}")

    sr = result.synthesizer_result
    sfail = " [FORMAT FAIL]" if sr.format_failed else ""
    print(f"\n--- Synthesizer ---")
    print(f"  CLARIFYING QUESTION{sfail}: {sr.clarifying_question}")

    print(f"\n--- Gold clarifying questions ({len(golds)}) ---")
    for gq in golds[:5]:
        print(f"  - {gq}")
    if len(golds) > 5:
        print(f"  ... and {len(golds) - 5} more")

    print(f"\n--- Meta ---")
    print(
        f"  rounds_completed={result.dialogue.rounds_completed}, "
        f"converged={result.dialogue.converged}, "
        f"converged_at_round={result.dialogue.converged_at_round}, "
        f"format_failure_rate={result.dialogue.format_failure_rate:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["clariq", "qulac", "clamber"],
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant", default="speech_act", choices=["original", "speech_act"]
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable Qwen3 thinking mode (default OFF — thinking conflicts "
        "with schema-constrained decoding and breaks format).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump raw LLM responses before parsing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.debug:
        _enable_raw_dump()

    sampled = sample_queries(args.dataset, args.n, args.seed)
    print(
        f"Loaded {len(sampled)} items from {args.dataset}. "
        f"Running D2C (variant={args.variant}, model={args.model}, "
        f"think={'on' if args.think else 'OFF'}, seed={args.seed}).\n"
    )

    summary = {
        "items": len(sampled),
        "format_fail_sum": 0.0,
        "converged": 0,
    }
    for item, golds in sampled:
        result = run_d2c(
            item.query,
            model=args.model,
            num_rounds=args.rounds,
            max_tokens=args.max_tokens,
            variant=args.variant,
            think=True if args.think else False,
        )
        print_result(result, item, golds)
        summary["format_fail_sum"] += result.dialogue.format_failure_rate
        if result.dialogue.converged:
            summary["converged"] += 1

    print("\n" + "=" * 72)
    print(
        f"SMOKE SUMMARY  dataset={args.dataset}  n={summary['items']}  "
        f"avg_format_fail_rate={summary['format_fail_sum']/max(1,summary['items']):.2f}  "
        f"converged={summary['converged']}/{summary['items']}"
    )
    print("=" * 72)


def _enable_raw_dump() -> None:
    from d2c import llm as _llm_mod

    _orig = _llm_mod.LLMClient.chat

    def _dumped(self, *args, **kwargs):
        out = _orig(self, *args, **kwargs)
        snippet = (out or "").strip().replace("\n", "\\n")
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        print(f"  [raw response, len={len(out or '')}] {snippet}")
        return out

    _llm_mod.LLMClient.chat = _dumped


if __name__ == "__main__":
    main()
