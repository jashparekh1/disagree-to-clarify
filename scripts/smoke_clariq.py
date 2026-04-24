"""Live smoke: run D2C against a handful of ClariQ dev queries and print
each agent's reading + the synthesizer's clarifying question next to the
gold question(s) so we can eyeball whether the three SAT lenses actually
produce distinguishable, sensible readings.

This is a qualitative check, not an eval. Use it to decide whether to
bother with the full eval sweep.

Usage:
    python -m scripts.smoke_clariq --n 3 --model qwen3:4b
    python -m scripts.smoke_clariq --n 3 --model qwen3:4b --debug   # dump raw LLM responses
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import OrderedDict
from pathlib import Path

from d2c.agents import _ROLE_DISPLAY
from d2c.pipeline import run_d2c


DEFAULT_TSV = Path("data/clariq/data/dev.tsv")


def load_clariq_topics(tsv_path: Path, n: int) -> list[dict]:
    """Group ClariQ dev rows by topic_id; return first N topics with their
    initial_request and the list of gold clarifying questions.
    """
    by_topic: OrderedDict[str, dict] = OrderedDict()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tid = row["topic_id"]
            if tid not in by_topic:
                by_topic[tid] = {
                    "topic_id": tid,
                    "initial_request": row["initial_request"],
                    "clarification_need": int(row.get("clarification_need", 0)),
                    "gold_questions": [],
                }
            q = (row.get("question") or "").strip()
            if q:
                by_topic[tid]["gold_questions"].append(q)
            if len(by_topic) >= n and tid not in list(by_topic.keys())[: n - 0]:
                break
    return list(by_topic.values())[:n]


def print_result(result, topic: dict) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"TOPIC {topic['topic_id']}  (clarification_need={topic['clarification_need']})")
    print(f"QUERY: {topic['initial_request']}")
    print(bar)

    # Round-by-round agent outputs.
    for round_idx, rnd in enumerate(result.dialogue.rounds):
        print(f"\n--- Round {round_idx} ---")
        for resp in rnd:
            flag = " [FORMAT FAIL]" if resp.format_failed else ""
            stance = f" [{resp.stance.value}]" if round_idx > 0 else ""
            print(f"\n  [{_ROLE_DISPLAY[resp.role]}]{stance}{flag}")
            print(f"    interpretation : {resp.interpretation}")
            if round_idx > 0 and resp.stance_reason:
                print(f"    stance_reason  : {resp.stance_reason}")

    # Synthesizer output.
    print(f"\n--- Synthesizer ---")
    sr = result.synthesizer_result
    sfail = " [FORMAT FAIL]" if sr.format_failed else ""
    print(f"  CLARIFYING QUESTION{sfail}: {sr.clarifying_question}")

    # Gold references.
    print(f"\n--- Gold clarifying questions (ClariQ, {len(topic['gold_questions'])}) ---")
    for gq in topic["gold_questions"][:5]:
        print(f"  - {gq}")
    if len(topic["gold_questions"]) > 5:
        print(f"  ... and {len(topic['gold_questions']) - 5} more")

    # Convergence summary.
    print(f"\n--- Meta ---")
    print(
        f"  rounds_completed={result.dialogue.rounds_completed}, "
        f"converged={result.dialogue.converged}, "
        f"converged_at_round={result.dialogue.converged_at_round}, "
        f"format_failure_rate={result.dialogue.format_failure_rate:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default=str(DEFAULT_TSV), help="Path to ClariQ dev TSV")
    parser.add_argument("--n", type=int, default=3, help="Number of topics to smoke-test")
    parser.add_argument("--model", default="qwen3:0.6b", help="Ollama model")
    parser.add_argument("--rounds", type=int, default=3, help="Dialogue rounds")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per call")
    parser.add_argument(
        "--variant", default="speech_act", choices=["original", "speech_act"]
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable Qwen3 thinking mode (default: off). Thinking eats the "
        "token budget and, with schema-constrained outputs, the reasoning "
        "block often collides with the JSON constraint — leave it off.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the raw LLM response before parsing, for diagnosing "
        "format failures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        raise SystemExit(
            f"ClariQ TSV not found at {tsv_path}. "
            "Clone it with: git clone --depth 1 "
            "https://github.com/aliannejadi/ClariQ.git data/clariq"
        )

    topics = load_clariq_topics(tsv_path, args.n)
    print(
        f"Loaded {len(topics)} ClariQ topics from {tsv_path}. "
        f"Running D2C (variant={args.variant}, model={args.model}, "
        f"think={'on' if args.think else 'OFF'}).\n"
    )

    if args.debug:
        _enable_raw_dump()

    for topic in topics:
        result = run_d2c(
            topic["initial_request"],
            model=args.model,
            num_rounds=args.rounds,
            max_tokens=args.max_tokens,
            variant=args.variant,
            think=True if args.think else False,
        )
        print_result(result, topic)


def _enable_raw_dump() -> None:
    """Monkey-patch LLMClient.chat to print the raw response before returning.

    Scoped to the smoke script so we can diagnose format failures without
    changing the production path.
    """
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
