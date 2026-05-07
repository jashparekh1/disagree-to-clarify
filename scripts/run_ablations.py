"""Comprehensive Ablation Study for MADISSE.
Tests three key hypotheses:
1. Role Necessity: Does removing a specific agent (e.g. Fact-Finder) hurt F1 or Quality?
2. Taxonomy Value: Do specialized roles perform better than generic 'Assistant' roles?
3. Adversarial Pressure: Does 'forcing' a stance improve coverage over 'collaborative' consensus?
"""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path
from tqdm import tqdm

from d2c.llm import LLMClient
from d2c.pipeline import run_d2c
from d2c.agents import Agent, AgentRole
from d2c.dialogue import DialogueResult, run_dialogue
from d2c.synthesizer import synthesize, _format_transcript, SYNTHESIZER_SCHEMA
from d2c.prompts import SYNTHESIZER_USER
from eval.metrics import (
    clarification_need_f1,
    llm_judge_quality_multi_ref,
    semantic_similarity_multi_ref,
    internal_divergence_score,
)

logger = logging.getLogger(__name__)

DATASETS = ["clamber", "clariq", "qulac", "abgcoqa"]

# --- Generic prompts for Ablation 2 ---
GENERIC_AGENT_SYSTEM = "You are an AI assistant. Analyze the user's query and identify any potential ambiguities. If it is clear, say CLEAR. If not, explain one specific interpretation. 1-2 sentences only."

# --- Role-agnostic synthesizer for no_X_finder variants ---
# Does not reference specific role names so it works with any 2-agent subset.
GENERIC_SYNTHESIZER_SYSTEM = """You read multiple agents' interpretations of a user query.
If the agents disagree significantly, formulate one clarifying question that resolves the key point of disagreement. Under 20 words. No preamble.
If the agents agree the query is clear, output ONLY the word CLEAR.
Output only the question or CLEAR."""

# --- Generic forced prompts for Ablation 4 (forced stance, no role taxonomy) ---
# Replaces _STANCE_INSTRUCTIONS for forced-stance variants to prevent premature concession
_FORCED_STANCE_INSTRUCTIONS = """\
Decide your stance:
1. Has another agent explicitly and EXACTLY addressed your specific argument (not just a related topic)? → CONCEDE only then.
2. Otherwise → HOLD. Do not concede because of surface overlap or partial agreement.\
"""

# --- Non-forced prompts for Ablation 3 ---
COLLAB_FACT_SYSTEM = "You are a Fact-Finder. Your goal is to see if the query is clear. If you find real ambiguity, you may admit it, but your default lens is to look for a clear reading. 1-2 sentences only."
COLLAB_FACET_SYSTEM = "You are a Facet-Finder. Your goal is to see if subtopics are missing. If the query is already specific, you may admit it is clear. 1-2 sentences only."
COLLAB_INTENT_SYSTEM = "You are an Intent-Finder. Your goal is to see if the action is missing. If the intent is obvious, you may admit it is clear. 1-2 sentences only."


def load_full_test_set(dataset: str) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ablation_variant(query, variant_name, llm, num_rounds=3, context=None):
    """Runs a modified D2C pipeline. Returns (clarifying_question, div_score)."""
    if variant_name == "no_fact_finder":
        roles = [AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER]
    elif variant_name == "no_facet_finder":
        roles = [AgentRole.FACT_FINDER, AgentRole.INTENT_FINDER]
    elif variant_name == "no_intent_finder":
        roles = [AgentRole.FACT_FINDER, AgentRole.FACET_FINDER]
    elif variant_name == "generic_agents":
        agents = [Agent(r, llm) for r in (AgentRole.FACT_FINDER, AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER)]
        for a in agents:
            a.system_prompt = GENERIC_AGENT_SYSTEM
        dialogue = run_dialogue(query, agents, num_rounds=num_rounds, context=context)
        res = synthesize(query, dialogue, llm, variant="madisse")
        div = internal_divergence_score([r.interpretation for r in dialogue.rounds[-1]])
        return res.clarifying_question, div
    elif variant_name == "generic":
        agents = [Agent(r, llm) for r in (AgentRole.FACT_FINDER, AgentRole.FACET_FINDER, AgentRole.INTENT_FINDER)]
        for a in agents:
            a.system_prompt = GENERIC_AGENT_SYSTEM
            a.stance_instructions_override = _FORCED_STANCE_INSTRUCTIONS
        dialogue = run_dialogue(query, agents, num_rounds=num_rounds, context=context)
        res = synthesize(query, dialogue, llm, variant="madisse")
        div = internal_divergence_score([r.interpretation for r in dialogue.rounds[-1]])
        return res.clarifying_question, div
    elif variant_name == "collaborative":
        agents = [
            Agent(AgentRole.FACT_FINDER, llm),
            Agent(AgentRole.FACET_FINDER, llm),
            Agent(AgentRole.INTENT_FINDER, llm),
        ]
        agents[0].system_prompt = COLLAB_FACT_SYSTEM
        agents[1].system_prompt = COLLAB_FACET_SYSTEM
        agents[2].system_prompt = COLLAB_INTENT_SYSTEM
        for a in agents:
            a.stance_instructions_override = _FORCED_STANCE_INSTRUCTIONS
        dialogue = run_dialogue(query, agents, num_rounds=num_rounds, context=context)
        res = synthesize(query, dialogue, llm, variant="madisse")
        div = internal_divergence_score([r.interpretation for r in dialogue.rounds[-1]])
        return res.clarifying_question, div
    else:
        # full_madisse
        res = run_d2c(
            query, variant="madisse", model=llm.model, num_rounds=num_rounds,
            context=context, backend=llm.backend, base_url=llm.base_url,
            think=llm.think,
        )
        final_interps = [r.interpretation for r in res.dialogue.rounds[-1]]
        div = internal_divergence_score(final_interps)
        return res.synthesizer_result.clarifying_question, div

    # no_X_finder variants: reduced agent set, bypass gatekeeper entirely
    agents = [Agent(r, llm) for r in roles]
    dialogue = run_dialogue(query, agents, num_rounds=num_rounds, context=context)
    transcript = _format_transcript(dialogue)
    raw = llm.chat(
        system_prompt=GENERIC_SYNTHESIZER_SYSTEM,
        user_prompt=SYNTHESIZER_USER.format(query=query, transcript=transcript),
        temperature=0.3,
        max_tokens=100,
        format_schema=SYNTHESIZER_SCHEMA,
    )
    try:
        import json as _json
        q_text = _json.loads(raw).get("clarifying_question", raw).strip()
    except Exception:
        q_text = raw.strip()
    div = internal_divergence_score([r.interpretation for r in dialogue.rounds[-1]])
    return q_text, div


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-dataset", type=int, default=150)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=None, help="Run only these variants (e.g. --variants generic collaborative)")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    random.seed(args.seed)

    think = False if args.no_think else None
    llm = LLMClient(model=args.model, backend=args.backend, base_url=args.base_url, think=think)
    judge_llm = LLMClient(model=args.judge_model, backend=args.backend, base_url=args.base_url, think=think)

    results_file = Path("ablation_results.txt")
    inspection_file = Path("ablation_inspections.txt")

    with open(results_file, "w") as f:
        f.write(f"MADISSE ABLATION STUDY\n")
        f.write(f"Seed: {args.seed} | Model: {args.model} | Judge: {args.judge_model}\n")
        f.write("=" * 100 + "\n")
    with open(inspection_file, "w") as f:
        f.write(f"ABLATION INSPECTION LOG\nSeed: {args.seed}\n")
        f.write("=" * 100 + "\n")

    variants = [
        "full_madisse",
        "no_fact_finder",
        "no_facet_finder",
        "no_intent_finder",
        "generic",
        "collaborative",
    ]
    if args.variants:
        variants = [v for v in args.variants if v in variants]

    all_results = {v: {"preds": [], "golds": [], "scores": [], "sims": [], "covers": [], "divs": []} for v in variants}
    total_samples = 0

    for dataset in DATASETS:
        all_items = load_full_test_set(dataset)
        if not all_items:
            continue

        ambig = [x for x in all_items if x["is_ambiguous"]]
        non_ambig = [x for x in all_items if not x["is_ambiguous"]]
        random.shuffle(ambig)
        random.shuffle(non_ambig)
        n = args.n_per_dataset
        n_non = min(len(non_ambig), n // 2)
        n_amb = min(len(ambig), n - n_non)
        n_non = min(len(non_ambig), n - n_amb)
        items = ambig[:n_amb] + non_ambig[:n_non]
        random.shuffle(items)

        msg = f"\n>>> {dataset.upper()} ({len(items)} items, {n_amb} ambig / {n_non} non-ambig)"
        print(msg)
        with open(results_file, "a") as f:
            f.write(msg + "\n")

        for item in tqdm(items, desc=dataset):
            total_samples += 1
            query = item["query"]
            is_ambig = item["is_ambiguous"]
            context = item.get("context")
            gold_qs = item.get("gold_clarifying_questions", [])

            with open(inspection_file, "a") as f_insp:
                f_insp.write(f"\nQuery: {query}\n")

            for v in variants:
                try:
                    q_text, div_score = run_ablation_variant(query, v, llm, args.num_rounds, context)
                    pred_ambig = "CLEAR" not in q_text.upper()

                    all_results[v]["preds"].append(pred_ambig)
                    all_results[v]["golds"].append(is_ambig)
                    all_results[v]["divs"].append(div_score)

                    if is_ambig and gold_qs:
                        j_res = llm_judge_quality_multi_ref(query, q_text, gold_qs, judge_llm)
                        sim_score = semantic_similarity_multi_ref(q_text, gold_qs)
                        all_results[v]["scores"].append(j_res["score"])
                        all_results[v]["sims"].append(sim_score)
                        all_results[v]["covers"].append(1 if j_res.get("covers") else 0)

                    with open(inspection_file, "a") as f_insp:
                        f_insp.write(f"  [{v:<18}] -> {q_text[:100]}\n")

                except Exception as e:
                    print(f"Error in variant {v}: {e}")

            if total_samples % 10 == 0:
                header = f"{'Variant':<20} | {'F1':<5} | {'Qual':<5} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5}"
                lines = [f"\n--- RUNNING RESULTS (N={total_samples}) ---", header, "-" * len(header)]
                for v in variants:
                    if not all_results[v]["preds"]:
                        continue
                    det = clarification_need_f1(all_results[v]["preds"], all_results[v]["golds"])
                    q_mean = statistics.mean(all_results[v]["scores"]) if all_results[v]["scores"] else 0.0
                    div_mean = statistics.mean(all_results[v]["divs"]) if all_results[v]["divs"] else 0.0
                    sim_mean = statistics.mean(all_results[v]["sims"]) if all_results[v]["sims"] else 0.0
                    cov_mean = (sum(all_results[v]["covers"]) / len(all_results[v]["covers"]) * 100) if all_results[v]["covers"] else 0.0
                    lines.append(f"{v:<20} | {det['f1']:>5.2f} | {q_mean:>5.2f} | {div_mean:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f}")
                lines.append("-" * len(header))
                print("\n".join(lines))

    # Final summary table
    master_header = f"\n{'='*80}\n  FINAL ABLATION RESULTS (N={total_samples})\n{'='*80}"
    header_row = f"{'Variant':<20} | {'F1':<5} | {'Qual':<5} | {'Div':<4} | {'Sim':<6} | {'Cov%':<5}"

    print(master_header)
    print(header_row)
    print("-" * 80)

    with open(results_file, "a") as f:
        f.write(master_header + "\n")
        f.write(header_row + "\n")
        f.write("-" * 80 + "\n")

    for v in variants:
        if not all_results[v]["preds"]:
            msg = f"{v:<20} | No data"
            print(msg)
            with open(results_file, "a") as f:
                f.write(msg + "\n")
            continue
        det = clarification_need_f1(all_results[v]["preds"], all_results[v]["golds"])
        q_mean = statistics.mean(all_results[v]["scores"]) if all_results[v]["scores"] else 0.0
        div_mean = statistics.mean(all_results[v]["divs"]) if all_results[v]["divs"] else 0.0
        sim_mean = statistics.mean(all_results[v]["sims"]) if all_results[v]["sims"] else 0.0
        cov_mean = (sum(all_results[v]["covers"]) / len(all_results[v]["covers"]) * 100) if all_results[v]["covers"] else 0.0
        row = f"{v:<20} | {det['f1']:>5.2f} | {q_mean:>5.2f} | {div_mean:>4.2f} | {sim_mean:>6.3f} | {cov_mean:>5.1f}"
        print(row)
        with open(results_file, "a") as f:
            f.write(row + "\n")

    print("=" * 80)
    with open(results_file, "a") as f:
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    main()
