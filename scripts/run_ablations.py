"""D2C Ablation Study Script.

This script runs a set of predefined ablations to evaluate the impact of:
1. Number of rounds (1 vs 3)
2. Role diversity (Heterogeneous vs Homogeneous)
3. The Concede mechanism (Normal vs No-Concede)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from d2c.agents import Agent, AgentRole, AgentResponse, Stance
from d2c.llm import LLMClient
from d2c.pipeline import run_d2c, D2CResult
from d2c.synthesizer import synthesize
from d2c.dialogue import DialogueResult, run_dialogue
from eval.judge import binary_judge
from d2c.prompts import VANILLA_CQG_SYSTEM, VANILLA_CQG_USER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Dialogue Runner for No-Concede Ablation ---

def run_dialogue_no_concede(
    query: str,
    agents: list[Agent],
    num_rounds: int = 3,
) -> DialogueResult:
    """Modified run_dialogue that ignores CONCEDE stances to force debate."""
    from concurrent.futures import ThreadPoolExecutor
    all_rounds: list[list[AgentResponse]] = []
    
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        # Round 0
        logger.info("Starting Round 0 (No-Concede)")
        futures = [executor.submit(agent.respond_initial, query) for agent in agents]
        all_rounds.append([f.result() for f in futures])

        # Rounds 1+
        for round_num in range(1, num_rounds):
            logger.info("Starting Round %d (No-Concede, 3 active agents)", round_num)
            def _get_resp(agent: Agent, prior_rounds: list, r_num: int):
                # We pass an empty set for conceded_roles to force everyone to stay
                return agent.respond_dialogue(query, prior_rounds, r_num, conceded_roles=set())

            futures = [executor.submit(_get_resp, agent, list(all_rounds), round_num) for agent in agents]
            round_responses = [f.result() for f in futures]
            # Override any CONCEDE to HOLD for the dialogue logic
            for r in round_responses:
                if r.stance == Stance.CONCEDE:
                    r.stance = Stance.HOLD
            all_rounds.append(round_responses)

    return DialogueResult(query=query, rounds=all_rounds, num_rounds=num_rounds)

# --- Ablation Runner ---

def run_ablation_config(
    query: str, 
    llm: LLMClient, 
    config_name: str, 
    num_rounds: int = 3
) -> str:
    if config_name == "vanilla":
        user_prompt = VANILLA_CQG_USER.format(query=query)
        raw = llm.chat(
            system_prompt=VANILLA_CQG_SYSTEM,
            user_prompt=user_prompt,
            format_schema={"type": "object", "properties": {"clarifying_question": {"type": "string"}}, "required": ["clarifying_question"]}
        )
        return json.loads(raw).get("clarifying_question", raw)

    # Configuration mapping
    if config_name == "standard":
        roles = [AgentRole.LOCUTIONARY, AgentRole.ILLOCUTIONARY, AgentRole.PERLOCUTIONARY]
        force_no_concede = False
    elif config_name == "homogeneous":
        # 3 agents with the same role
        roles = [AgentRole.PERLOCUTIONARY, AgentRole.PERLOCUTIONARY, AgentRole.PERLOCUTIONARY]
        force_no_concede = False
    elif config_name == "no_concede":
        roles = [AgentRole.LOCUTIONARY, AgentRole.ILLOCUTIONARY, AgentRole.PERLOCUTIONARY]
        force_no_concede = True
    else:
        raise ValueError(f"Unknown config: {config_name}")

    agents = [Agent(role, llm) for role in roles]
    
    if force_no_concede:
        dialogue = run_dialogue_no_concede(query, agents, num_rounds)
    else:
        dialogue = run_dialogue(query, agents, num_rounds)
        
    synth = synthesize(query, dialogue, llm)
    return synth.clarifying_question

def load_test_set(dataset: str, n: int) -> list[dict]:
    path = Path("test_sets") / f"{dataset}_test.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
            if len(records) >= n: break
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--dataset", default="clariq")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--backend", default="ollama")
    args = parser.parse_args()

    llm = LLMClient(model=args.model, backend=args.backend)
    judge_llm = LLMClient(model=args.judge_model, backend=args.backend, think=False)

    configs = [
        ("Vanilla", "vanilla", 0),
        ("D2C (Standard, 3 rounds)", "standard", 3),
        ("D2C (Standard, 1 round)", "standard", 1),
        ("D2C (Homogeneous, 3 rounds)", "homogeneous", 3),
        ("D2C (No-Concede, 3 rounds)", "no_concede", 3),
    ]

    items = load_test_set(args.dataset, args.n)
    results_table = []

    print(f"\nRunning Ablations on {args.dataset.upper()} (n={args.n})...\n")

    for label, config_id, rounds in configs:
        matches = 0
        print(f"Testing Config: {label}...")
        for item in items:
            query = item["query"]
            golds = item.get("gold_clarifying_questions", [])
            
            try:
                q = run_ablation_config(query, llm, config_id, num_rounds=rounds)
                judgment = binary_judge(query, q, golds, judge_llm)
                matches += judgment.match
            except Exception as e:
                logger.error(f"Error in {config_id}: {e}")
        
        score = (matches / args.n) * 100
        results_table.append({"Configuration": label, "Match@1 (%)": score})

    print("\n" + "="*50)
    print(f"ABLATION RESULTS - {args.dataset.upper()}")
    print("="*50)
    import pandas as pd
    print(pd.DataFrame(results_table).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
