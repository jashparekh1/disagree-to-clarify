"""Multi-round dialogue loop between agents."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from d2c.agents import Agent, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class DialogueResult:
    query: str
    rounds: list[list[AgentResponse]]  # rounds[i] = list of 3 agent responses
    num_rounds: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rounds": [[r.to_dict() for r in rnd] for rnd in self.rounds],
            "num_rounds": self.num_rounds,
        }


def run_dialogue(
    query: str,
    agents: list[Agent],
    num_rounds: int = 3,
) -> DialogueResult:
    """Run a multi-round dialogue.

    Round 0: each agent independently interprets the query.
    Rounds 1..num_rounds-1: each agent sees the other two agents' previous
    responses and updates its interpretation.
    """
    all_rounds: list[list[AgentResponse]] = []

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        # --- Round 0 ---
        logger.info("Starting Round 0")
        futures = [executor.submit(agent.respond_initial, query) for agent in agents]
        round_responses = [f.result() for f in futures]
        all_rounds.append(round_responses)

        # --- Rounds 1+ ---
        for round_num in range(1, num_rounds):
            prev_round = all_rounds[-1]
            logger.info("Starting Round %d", round_num)

            def _get_resp(agent: Agent, prev_round: list[AgentResponse], r_num: int):
                other_responses = [r for r in prev_round if r.role != agent.role]
                return agent.respond_dialogue(query, other_responses, r_num)

            futures = [
                executor.submit(_get_resp, agent, prev_round, round_num)
                for agent in agents
            ]
            round_responses = [f.result() for f in futures]
            all_rounds.append(round_responses)

    return DialogueResult(
        query=query,
        rounds=all_rounds,
        num_rounds=num_rounds,
    )
