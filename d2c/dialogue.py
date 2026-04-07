"""Multi-round dialogue loop between agents."""

from __future__ import annotations

import logging
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

    # --- Round 0 ---
    round_responses: list[AgentResponse] = []
    for agent in agents:
        logger.info("Round 0 — %s responding", agent.role.value)
        try:
            resp = agent.respond_initial(query)
        except Exception:
            logger.exception("Agent %s failed in round 0", agent.role.value)
            raise
        round_responses.append(resp)
    all_rounds.append(round_responses)

    # --- Rounds 1+ ---
    for round_num in range(1, num_rounds):
        prev_round = all_rounds[-1]
        round_responses = []
        for agent in agents:
            # Give each agent the OTHER agents' responses from the previous round
            other_responses = [r for r in prev_round if r.role != agent.role]
            logger.info("Round %d — %s responding", round_num, agent.role.value)
            try:
                resp = agent.respond_dialogue(query, other_responses, round_num)
            except Exception:
                logger.exception(
                    "Agent %s failed in round %d", agent.role.value, round_num
                )
                raise
            round_responses.append(resp)
        all_rounds.append(round_responses)

    return DialogueResult(
        query=query,
        rounds=all_rounds,
        num_rounds=num_rounds,
    )
