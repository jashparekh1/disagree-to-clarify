"""Multi-round interpretive loop over a single user turn.

Each "round" is not a debate move toward consensus; it is a re-reading of the
user's turn with the other agents' readings now visible. Divergence that
persists across rounds is the diagnostic the synthesizer uses to choose a
grounding move.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from d2c.agents import Agent, AgentResponse, Stance

logger = logging.getLogger(__name__)


@dataclass
class DialogueResult:
    query: str
    rounds: list[list[AgentResponse]]  # rounds[i] = list of agent responses
    num_rounds: int  # configured round budget (not necessarily the count completed)
    converged: bool = False
    converged_at_round: int | None = None

    @property
    def rounds_completed(self) -> int:
        return len(self.rounds)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rounds": [[r.to_dict() for r in rnd] for rnd in self.rounds],
            "num_rounds": self.num_rounds,
            "rounds_completed": self.rounds_completed,
            "converged": self.converged,
            "converged_at_round": self.converged_at_round,
        }


def run_dialogue(
    query: str,
    agents: list[Agent],
    num_rounds: int = 3,
) -> DialogueResult:
    """Run a multi-round interpretive loop.

    Round 0: each agent independently interprets the query.
    Rounds 1..num_rounds-1: each agent sees the other agents' previous
    responses and either HOLDs or CONCEDEs its reading.

    Early-stop: if every agent CONCEDEs in a given round, the dialogue has
    converged — no residual divergence remains, so running further rounds
    would only add noise.
    """
    all_rounds: list[list[AgentResponse]] = []
    converged = False
    converged_at: int | None = None

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

            if all(r.stance == Stance.CONCEDE for r in round_responses):
                converged = True
                converged_at = round_num
                logger.info(
                    "Dialogue converged at round %d (all agents CONCEDE)", round_num
                )
                break

    return DialogueResult(
        query=query,
        rounds=all_rounds,
        num_rounds=num_rounds,
        converged=converged,
        converged_at_round=converged_at,
    )
