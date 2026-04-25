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

    @property
    def format_failure_rate(self) -> float:
        """Fraction of agent responses that failed JSON parsing even after
        retry. Reported as a first-class eval metric so we can distinguish
        capability signal from format-adherence signal when comparing models
        of different sizes.
        """
        total = sum(len(rnd) for rnd in self.rounds)
        if total == 0:
            return 0.0
        failures = sum(
            1 for rnd in self.rounds for resp in rnd if resp.format_failed
        )
        return failures / total

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rounds": [[r.to_dict() for r in rnd] for rnd in self.rounds],
            "num_rounds": self.num_rounds,
            "rounds_completed": self.rounds_completed,
            "converged": self.converged,
            "converged_at_round": self.converged_at_round,
            "format_failure_rate": self.format_failure_rate,
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

    conceded_roles: set = set()

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        # --- Round 0 ---
        logger.info("Starting Round 0")
        futures = [executor.submit(agent.respond_initial, query) for agent in agents]
        round_responses = [f.result() for f in futures]
        all_rounds.append(round_responses)

        # --- Rounds 1+ ---
        for round_num in range(1, num_rounds):
            active_agents = [a for a in agents if a.role not in conceded_roles]
            if len(active_agents) <= 1:
                logger.info("Only %d agent(s) remain — stopping early", len(active_agents))
                converged = True
                converged_at = round_num
                break

            logger.info("Starting Round %d (%d active agents)", round_num, len(active_agents))

            def _get_resp(agent: Agent, prior_rounds: list, r_num: int, c_roles: set):
                return agent.respond_dialogue(query, prior_rounds, r_num, c_roles)

            futures = [
                executor.submit(_get_resp, agent, list(all_rounds), round_num, set(conceded_roles))
                for agent in active_agents
            ]
            round_responses = [f.result() for f in futures]
            all_rounds.append(round_responses)

            for resp in round_responses:
                if resp.stance == Stance.CONCEDE:
                    conceded_roles.add(resp.role)
                    logger.info("%s concedes at round %d", resp.role.value, round_num)

            if all(r.stance == Stance.CONCEDE for r in round_responses):
                converged = True
                converged_at = round_num
                logger.info("Dialogue converged at round %d (all active agents CONCEDE)", round_num)
                break

    return DialogueResult(
        query=query,
        rounds=all_rounds,
        num_rounds=num_rounds,
        converged=converged,
        converged_at_round=converged_at,
    )
