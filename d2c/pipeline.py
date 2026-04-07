"""End-to-end D2C pipeline: query in -> clarifying question out."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from tqdm import tqdm

from d2c.agents import Agent, AgentRole
from d2c.dialogue import DialogueResult, run_dialogue
from d2c.llm import LLMClient
from d2c.synthesizer import SynthesizerResult, synthesize

logger = logging.getLogger(__name__)


@dataclass
class D2CResult:
    query: str
    dialogue: DialogueResult
    synthesizer_result: SynthesizerResult
    model: str
    num_rounds: int
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "dialogue": self.dialogue.to_dict(),
            "synthesizer_result": self.synthesizer_result.to_dict(),
            "model": self.model,
            "num_rounds": self.num_rounds,
            "timestamp": self.timestamp,
        }


def run_d2c(
    query: str,
    model: str = "qwen3:4b",
    num_rounds: int = 3,
) -> D2CResult:
    """Full pipeline: query -> agents -> dialogue -> synthesizer -> clarifying question."""
    llm = LLMClient(model=model)
    agents = [Agent(role, llm) for role in AgentRole]
    dialogue = run_dialogue(query, agents, num_rounds)
    synth = synthesize(query, dialogue, llm)
    return D2CResult(
        query=query,
        dialogue=dialogue,
        synthesizer_result=synth,
        model=model,
        num_rounds=num_rounds,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_d2c_batch(
    queries: list[dict],
    output_path: str,
    model: str = "qwen3:4b",
    num_rounds: int = 3,
    resume: bool = False,
) -> None:
    """Run D2C on a list of query dicts, save results as JSONL.

    Each item in *queries* must have a "query" key; other fields are passed through.
    """
    # If resuming, load already-processed queries
    done_queries: set[str] = set()
    if resume:
        try:
            with open(output_path) as f:
                for line in f:
                    obj = json.loads(line)
                    done_queries.add(obj.get("query", ""))
        except FileNotFoundError:
            pass

    with open(output_path, "a") as out:
        for item in tqdm(queries, desc="D2C"):
            query = item["query"]
            if query in done_queries:
                logger.info("Skipping already-processed query: %s", query)
                continue

            try:
                result = run_d2c(query, model=model, num_rounds=num_rounds)
                record = result.to_dict()
                # Pass through any extra fields from the input
                for k, v in item.items():
                    if k != "query":
                        record[k] = v
                out.write(json.dumps(record) + "\n")
                out.flush()
            except Exception:
                logger.exception("Failed on query: %s", query)
