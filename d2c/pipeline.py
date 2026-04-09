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
    model: str = "qwen2.5:0.5b",
    num_rounds: int = 3,
    max_tokens: int = 300,
) -> D2CResult:
    """Full pipeline: query -> agents -> dialogue -> synthesizer -> clarifying question."""
    llm = LLMClient(model=model)
    agents = [Agent(role, llm, max_tokens=max_tokens) for role in AgentRole]
    dialogue = run_dialogue(query, agents, num_rounds)
    synth = synthesize(query, dialogue, llm, max_tokens=max_tokens)
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
    model: str = "qwen2.5:0.5b",
    num_rounds: int = 3,
    resume: bool = False,
    max_workers: int = 4,
    max_tokens: int = 300,
) -> None:
    """Run D2C on a list of query dicts, save results as JSONL.

    Each item in *queries* must have a "query" key; other fields are passed through.
    """
    import concurrent.futures

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

    # Filter out already-processed queries
    to_process = [q for q in queries if q["query"] not in done_queries]
    if len(to_process) < len(queries):
        logger.info("Skipping %d already-processed queries", len(queries) - len(to_process))

    with open(output_path, "a") as out:

        def _worker(item: dict):
            query = item["query"]
            try:
                result = run_d2c(query, model=model, num_rounds=num_rounds, max_tokens=max_tokens)
                record = result.to_dict()
                for k, v in item.items():
                    if k != "query":
                        record[k] = v
                return record
            except Exception:
                logger.exception("Failed on query: %s", query)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, item) for item in to_process]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="D2C",
            ):
                record = future.result()
                if record:
                    out.write(json.dumps(record) + "\n")
                    out.flush()
