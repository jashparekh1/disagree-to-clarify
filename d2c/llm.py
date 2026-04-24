"""Lightweight LLM client wrapping Ollama's HTTP API."""

import re
import time
import logging

import requests

logger = logging.getLogger(__name__)

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMClient:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        think: bool | None = None,
    ):
        """Thin Ollama wrapper.

        think: if not None, passed to /api/chat as a top-level field. For
            Qwen3 and other thinking-toggle models, pass think=False to
            disable the <think>...</think> preamble entirely. None leaves
            the model's default behavior unchanged.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        strip_thinking: bool = True,
        think: bool | None = None,
        format_schema: dict | None = None,
    ) -> str:
        """Call Ollama's /api/chat endpoint. Return assistant message content.

        format_schema: if set, passed as Ollama's top-level `format` field to
            constrain generation to match the JSON schema (structured outputs).
            Eliminates prompt-based JSON begging.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        # Per-call think overrides the client default. Only sent when set.
        effective_think = think if think is not None else self.think
        if effective_think is not None:
            payload["think"] = effective_think

        if format_schema is not None:
            payload["format"] = format_schema

        url = f"{self.base_url}/api/chat"

        # Try up to 2 times (initial + 1 retry)
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=300)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                if strip_thinking:
                    content = THINK_TAG_RE.sub("", content).strip()
                return content
            except (requests.RequestException, KeyError) as e:
                last_err = e
                if attempt == 0:
                    logger.warning("LLM call failed (attempt 1), retrying: %s", e)
                    time.sleep(1)

        raise RuntimeError(f"LLM call failed after 2 attempts: {last_err}")
