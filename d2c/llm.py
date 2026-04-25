"""Lightweight LLM client supporting Ollama and OpenAI-compatible APIs (e.g. vLLM)."""

import re
import time
import logging
from typing import Literal

import requests

logger = logging.getLogger(__name__)

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRAY_THINK_RE = re.compile(r"</?think>", re.DOTALL)

_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OPENAI_DEFAULT_URL = "http://localhost:8000"


class LLMClient:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str | None = None,
        think: bool | None = None,
        backend: Literal["ollama", "openai"] = "ollama",
    ):
        """Supports Ollama (/api/chat) and OpenAI-compatible backends like vLLM.

        backend: "ollama" (default) or "openai" for vLLM / any OpenAI-compatible server.
        base_url: defaults to localhost:11434 for ollama, localhost:8000 for openai.
        think: Ollama-only. If not None, passed as top-level `think` field to toggle
            the <think>...</think> preamble (e.g. think=False for Qwen3).
        """
        self.model = model
        self.backend = backend
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        elif backend == "openai":
            self.base_url = _OPENAI_DEFAULT_URL
        else:
            self.base_url = _OLLAMA_DEFAULT_URL
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
        """Single-turn call. Builds [system, user] internally."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat_with_messages(
            messages, temperature, max_tokens, strip_thinking, think, format_schema
        )

    def chat_with_messages(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        strip_thinking: bool = True,
        think: bool | None = None,
        format_schema: dict | None = None,
    ) -> str:
        """Multi-turn call with a full messages array (system/user/assistant turns)."""
        if self.backend == "openai":
            effective_think = think if think is not None else self.think
            return self._send_openai(
                messages, temperature, max_tokens, strip_thinking, format_schema, effective_think
            )
        return self._send_ollama(messages, temperature, max_tokens, strip_thinking, think, format_schema)

    def _send_ollama(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        strip_thinking: bool,
        think: bool | None,
        format_schema: dict | None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        effective_think = think if think is not None else self.think
        if effective_think is not None:
            payload["think"] = effective_think

        if format_schema is not None:
            payload["format"] = format_schema

        url = f"{self.base_url}/api/chat"
        return self._post(url, payload, lambda r: r["message"]["content"], strip_thinking)

    def _send_openai(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        strip_thinking: bool,
        format_schema: dict | None,
        think: bool | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if format_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        if think is False:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.base_url}/v1/chat/completions"
        return self._post(url, payload, lambda r: r["choices"][0]["message"]["content"], strip_thinking)

    def _post(self, url: str, payload: dict, extract: object, strip_thinking: bool) -> str:
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=300)
                resp.raise_for_status()
                content = extract(resp.json())
                if strip_thinking:
                    content = THINK_TAG_RE.sub("", content)
                    content = _STRAY_THINK_RE.sub("", content).strip()
                return content
            except (requests.RequestException, KeyError) as e:
                last_err = e
                if attempt == 0:
                    logger.warning("LLM call failed (attempt 1), retrying: %s", e)
                    time.sleep(1)

        raise RuntimeError(f"LLM call failed after 2 attempts: {last_err}")
