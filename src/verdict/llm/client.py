"""Thin wrapper over any OpenAI-compatible chat API with tool calling.

Works unchanged with Ollama, Groq, OpenRouter, vLLM, Together, OpenAI itself, or
Anthropic through a compatibility gateway. The rest of the codebase never imports
the provider SDK directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from verdict.config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


class LLM:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.think = settings.llm_think
        # max_retries=0: we do our own bounded retry below with a visible log line
        # per attempt. Stacking the SDK's internal retries on top of ours is how a
        # single slow/hung backend call turns into a silent multi-minute stall with
        # nothing printed until everything gives up.
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4), reraise=True)
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Ollama-specific: disable hidden reasoning on hybrid-thinking models
            # (qwen3, deepseek-r1, ...). Backends that don't recognize "think"
            # ignore it, so this is safe to send unconditionally.
            "extra_body": {"think": self.think},
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = resp.usage
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        raw = msg.model_dump(exclude_none=True)
        raw.pop("function_call", None)
        return ChatResponse(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=pt,
            completion_tokens=ct,
            raw_assistant_message=raw,
        )
