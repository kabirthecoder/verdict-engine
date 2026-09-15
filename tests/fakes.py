"""A scripted LLM for tests: replays a list of ChatResponses in order."""

from __future__ import annotations

from typing import Any

from verdict.llm.client import LLM, ChatResponse, ToolCall


class FakeLLM(LLM):
    def __init__(self, script: list[ChatResponse]) -> None:  # noqa: D107 - no super()
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = "fake"

    def chat(self, messages, tools=None, tool_choice=None) -> ChatResponse:  # type: ignore[override]
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        return self.script.pop(0)


def calls(*specs: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = [ToolCall(id=f"call_{i}", name=n, arguments=a) for i, (n, a) in enumerate(specs)]
    return ChatResponse(
        content=None,
        tool_calls=tcs,
        raw_assistant_message={
            "role": "assistant",
            "tool_calls": [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.name, "arguments": "{}"},
                }
                for t in tcs
            ],
        },
    )


def text(content: str) -> ChatResponse:
    return ChatResponse(
        content=content, raw_assistant_message={"role": "assistant", "content": content}
    )
