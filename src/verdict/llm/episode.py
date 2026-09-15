"""One bounded tool-calling episode for an agent.

The loop: send system + task, let the model call tools (several per turn, executed
concurrently), feed the results back with their evidence ids, repeat until the model
calls `submit` with a payload that validates against the role's output schema.

Why a `submit` tool instead of JSON mode: every OpenAI-compatible backend that does
tool calling supports it, including small open models, and validation failures can
be sent back as a tool error the model can correct.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from verdict.graph.models import Evidence
from verdict.llm.client import LLM
from verdict.tools.registry import ToolRuntime

SUBMIT = "submit"


@dataclass
class EpisodeResult:
    output: BaseModel | None
    evidence: list[Evidence]
    steps: int
    stopped: str  # submitted | max_steps | no_tool_calls
    transcript: list[dict[str, Any]] = field(default_factory=list)


def _tool_message(call_id: str, ev: Evidence) -> dict[str, Any]:
    body = {
        "evidence_id": ev.id,
        "ok": ev.ok,
        "summary": ev.summary,
        "output": ev.output,
    }
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(body, ensure_ascii=False),
    }


def run_episode(
    llm: LLM,
    runtime: ToolRuntime,
    system: str,
    task: str,
    output_schema: type[BaseModel],
    max_steps: int = 12,
    max_parallel: int = 6,
) -> EpisodeResult:
    submit_schema = output_schema.model_json_schema()
    submit_schema.pop("title", None)
    tools = runtime.schemas() + [
        {
            "type": "function",
            "function": {
                "name": SUBMIT,
                "description": "Submit your final, structured result. Call exactly once, "
                "after you have gathered the evidence you need.",
                "parameters": submit_schema,
            },
        }
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    seen: list[Evidence] = []

    for step in range(1, max_steps + 1):
        resp = llm.chat(messages, tools=tools, tool_choice="auto")
        messages.append(resp.raw_assistant_message)

        if not resp.tool_calls:
            # Nudge once: the model must finish through submit.
            messages.append(
                {
                    "role": "user",
                    "content": "You must finish by calling the `submit` tool with your result.",
                }
            )
            if step == max_steps:
                return EpisodeResult(None, seen, step, "no_tool_calls", messages)
            continue

        submits = [c for c in resp.tool_calls if c.name == SUBMIT]
        others = [c for c in resp.tool_calls if c.name != SUBMIT]

        if others:
            with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                results = list(pool.map(lambda c: runtime.call(c.name, c.arguments), others))
            for call, ev in zip(others, results, strict=True):
                seen.append(ev)
                messages.append(_tool_message(call.id, ev))

        for call in submits:
            try:
                out = output_schema.model_validate(call.arguments)
            except ValidationError as e:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"submit rejected, fix and resubmit: {e.errors()}",
                    }
                )
                continue
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "accepted"})
            return EpisodeResult(out, seen, step, "submitted", messages)

    return EpisodeResult(None, seen, max_steps, "max_steps", messages)
