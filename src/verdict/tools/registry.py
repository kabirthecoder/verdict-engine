"""Typed tools and the runtime that turns every call into Evidence.

A tool is a plain function with Pydantic-typed arguments. The decorator captures its
JSON schema (what the LLM sees) and the runtime executes it, serializes the result,
and writes an Evidence node before the calling agent ever sees the output. Agents
never touch the store for evidence — they only get back evidence ids to cite.
"""

from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from verdict.graph.models import Author, Evidence, Role
from verdict.graph.store import GraphStore


class ToolResult(BaseModel):
    """Optional base for tool outputs that want to control their one-line summary."""

    def summary(self) -> str:
        return ""


@dataclass
class Tool:
    name: str
    description: str
    params: type[BaseModel]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function tool schema."""
        s = self.params.model_json_schema()
        s.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": s},
        }

    def invoke(self, args: dict[str, Any]) -> Any:
        parsed = self.params(**args)
        return self.fn(**parsed.model_dump())


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Decorator: build a Tool from a function's signature and docstring."""

    def wrap(f: Callable[..., Any]) -> Tool:
        sig = inspect.signature(f)
        fields: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else Any
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[pname] = (ann, default)
        params = create_model(f"{f.__name__}_params", **fields)
        doc = inspect.getdoc(f) or f.__name__
        return Tool(name=name or f.__name__, description=doc, params=params, fn=f)

    return wrap(fn) if fn is not None else wrap


def serialize(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=None)
    return json.dumps(value, default=str, sort_keys=True, ensure_ascii=False)


@dataclass
class ToolRuntime:
    """Executes tools for one question and records each call as Evidence.

    Identical (tool, args) calls within a question are de-duplicated: the second caller
    gets the same evidence node back, so agents can't inflate evidence by re-calling.
    """

    store: GraphStore
    question_id: str
    tools: dict[str, Tool]
    max_output_chars: int = 40_000
    _cache: dict[str, Evidence] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    calls: int = 0

    @classmethod
    def for_question(cls, store: GraphStore, question_id: str, tools: list[Tool]) -> ToolRuntime:
        return cls(store=store, question_id=question_id, tools={t.name: t for t in tools})

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> Evidence:
        key = name + ":" + json.dumps(args, sort_keys=True, default=str)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        t = self.tools.get(name)
        author = Author(role=Role.RUNTIME, name="tool-runtime")
        if t is None:
            ev = Evidence(
                question_id=self.question_id,
                tool=name,
                args=args,
                ok=False,
                output=f"error: unknown tool {name!r}",
                summary="unknown tool",
                author=author,
            )
        else:
            try:
                result = t.invoke(args)
                out = serialize(result)
                truncated = len(out) > self.max_output_chars
                if truncated:
                    out = out[: self.max_output_chars] + "\n…[truncated]"
                summary = result.summary() if isinstance(result, ToolResult) else ""
                ev = Evidence(
                    question_id=self.question_id,
                    tool=name,
                    args=args,
                    ok=True,
                    output=out,
                    summary=summary,
                    author=author,
                )
            except ValidationError as e:
                ev = Evidence(
                    question_id=self.question_id,
                    tool=name,
                    args=args,
                    ok=False,
                    output=f"error: invalid arguments: {e}",
                    summary="invalid arguments",
                    author=author,
                )
            except Exception as e:  # noqa: BLE001 - tool failures are evidence too
                ev = Evidence(
                    question_id=self.question_id,
                    tool=name,
                    args=args,
                    ok=False,
                    output=f"error: {type(e).__name__}: {e}",
                    summary=f"{type(e).__name__}",
                    author=author,
                )

        self.store.add_evidence(ev)
        with self._lock:
            self._cache[key] = ev
            self.calls += 1
        return ev
