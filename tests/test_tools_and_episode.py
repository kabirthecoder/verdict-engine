from pydantic import BaseModel

from tests.fakes import FakeLLM, calls, text
from verdict.graph import GraphStore, Question
from verdict.llm import run_episode
from verdict.tools import ToolResult, ToolRuntime, tool


class Lookup(ToolResult):
    package: str
    vulns: list[str]

    def summary(self) -> str:
        return f"{len(self.vulns)} vulns for {self.package}"


@tool
def lookup_vulns(package: str, version: str = "latest") -> Lookup:
    """Look up known vulnerabilities for a package version."""
    return Lookup(package=package, vulns=["CVE-1"] if package == "bad" else [])


@tool
def explode(x: int) -> int:
    """Always fails."""
    raise RuntimeError("boom")


class Out(BaseModel):
    verdict: str
    evidence_ids: list[str]


def _rt():
    store = GraphStore("sqlite://")
    q = store.add_question(Question(text="?", domain="t"))
    return store, q, ToolRuntime.for_question(store, q.id, [lookup_vulns, explode])


def test_tool_schema_from_signature():
    s = lookup_vulns.schema()
    assert s["function"]["name"] == "lookup_vulns"
    assert s["function"]["parameters"]["required"] == ["package"]
    assert "version" in s["function"]["parameters"]["properties"]


def test_runtime_records_evidence_and_dedupes():
    store, q, rt = _rt()
    e1 = rt.call("lookup_vulns", {"package": "bad"})
    e2 = rt.call("lookup_vulns", {"package": "bad"})
    assert e1.id == e2.id
    assert e1.ok and e1.summary == "1 vulns for bad" and '"CVE-1"' in e1.output
    assert len(store.load(q.id).evidence) == 1
    assert rt.calls == 1


def test_runtime_failures_are_evidence():
    _, _, rt = _rt()
    e = rt.call("explode", {"x": 1})
    assert not e.ok and "boom" in e.output
    e = rt.call("lookup_vulns", {"nope": 1})
    assert not e.ok and "invalid arguments" in e.output
    e = rt.call("missing", {})
    assert not e.ok and "unknown tool" in e.output


def test_episode_runs_parallel_tools_then_submits():
    store, q, rt = _rt()
    llm = FakeLLM(
        [
            calls(("lookup_vulns", {"package": "bad"}), ("lookup_vulns", {"package": "good"})),
            text("thinking..."),  # no tool calls -> nudged
            calls(("submit", {"verdict": "bad has CVE-1", "evidence_ids": ["x"]})),
        ]
    )
    res = run_episode(llm, rt, "sys", "task", Out, max_steps=5)
    assert res.stopped == "submitted"
    assert res.output.verdict == "bad has CVE-1"
    assert len(res.evidence) == 2
    # tool results were fed back with evidence ids
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and all("evidence_id" in m["content"] for m in tool_msgs)


def test_episode_rejects_invalid_submit_then_accepts():
    _, _, rt = _rt()
    llm = FakeLLM(
        [
            calls(("submit", {"verdict": 123})),  # missing evidence_ids -> rejected
            calls(("submit", {"verdict": "ok", "evidence_ids": []})),
        ]
    )
    res = run_episode(llm, rt, "s", "t", Out, max_steps=3)
    assert res.stopped == "submitted" and res.output.verdict == "ok"
    assert any("submit rejected" in m.get("content", "") for m in llm.calls[1])


def test_episode_gives_up_at_max_steps():
    _, _, rt = _rt()
    llm = FakeLLM([calls(("lookup_vulns", {"package": "a"}))] * 3)
    res = run_episode(llm, rt, "s", "t", Out, max_steps=3)
    assert res.stopped == "max_steps" and res.output is None
