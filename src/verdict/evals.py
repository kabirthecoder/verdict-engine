"""Run the court on cases with known answers and score it.

Scores per case:
  conclusion  — the accepted conclusion claim matches `expect`
  kinds       — every kind in `expect_kinds` has at least one ACCEPTED claim
  faithful    — no citation failed audit (zero hallucinated quotes)
  cost        — tool calls, tokens, seconds
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from verdict.config import Settings
from verdict.court.loop import Court
from verdict.domains.base import get_domain
from verdict.graph.models import Label, Question
from verdict.graph.store import GraphStore
from verdict.llm.client import LLM


@dataclass
class CaseResult:
    id: str
    question_id: str
    expected: str
    got: str
    conclusion_ok: bool
    kinds_ok: bool
    missing_kinds: list[str]
    faithful: bool
    audits_failed: int
    tool_calls: int
    tokens: int
    seconds: float
    stop_reason: str


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["cases"]


def run_case(case: dict[str, Any], store: GraphStore, settings: Settings) -> CaseResult:
    dom = get_domain(case.get("domain", "supply-chain"))
    q = store.add_question(
        Question(text=case["question"], domain=dom.name, context=case.get("context", {}))
    )
    court = Court(store, LLM(settings), dom, settings)
    verdict = court.run(q)
    g = store.load(q.id)

    accepted_kinds = {c.kind for c in g.claims if verdict.labels.get(c.id) is Label.ACCEPTED}
    got = next((k for k in dom.conclusion_kinds if k in accepted_kinds), "undecided")
    missing = [k for k in case.get("expect_kinds", []) if k not in accepted_kinds]
    st = court.stats
    return CaseResult(
        id=case["id"],
        question_id=q.id,
        expected=case["expect"],
        got=got,
        conclusion_ok=got == case["expect"],
        kinds_ok=not missing,
        missing_kinds=missing,
        faithful=st.audits_failed == 0,
        audits_failed=st.audits_failed,
        tool_calls=st.tool_calls,
        tokens=st.prompt_tokens + st.completion_tokens,
        seconds=round(st.seconds, 1),
        stop_reason=st.stop_reason,
    )


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results) or 1
    return {
        "cases": len(results),
        "conclusion_accuracy": sum(r.conclusion_ok for r in results) / n,
        "kinds_accuracy": sum(r.kinds_ok for r in results) / n,
        "faithfulness": sum(r.faithful for r in results) / n,
        "avg_tool_calls": sum(r.tool_calls for r in results) / n,
        "avg_tokens": sum(r.tokens for r in results) / n,
        "avg_seconds": sum(r.seconds for r in results) / n,
        "results": [asdict(r) for r in results],
    }
