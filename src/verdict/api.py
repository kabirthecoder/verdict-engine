"""HTTP service: open cases, run them in the background, read the proof.

    uv run uvicorn verdict.api:app --reload

POST /cases            {"question": "...", "domain": "supply-chain", "context": {...}}
GET  /cases            list with headline verdicts
GET  /cases/{id}       status + labels + headline
GET  /cases/{id}/graph the full argument graph (claims, evidence, supports, attacks, audits)
GET  /cases/{id}/events court log so far
GET  /health

Jobs run on a small thread pool inside the process. Good for one box; a queue-backed
worker comes with the GitHub App.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from verdict import __version__
from verdict.config import Settings, load_settings
from verdict.court.loop import Court
from verdict.domains.base import get_domain
from verdict.graph.models import Graph, Label, Question
from verdict.graph.store import GraphStore
from verdict.llm.client import LLM


class OpenCase(BaseModel):
    question: str = Field(min_length=3)
    domain: str = "supply-chain"
    context: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Job:
    status: str = "queued"  # queued | running | done | failed
    events: list[str] = field(default_factory=list)
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class Service:
    """Holds the store, the LLM factory and the job table. Tests swap the LLM factory."""

    def __init__(self, settings: Settings | None = None, llm_factory=None, workers: int = 2):
        self.settings = settings or load_settings()
        self.store = GraphStore(self.settings.database_url)
        self.llm_factory = llm_factory or (lambda: LLM(self.settings))
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="court")
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def open(self, req: OpenCase) -> Question:
        dom = get_domain(req.domain)
        q = self.store.add_question(
            Question(text=req.question, domain=dom.name, context=req.context)
        )
        job = Job()
        with self._lock:
            self.jobs[q.id] = job
        self.pool.submit(self._run, q, job)
        return q

    def _run(self, q: Question, job: Job) -> None:
        job.status = "running"
        try:
            dom = get_domain(q.domain)
            court = Court(
                self.store, self.llm_factory(), dom, self.settings, on_event=job.events.append
            )
            court.run(q)
            st = court.stats
            job.stats = {
                "rounds": st.rounds,
                "episodes": st.episodes,
                "tool_calls": st.tool_calls,
                "claims": st.claims,
                "attacks": st.attacks,
                "audits_failed": st.audits_failed,
                "tokens": st.prompt_tokens + st.completion_tokens,
                "seconds": round(st.seconds, 1),
                "stop_reason": st.stop_reason,
            }
            job.status = "done"
        except Exception as e:  # noqa: BLE001 - surfaced to the client
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"

    def wait(self, timeout: float = 30.0) -> None:
        """Test helper: block until all submitted jobs finish."""
        self.pool.shutdown(wait=True)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="court")


def headline(g: Graph) -> str:
    v = g.latest_verdict
    if v is None:
        return "pending"
    kinds = get_domain(g.question.domain).conclusion_kinds
    accepted = [c for c in g.claims if c.kind in kinds and v.labels.get(c.id) is Label.ACCEPTED]
    if accepted:
        return accepted[0].kind
    if any(c.kind in kinds for c in g.claims):
        return "undecided"
    return "no_conclusion"


def create_app(service: Service | None = None) -> FastAPI:
    svc = service or Service()
    app = FastAPI(title="verdict-engine", version=__version__)
    app.state.service = svc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "model": svc.settings.llm_model}

    @app.post("/cases", status_code=202)
    def open_case(req: OpenCase) -> dict[str, Any]:
        try:
            q = svc.open(req)
        except KeyError as e:
            raise HTTPException(400, str(e)) from None
        return {"id": q.id, "status": "queued"}

    @app.get("/cases")
    def list_cases() -> list[dict[str, Any]]:
        out = []
        for q in svc.store.list_questions():
            g = svc.store.load(q.id)
            job = svc.jobs.get(q.id)
            out.append(
                {
                    "id": q.id,
                    "question": q.text,
                    "domain": q.domain,
                    "status": job.status if job else ("done" if g.latest_verdict else "unknown"),
                    "headline": headline(g),
                }
            )
        return out

    @app.get("/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        try:
            g = svc.store.load(case_id)
        except KeyError:
            raise HTTPException(404, "no such case") from None
        job = svc.jobs.get(case_id)
        v = g.latest_verdict
        return {
            "id": g.question.id,
            "question": g.question.text,
            "domain": g.question.domain,
            "status": job.status if job else ("done" if v else "unknown"),
            "error": job.error if job else None,
            "headline": headline(g),
            "labels": v.labels if v else {},
            "stop_reason": v.reason if v else None,
            "stats": job.stats if job else {},
            "counts": {
                "claims": len(g.claims),
                "evidence": len(g.evidence),
                "attacks": len(g.attacks),
                "audits": len(g.audits),
            },
        }

    @app.get("/cases/{case_id}/graph")
    def get_graph(case_id: str) -> dict[str, Any]:
        try:
            return svc.store.load(case_id).model_dump(mode="json")
        except KeyError:
            raise HTTPException(404, "no such case") from None

    @app.get("/cases/{case_id}/events")
    def get_events(case_id: str) -> list[str]:
        job = svc.jobs.get(case_id)
        if job is None:
            raise HTTPException(404, "no such case in this process")
        return job.events

    return app


app = create_app()
