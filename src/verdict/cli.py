"""verdict — command line for the evidence court."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from verdict.config import load_settings
from verdict.engine.verdict import explain_claim
from verdict.graph.models import Graph, Label, Question
from verdict.graph.store import GraphStore

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
console = Console()

LABEL_STYLE = {
    Label.ACCEPTED: "bold green",
    Label.REJECTED: "bold red",
    Label.UNDECIDED: "bold yellow",
    Label.UNSUPPORTED: "dim",
}


def _store() -> GraphStore:
    return GraphStore(load_settings().database_url)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The decision to make, in plain words")],
    domain: Annotated[str, typer.Option("--domain", "-d")] = "supply-chain",
    context: Annotated[
        list[str] | None,
        typer.Option("--ctx", "-c", help="key=value context, e.g. -c ecosystem=npm"),
    ] = None,
    max_rounds: Annotated[int | None, typer.Option(help="override VERDICT_MAX_ROUNDS")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Open a case, run the court, print the verdict with its proof."""
    from verdict.court.loop import Court
    from verdict.domains.base import get_domain
    from verdict.llm.client import LLM

    settings = load_settings()
    if max_rounds is not None:
        settings.max_rounds = max_rounds
    dom = get_domain(domain)
    store = GraphStore(settings.database_url)
    ctx = dict(kv.split("=", 1) for kv in (context or []))
    q = store.add_question(Question(text=question, domain=dom.name, context=ctx))
    console.print(f"[dim]case {q.id} · domain {dom.name} · model {settings.llm_model}[/dim]")

    def on_event(msg: str) -> None:
        if not quiet:
            console.print(f"[dim]  · {msg}[/dim]")

    court = Court(store, LLM(settings), dom, settings, on_event=on_event)
    court.run(q)
    _print_case(store.load(q.id))
    st = court.stats
    console.print(
        f"[dim]{st.rounds} rounds · {st.episodes} episodes · {st.tool_calls} tool calls · "
        f"{st.claims} claims · {st.attacks} attacks · {st.audits_failed} failed audits · "
        f"{st.prompt_tokens + st.completion_tokens} tokens · {st.seconds:.0f}s · "
        f"stopped: {st.stop_reason}[/dim]"
    )


@app.command()
def show(
    question_id: Annotated[str, typer.Argument()],
    evidence: Annotated[bool, typer.Option("--evidence", "-e", help="print raw evidence")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print a case: claims, labels, attacks, citations — the proof."""
    g = _store().load(question_id)
    if as_json:
        console.print_json(g.model_dump_json())
        return
    _print_case(g, show_evidence=evidence)


@app.command("list")
def list_cases() -> None:
    """List cases in the database."""
    store = _store()
    t = Table("id", "domain", "question", "verdict")
    for q in store.list_questions():
        g = store.load(q.id)
        v = g.latest_verdict
        t.add_row(q.id, q.domain, q.text[:70], _headline(g) if v else "-")
    console.print(t)


@app.command()
def doctor() -> None:
    """Check the LLM endpoint and every data source this machine can reach."""
    import httpx

    s = load_settings()
    t = Table("check", "result")
    try:
        r = httpx.get(
            f"{s.llm_base_url.rstrip('/')}/models",
            timeout=10,
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
        )
        ok = r.status_code == 200
        t.add_row(
            f"LLM {s.llm_base_url}", "[green]ok[/green]" if ok else f"[red]{r.status_code}[/red]"
        )
    except Exception as e:  # noqa: BLE001
        t.add_row(f"LLM {s.llm_base_url}", f"[red]{type(e).__name__}[/red]")
    probes = {
        "OSV.dev": "https://api.osv.dev/v1/vulns/GHSA-9wx4-h78v-vm56",
        "deps.dev": "https://api.deps.dev/v3/systems/pypi/packages/requests",
        "PyPI": "https://pypi.org/pypi/requests/json",
        "npm": "https://registry.npmjs.org/left-pad",
        "GitHub": "https://api.github.com/repos/psf/requests",
    }
    for name, url in probes.items():
        try:
            code = httpx.get(url, timeout=10, headers={"User-Agent": "verdict-doctor"}).status_code
            t.add_row(name, "[green]ok[/green]" if code == 200 else f"[yellow]{code}[/yellow]")
        except Exception as e:  # noqa: BLE001
            t.add_row(name, f"[red]{type(e).__name__}[/red]")
    console.print(t)


# ------------------------------------------------------------------ rendering


def _headline(g: Graph) -> str:
    v = g.latest_verdict
    if v is None:
        return "no verdict yet"
    from verdict.domains.base import get_domain

    try:
        kinds = get_domain(g.question.domain).conclusion_kinds
    except KeyError:
        kinds = []
    concl = [c for c in g.claims if c.kind in kinds]
    accepted = [c for c in concl if v.labels.get(c.id) is Label.ACCEPTED]
    if accepted:
        return accepted[0].kind.replace("_", " ")
    if concl:
        return "undecided"
    n = {lab: sum(1 for x in v.labels.values() if x is lab) for lab in Label}
    return (
        f"{n[Label.ACCEPTED]} accepted / {n[Label.REJECTED]} rejected / "
        f"{n[Label.UNDECIDED]} undecided"
    )


def _print_case(g: Graph, show_evidence: bool = False) -> None:
    v = g.latest_verdict
    labels = v.labels if v else {}
    head = _headline(g)
    style = "green" if head.startswith("safe") else "red" if head.startswith("not") else "yellow"
    console.print(Panel(f"[bold {style}]{head.upper()}[/bold {style}]", title=g.question.text))

    tree = Tree("[bold]claims[/bold]")
    for c in g.claims:
        lab = labels.get(c.id, Label.UNDECIDED)
        node = tree.add(
            f"[{LABEL_STYLE[lab]}]{lab.value:11}[/] {c.key}  [dim]{c.id} · {c.author.role}[/dim]\n"
            f"  {c.text}\n  [dim]{explain_claim(g, c.id)}[/dim]"
        )
        for s in g.supports_for(c.id):
            e = g.evidence_by_id(s.evidence_id)
            audits = g.audits_for(s.id)
            mark = "✓" if audits and all(a.ok for a in audits) else "✗" if audits else "?"
            node.add(
                f"[dim]cites[/dim] {mark} {e.tool}({json.dumps(e.args)}) "
                f'→ "{s.quote[:90]}"'
                + ("" if not audits or audits[0].ok else f"  [red]{audits[0].note}[/red]")
            )
        for a in g.attacks_on(c.id):
            node.add(f"[red]attacked by[/red] {a.attacker_id}: {a.rationale[:100]}")
    console.print(tree)

    if show_evidence:
        for e in g.evidence:
            console.print(Panel(e.output[:2000], title=f"{e.id} {e.tool}({json.dumps(e.args)})"))

    if v:
        console.print(f"[dim]verdict {v.id} · round {v.round} · stopped: {v.reason}[/dim]")


if __name__ == "__main__":
    app()
