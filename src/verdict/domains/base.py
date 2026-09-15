"""What a domain plug-in provides. The court, roles and engine never change per domain."""

from __future__ import annotations

from dataclasses import dataclass, field

from verdict.graph.models import Question
from verdict.tools.registry import Tool


@dataclass
class ClaimKind:
    name: str
    description: str
    params: list[str] = field(default_factory=list)  # names of parameters that identify it

    def render(self) -> str:
        p = ", ".join(self.params) if self.params else "-"
        return f"- {self.name}({p}): {self.description}"


@dataclass
class Domain:
    name: str
    description: str
    tools: list[Tool]
    claim_kinds: list[ClaimKind]
    # Extra guidance per role, appended to the generic role prompt.
    guidance: dict[str, str] = field(default_factory=dict)
    # Claim kinds that state the overall answer (shown as the headline of a case).
    conclusion_kinds: list[str] = field(default_factory=list)

    def vocabulary(self) -> str:
        return "\n".join(k.render() for k in self.claim_kinds)

    def kind_names(self) -> set[str]:
        return {k.name for k in self.claim_kinds}

    def frame(self, question: Question) -> str:
        """Domain-specific framing shown to every role. Override for richer context."""
        ctx = "\n".join(f"- {k}: {v}" for k, v in question.context.items())
        return f"Question: {question.text}\n" + (f"Context:\n{ctx}\n" if ctx else "")


_REGISTRY: dict[str, Domain] = {}


def register(domain: Domain) -> Domain:
    _REGISTRY[domain.name] = domain
    return domain


def get_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        # import built-ins lazily so registering is a side effect of importing the package
        import importlib

        try:
            importlib.import_module(f"verdict.domains.{name.replace('-', '_')}")
        except ModuleNotFoundError:
            pass
    if name not in _REGISTRY:
        raise KeyError(f"unknown domain {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
