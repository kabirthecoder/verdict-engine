"""Turn an argument graph into a verdict.

1. Claims without live support are UNSUPPORTED and leave the framework entirely: they
   can neither be accepted nor attack anything. (A "live" support is one no audit has
   failed.)
2. The remaining claims and the attacks between them form a Dung framework.
3. Grounded semantics labels each claim accepted / rejected / undecided.
"""

from __future__ import annotations

from verdict.engine.grounded import Framework, Status, explain, grounded
from verdict.graph.models import Graph, Label, Verdict


def framework_of(graph: Graph) -> tuple[Framework, set[str]]:
    supported = {c.id for c in graph.claims if graph.is_supported(c.id)}
    attacks = {
        (a.attacker_id, a.target_id)
        for a in graph.attacks
        if a.attacker_id in supported and a.target_id in supported
    }
    return Framework.build(supported, attacks), supported


def compute_verdict(graph: Graph, round: int, reason: str) -> Verdict:
    fw, supported = framework_of(graph)
    lab = grounded(fw)
    labels: dict[str, Label] = {}
    for c in graph.claims:
        if c.id not in supported:
            labels[c.id] = Label.UNSUPPORTED
            continue
        s = lab.status[c.id]
        labels[c.id] = {
            Status.IN: Label.ACCEPTED,
            Status.OUT: Label.REJECTED,
            Status.UNDEC: Label.UNDECIDED,
        }[s]
    return Verdict(question_id=graph.question.id, round=round, labels=labels, reason=reason)


def explain_claim(graph: Graph, claim_id: str) -> str:
    fw, supported = framework_of(graph)
    if claim_id not in supported:
        return "unsupported: no evidence survived audit"
    return explain(fw, grounded(fw), claim_id)
