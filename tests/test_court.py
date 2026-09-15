"""End-to-end court run with a scripted LLM and a toy domain.

Scenario: the claimant says package foo is safe (cites a registry lookup). The
adversary finds an advisory and attacks. Nobody can refute the advisory. Final
verdict: 'safe' rejected, 'vulnerable' accepted — computed, not decided by anyone.
"""

from tests.fakes import FakeLLM, calls, last_evidence_id
from verdict.config import Settings
from verdict.court import Court
from verdict.domains import ClaimKind, Domain
from verdict.graph import GraphStore, Label, Question, Role
from verdict.tools import ToolResult, tool


class Registry(ToolResult):
    package: str
    vulns: list[str]


class Advisories(ToolResult):
    package: str
    advisories: list[str]


@tool
def registry_lookup(package: str) -> Registry:
    """Look the package up in the registry."""
    return Registry(package=package, vulns=[])


@tool
def advisories(package: str) -> Advisories:
    """Search the advisory database."""
    return Advisories(package=package, advisories=["GHSA-1"] if package == "foo" else [])


TOY = Domain(
    name="toy",
    description="toy packages",
    tools=[registry_lookup, advisories],
    claim_kinds=[
        ClaimKind("safe", "package has no known issues", ["package"]),
        ClaimKind("vulnerable", "package has an advisory", ["package", "advisory"]),
    ],
)


def _claimant_submit(messages):
    eid = last_evidence_id(messages)
    return calls(
        (
            "submit",
            {
                "claims": [
                    {
                        "kind": "safe",
                        "params": {"package": "foo"},
                        "text": "foo has no known vulnerabilities",
                        "citations": [{"evidence_id": eid, "quote": '"vulns":[]'}],
                    }
                ]
            },
        )
    )


def _adversary_submit(messages):
    eid = last_evidence_id(messages)
    # the target id is whatever claim is listed; pull it from the task text
    task = messages[1]["content"]
    target = next(t for t in task.split() if t.startswith("c_"))
    return calls(
        (
            "submit",
            {
                "attacks": [
                    {
                        "target_claim_id": target,
                        "rationale": "an advisory exists",
                        "claim": {
                            "kind": "vulnerable",
                            "params": {"package": "foo", "advisory": "GHSA-1"},
                            "text": "foo is affected by GHSA-1",
                            "citations": [{"evidence_id": eid, "quote": '"GHSA-1"'}],
                        },
                    }
                ]
            },
        )
    )


def test_court_reaches_computed_verdict():
    store = GraphStore("sqlite://")
    q = store.add_question(Question(text="Is foo safe?", domain="toy"))
    llm = FakeLLM(
        [
            # round 1: claimant
            calls(("registry_lookup", {"package": "foo"})),
            _claimant_submit,
            # audit of the claimant's citation
            calls(("submit", {"entailed": True, "note": "no vulns listed"})),
            # adversary on the accepted 'safe' claim
            calls(("advisories", {"package": "foo"})),
            _adversary_submit,
            # audit of the adversary's citation
            calls(("submit", {"entailed": True, "note": "advisory listed"})),
            # round 2: investigator on the attacking claim finds nothing new
            calls(("submit", {"new_citations": [], "attacks": []})),
            # adversary on the now-accepted 'vulnerable' claim gives up
            calls(("submit", {"attacks": [], "gave_up": ["x"]})),
            # round 3 would be a fixed point only if nothing was written in round 2;
            # round 2 wrote nothing, so the loop stops there.
        ]
    )
    court = Court(store, llm, TOY, Settings(max_rounds=5))
    verdict = court.run(q)

    g = store.load(q.id)
    by_kind = {c.kind: c for c in g.claims}
    assert verdict.labels[by_kind["safe"].id] is Label.REJECTED
    assert verdict.labels[by_kind["vulnerable"].id] is Label.ACCEPTED
    assert verdict.reason == "fixed_point"
    assert court.stats.rounds == 2
    assert court.stats.attacks == 1 and court.stats.audits_failed == 0
    # provenance: every support points at evidence written by the runtime
    for s in g.supports:
        assert g.evidence_by_id(s.evidence_id).author.role is Role.RUNTIME
    assert not llm.script, "every scripted LLM turn should have been consumed"


def test_fabricated_quote_is_thrown_out():
    store = GraphStore("sqlite://")
    q = store.add_question(Question(text="Is foo safe?", domain="toy"))

    def bad_claimant(messages):
        eid = last_evidence_id(messages)
        return calls(
            (
                "submit",
                {
                    "claims": [
                        {
                            "kind": "safe",
                            "params": {"package": "foo"},
                            "text": "foo is safe",
                            "citations": [{"evidence_id": eid, "quote": "audited and certified"}],
                        }
                    ]
                },
            )
        )

    llm = FakeLLM(
        [
            calls(("registry_lookup", {"package": "foo"})),
            bad_claimant,
            # round 2: the claim is now unsupported -> investigator; finds nothing
            calls(("submit", {"new_citations": [], "attacks": []})),
        ]
    )
    court = Court(store, llm, TOY, Settings(max_rounds=4))
    verdict = court.run(q)
    (claim,) = store.load(q.id).claims
    assert verdict.labels[claim.id] is Label.UNSUPPORTED
    assert court.stats.audits_failed == 1
