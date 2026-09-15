import pytest

from verdict.graph import (
    Attack,
    Audit,
    Author,
    Claim,
    Evidence,
    GraphStore,
    Question,
    Role,
    Support,
)


@pytest.fixture
def store() -> GraphStore:
    return GraphStore("sqlite://")  # in-memory


def runtime() -> Author:
    return Author(role=Role.RUNTIME, name="tools")


def agent(role: Role = Role.CLAIMANT) -> Author:
    return Author(role=role, run_id="run1", name="test-model")


def test_round_trip(store: GraphStore) -> None:
    q = store.add_question(Question(text="Is X safe?", domain="test"))
    c = store.add_claim(
        Claim(question_id=q.id, kind="safe", params={"pkg": "x"}, text="x is safe", author=agent())
    )
    e = store.add_evidence(
        Evidence(
            question_id=q.id,
            tool="lookup",
            args={"pkg": "x"},
            ok=True,
            output='{"vulns": []}',
            summary="no vulns",
            author=runtime(),
        )
    )
    s = store.add_support(
        Support(claim_id=c.id, evidence_id=e.id, quote='"vulns": []', author=agent())
    )
    store.add_audit(Audit(support_id=s.id, ok=True, author=agent(Role.AUDITOR)))

    g = store.load(q.id)
    assert g.question.text == "Is X safe?"
    assert [x.id for x in g.claims] == [c.id]
    assert g.evidence[0].output_hash == e.output_hash
    assert g.is_supported(c.id)
    assert g.claim(c.id).key == "safe(pkg=x)"


def test_evidence_must_come_from_runtime() -> None:
    with pytest.raises(ValueError):
        Evidence(question_id="q", tool="t", args={}, ok=True, output="x", author=agent())


def test_failed_audit_removes_support(store: GraphStore) -> None:
    q = store.add_question(Question(text="?", domain="test"))
    c = store.add_claim(Claim(question_id=q.id, kind="k", text="t", author=agent()))
    e = store.add_evidence(
        Evidence(
            question_id=q.id, tool="t", args={}, ok=True, output="real output", author=runtime()
        )
    )
    s = store.add_support(Support(claim_id=c.id, evidence_id=e.id, quote="made up", author=agent()))
    assert store.load(q.id).is_supported(c.id)
    store.add_audit(
        Audit(support_id=s.id, ok=False, note="quote not in output", author=agent(Role.AUDITOR))
    )
    assert not store.load(q.id).is_supported(c.id)


def test_cross_question_links_rejected(store: GraphStore) -> None:
    q1 = store.add_question(Question(text="a", domain="test"))
    q2 = store.add_question(Question(text="b", domain="test"))
    c1 = store.add_claim(Claim(question_id=q1.id, kind="k", text="t", author=agent()))
    c2 = store.add_claim(Claim(question_id=q2.id, kind="k", text="t", author=agent()))
    with pytest.raises(ValueError):
        store.add_attack(Attack(attacker_id=c1.id, target_id=c2.id, rationale="x", author=agent()))
    with pytest.raises(ValueError):
        Attack(attacker_id=c1.id, target_id=c1.id, rationale="self", author=agent())
