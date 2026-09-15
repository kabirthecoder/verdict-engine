from verdict.engine import compute_verdict, explain_claim
from verdict.graph import (
    Attack,
    Audit,
    Author,
    Claim,
    Evidence,
    GraphStore,
    Label,
    Question,
    Role,
    Support,
)

RT = Author(role=Role.RUNTIME, name="tools")
AG = Author(role=Role.CLAIMANT, run_id="r", name="m")
AD = Author(role=Role.ADVERSARY, run_id="r", name="m")
AU = Author(role=Role.AUDITOR, run_id="r", name="m")


def _claim(store, q, kind, text, output=None):
    c = store.add_claim(Claim(question_id=q.id, kind=kind, text=text, author=AG))
    if output is not None:
        e = store.add_evidence(
            Evidence(question_id=q.id, tool="t", args={}, ok=True, output=output, author=RT)
        )
        store.add_support(Support(claim_id=c.id, evidence_id=e.id, quote=output[:10], author=AG))
    return c


def test_supported_claim_attacked_by_unsupported_claim_still_accepted():
    store = GraphStore("sqlite://")
    q = store.add_question(Question(text="?", domain="t"))
    safe = _claim(store, q, "safe", "x is safe", output="no known vulnerabilities")
    hunch = _claim(store, q, "unsafe", "x is unsafe")  # no evidence
    store.add_attack(Attack(attacker_id=hunch.id, target_id=safe.id, rationale="gut", author=AD))
    v = compute_verdict(store.load(q.id), round=1, reason="fixed_point")
    assert v.labels[safe.id] is Label.ACCEPTED
    assert v.labels[hunch.id] is Label.UNSUPPORTED


def test_supported_attack_rejects_claim():
    store = GraphStore("sqlite://")
    q = store.add_question(Question(text="?", domain="t"))
    safe = _claim(store, q, "safe", "x is safe", output="no known vulnerabilities")
    vuln = _claim(store, q, "vulnerable", "x has CVE-1", output="CVE-1 affects x")
    store.add_attack(Attack(attacker_id=vuln.id, target_id=safe.id, rationale="cve", author=AD))
    g = store.load(q.id)
    v = compute_verdict(g, round=1, reason="fixed_point")
    assert v.labels == {safe.id: Label.REJECTED, vuln.id: Label.ACCEPTED}
    assert explain_claim(g, safe.id) == f"rejected: attacked by accepted {vuln.id}"


def test_failed_audit_drops_attacker_and_reinstates_target():
    store = GraphStore("sqlite://")
    q = store.add_question(Question(text="?", domain="t"))
    safe = _claim(store, q, "safe", "x is safe", output="no known vulnerabilities")
    vuln = _claim(store, q, "vulnerable", "x has CVE-1", output="CVE-1 affects y, not x")
    store.add_attack(Attack(attacker_id=vuln.id, target_id=safe.id, rationale="cve", author=AD))
    sup = store.load(q.id).supports_for(vuln.id)[0]
    store.add_audit(Audit(support_id=sup.id, ok=False, note="quote misreads evidence", author=AU))
    v = compute_verdict(store.load(q.id), round=2, reason="fixed_point")
    assert v.labels == {safe.id: Label.ACCEPTED, vuln.id: Label.UNSUPPORTED}
