from fastapi.testclient import TestClient

import tests.test_court as tc
from tests.fakes import FakeLLM, calls
from verdict.api import Service, create_app
from verdict.config import Settings
from verdict.domains.base import register


def _script():
    return [
        calls(("registry_lookup", {"package": "foo"})),
        tc._claimant_submit,
        calls(("submit", {"entailed": True, "note": "no vulns listed"})),
        calls(("advisories", {"package": "foo"})),
        tc._adversary_submit,
        calls(("submit", {"entailed": True, "note": "advisory listed"})),
        calls(("submit", {"new_citations": [], "attacks": []})),
        calls(("submit", {"attacks": [], "gave_up": ["x"]})),
    ]


def test_open_case_runs_and_exposes_proof():
    register(tc.TOY)
    svc = Service(
        Settings(database_url="sqlite://", max_rounds=5), llm_factory=lambda: FakeLLM(_script())
    )
    client = TestClient(create_app(svc))

    assert client.get("/health").json()["ok"] is True
    r = client.post("/cases", json={"question": "Is foo 1.0 safe?", "domain": "toy"})
    assert r.status_code == 202
    cid = r.json()["id"]
    svc.wait()

    case = client.get(f"/cases/{cid}").json()
    assert case["status"] == "done" and case["stop_reason"] == "fixed_point"
    assert case["counts"]["attacks"] == 1
    assert set(case["labels"].values()) == {"accepted", "rejected"}

    graph = client.get(f"/cases/{cid}/graph").json()
    assert len(graph["claims"]) == 2 and len(graph["evidence"]) == 2
    assert all(e["author"]["role"] == "runtime" for e in graph["evidence"])

    events = client.get(f"/cases/{cid}/events").json()
    assert any("verdict: fixed_point" in e for e in events)
    assert client.get("/cases").json()[0]["id"] == cid


def test_unknown_domain_and_case():
    svc = Service(Settings(database_url="sqlite://"), llm_factory=lambda: FakeLLM([]))
    client = TestClient(create_app(svc))
    assert (
        client.post("/cases", json={"question": "is x safe?", "domain": "nope"}).status_code == 400
    )
    assert client.get("/cases/q_missing").status_code == 404
