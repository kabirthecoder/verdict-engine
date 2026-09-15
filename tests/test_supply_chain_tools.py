"""Supply-chain tools against recorded responses (shape-accurate for OSV, deps.dev, GitHub)."""

import respx
from httpx import Response

from verdict.domains import get_domain
from verdict.domains.supply_chain import tools as t

OSV_QUERY = {
    "vulns": [
        {
            "id": "GHSA-9wx4-h78v-vm56",
            "aliases": ["CVE-2024-35195"],
            "summary": "Session does not verify after first request with verify=False",
            "published": "2024-05-20T20:15:00Z",
            "modified": "2024-06-21T16:24:11Z",
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:H/I:H/A:N"}
            ],
            "affected": [
                {
                    "package": {"name": "requests", "ecosystem": "PyPI"},
                    "ranges": [
                        {"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.32.0"}]}
                    ],
                }
            ],
            "references": [
                {
                    "type": "ADVISORY",
                    "url": "https://github.com/psf/requests/security/advisories/GHSA-9wx4-h78v-vm56",
                }
            ],
        }
    ]
}

DEPS_GRAPH = {
    "nodes": [
        {
            "versionKey": {"system": "PYPI", "name": "requests", "version": "2.28.0"},
            "relation": "SELF",
        },
        {
            "versionKey": {"system": "PYPI", "name": "urllib3", "version": "1.26.9"},
            "relation": "DIRECT",
        },
        {"versionKey": {"system": "PYPI", "name": "idna", "version": "3.3"}, "relation": "DIRECT"},
    ],
    "edges": [{"fromNode": 0, "toNode": 1}, {"fromNode": 0, "toNode": 2}],
}

GH_REPO = {
    "default_branch": "main",
    "archived": False,
    "pushed_at": "2026-09-01T10:00:00Z",
    "stargazers_count": 52000,
    "open_issues_count": 240,
}
GH_COMMITS = [
    {
        "sha": "abcdef1234567890",
        "commit": {
            "author": {"name": "Nate", "date": "2026-09-01T09:00:00Z"},
            "message": "fix: thing\n\nlong body",
        },
        "author": {"login": "nateprewitt"},
    }
]


@respx.mock
def test_osv_query_flattens_ranges():
    respx.post(f"{t.OSV}/query").mock(return_value=Response(200, json=OSV_QUERY))
    r = t.osv_query.invoke({"ecosystem": "pypi", "package": "requests", "version": "2.28.0"})
    assert r.one_line() == "1 advisories for requests 2.28.0 (osv.dev)"
    adv = r.advisories[0]
    assert adv.id == "GHSA-9wx4-h78v-vm56" and adv.aliases == ["CVE-2024-35195"]
    assert adv.affected_ranges[0]["events"][1] == {"fixed": "2.32.0"}


@respx.mock
def test_osv_query_empty_and_bad_ecosystem():
    respx.post(f"{t.OSV}/query").mock(return_value=Response(200, json={}))
    r = t.osv_query.invoke({"ecosystem": "npm", "package": "left-pad", "version": "1.3.0"})
    assert r.advisories == []
    try:
        t.osv_query.invoke({"ecosystem": "gem", "package": "x", "version": "1"})
    except ValueError as e:
        assert "unsupported ecosystem" in str(e)


@respx.mock
def test_dependency_graph_direct_vs_transitive():
    respx.get(f"{t.DEPS}/systems/PYPI/packages/requests/versions/2.28.0:dependencies").mock(
        return_value=Response(200, json=DEPS_GRAPH)
    )
    r = t.dependency_graph.invoke({"ecosystem": "pypi", "package": "requests", "version": "2.28.0"})
    assert [d["name"] for d in r.direct] == ["urllib3", "idna"]
    assert r.transitive_count == 2


@respx.mock
def test_repo_activity():
    respx.get(f"{t.GH}/repos/psf/requests").mock(return_value=Response(200, json=GH_REPO))
    respx.get(f"{t.GH}/repos/psf/requests/commits").mock(
        return_value=Response(200, json=GH_COMMITS)
    )
    r = t.repo_activity.invoke({"owner": "psf", "repo": "requests"})
    assert r.archived is False and r.days_since_last_push is not None
    assert r.recent_commits[0]["message"] == "fix: thing"
    assert r.recent_commits[0]["login"] == "nateprewitt"


@respx.mock
def test_not_found_becomes_lookup_error():
    respx.get(f"{t.DEPS}/projects/github.com/nope/nope").mock(return_value=Response(404))
    try:
        t.project_info.invoke({"project": "github.com/nope/nope"})
        raise AssertionError("expected LookupError")
    except LookupError:
        pass


def test_domain_registered_and_frames_question():
    from verdict.graph import Question

    d = get_domain("supply-chain")
    assert {k.name for k in d.claim_kinds} >= {"vulnerable", "safe_to_adopt", "not_safe_to_adopt"}
    q = Question(text="Is requests 2.28.0 safe to adopt?", domain="supply-chain")
    frame = d.frame(q)
    assert (
        "package: requests" in frame and "version: 2.28.0" in frame and "ecosystem: pypi" in frame
    )


@respx.mock
def test_osv_falls_back_to_github_advisories():
    import httpx

    respx.post(f"{t.OSV}/query").mock(side_effect=httpx.ConnectError("blocked"))
    respx.get(f"{t.GH}/advisories").mock(
        return_value=Response(
            200,
            json=[
                {
                    "ghsa_id": "GHSA-9wx4-h78v-vm56",
                    "cve_id": "CVE-2024-35195",
                    "summary": "verify=False persists",
                    "severity": "medium",
                    "published_at": "2024-05-20T20:15:00Z",
                    "updated_at": "2024-06-21T16:24:11Z",
                    "html_url": "https://github.com/advisories/GHSA-9wx4-h78v-vm56",
                    "vulnerabilities": [
                        {
                            "package": {"ecosystem": "pip", "name": "requests"},
                            "vulnerable_version_range": "< 2.32.0",
                            "patched_versions": "2.32.0",
                        }
                    ],
                }
            ],
        )
    )
    r = t.osv_query.invoke({"ecosystem": "pypi", "package": "requests", "version": "2.28.0"})
    assert r.source.startswith("github-advisory-db")
    assert r.advisories[0].id == "GHSA-9wx4-h78v-vm56"
    assert r.advisories[0].aliases == ["CVE-2024-35195"]
    assert r.advisories[0].affected_ranges[0]["patched_versions"] == "2.32.0"
