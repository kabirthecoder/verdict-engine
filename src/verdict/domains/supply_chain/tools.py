"""Real evidence sources for supply-chain questions.

Every function here is a tool the agents can call. Outputs are trimmed to what an
agent needs to cite, but nothing is interpreted: no "safe"/"unsafe" fields, just data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from verdict.domains.supply_chain.http import get_json, github_headers, post_json
from verdict.tools.registry import ToolResult, tool

OSV = "https://api.osv.dev/v1"
DEPS = "https://api.deps.dev/v3"
PYPI = "https://pypi.org/pypi"
NPM = "https://registry.npmjs.org"
GH = "https://api.github.com"

_ECOSYSTEMS = {"pypi": "PyPI", "npm": "npm", "go": "Go", "cargo": "crates.io", "maven": "Maven"}
_DEPS_SYSTEMS = {"pypi": "PYPI", "npm": "NPM", "go": "GO", "cargo": "CARGO", "maven": "MAVEN"}


def _eco(ecosystem: str) -> str:
    key = ecosystem.lower()
    if key not in _ECOSYSTEMS:
        raise ValueError(f"unsupported ecosystem {ecosystem!r}; use one of {sorted(_ECOSYSTEMS)}")
    return _ECOSYSTEMS[key]


# ----------------------------------------------------------------------------- OSV


class Advisory(ToolResult):
    id: str
    aliases: list[str]
    summary: str
    severity: list[dict[str, Any]]
    published: str | None
    modified: str | None
    affected_ranges: list[dict[str, Any]]
    references: list[str]


class VulnQuery(ToolResult):
    ecosystem: str
    package: str
    version: str
    advisories: list[Advisory]

    def one_line(self) -> str:
        return f"{len(self.advisories)} advisories for {self.package} {self.version}"


def _advisory(v: dict[str, Any], package: str | None = None) -> Advisory:
    ranges: list[dict[str, Any]] = []
    for aff in v.get("affected", []):
        pkg = aff.get("package", {})
        if package and pkg.get("name", "").lower() != package.lower():
            continue
        for r in aff.get("ranges", []):
            ranges.append(
                {
                    "package": pkg.get("name"),
                    "ecosystem": pkg.get("ecosystem"),
                    "type": r.get("type"),
                    "events": r.get("events", []),
                }
            )
        if aff.get("versions"):
            ranges.append({"package": pkg.get("name"), "versions": aff["versions"][:50]})
    return Advisory(
        id=v["id"],
        aliases=v.get("aliases", []),
        summary=(v.get("summary") or v.get("details") or "")[:500],
        severity=v.get("severity", []),
        published=v.get("published"),
        modified=v.get("modified"),
        affected_ranges=ranges,
        references=[r.get("url") for r in v.get("references", []) if r.get("url")][:10],
    )


@tool
def osv_query(ecosystem: str, package: str, version: str) -> VulnQuery:
    """Query the OSV.dev vulnerability database for advisories affecting one exact
    package version. ecosystem: pypi | npm | go | cargo | maven."""
    body = {"package": {"name": package, "ecosystem": _eco(ecosystem)}, "version": version}
    data = post_json(f"{OSV}/query", body) or {}
    return VulnQuery(
        ecosystem=ecosystem,
        package=package,
        version=version,
        advisories=[_advisory(v, package) for v in data.get("vulns", [])],
    )


@tool
def osv_advisory(advisory_id: str) -> Advisory:
    """Fetch one advisory by OSV/GHSA/CVE id, including affected version ranges."""
    data = get_json(f"{OSV}/vulns/{advisory_id}")
    if data is None:
        raise LookupError(f"advisory {advisory_id} not found in OSV")
    return _advisory(data)


# ---------------------------------------------------------------------- deps.dev


class DependencyGraph(ToolResult):
    package: str
    version: str
    direct: list[dict[str, str]]
    transitive_count: int
    nodes: list[dict[str, Any]]

    def one_line(self) -> str:
        return f"{len(self.direct)} direct, {self.transitive_count} transitive dependencies"


@tool
def dependency_graph(ecosystem: str, package: str, version: str) -> DependencyGraph:
    """Resolved dependency graph (direct + transitive) of a package version, from deps.dev."""
    system = _DEPS_SYSTEMS[ecosystem.lower()]
    data = get_json(f"{DEPS}/systems/{system}/packages/{package}/versions/{version}:dependencies")
    if data is None:
        raise LookupError(f"{package} {version} not found on deps.dev")
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    root_children = {e["toNode"] for e in edges if e.get("fromNode") == 0}
    direct = [
        {"name": nodes[i]["versionKey"]["name"], "version": nodes[i]["versionKey"]["version"]}
        for i in sorted(root_children)
    ]
    return DependencyGraph(
        package=package,
        version=version,
        direct=direct,
        transitive_count=max(len(nodes) - 1, 0),
        nodes=[
            {
                "name": n["versionKey"]["name"],
                "version": n["versionKey"]["version"],
                "relation": n.get("relation"),
                "errors": n.get("errors", []),
            }
            for n in nodes[:200]
        ],
    )


class VersionInfo(ToolResult):
    package: str
    version: str
    published_at: str | None
    licenses: list[str]
    advisory_keys: list[str]
    is_default: bool | None


@tool
def version_info(ecosystem: str, package: str, version: str) -> VersionInfo:
    """Publication date, licenses and advisory keys for a package version, from deps.dev."""
    system = _DEPS_SYSTEMS[ecosystem.lower()]
    data = get_json(f"{DEPS}/systems/{system}/packages/{package}/versions/{version}")
    if data is None:
        raise LookupError(f"{package} {version} not found on deps.dev")
    return VersionInfo(
        package=package,
        version=version,
        published_at=data.get("publishedAt"),
        licenses=data.get("licenses", []),
        advisory_keys=[a.get("id") for a in data.get("advisoryKeys", [])],
        is_default=data.get("isDefault"),
    )


class ProjectInfo(ToolResult):
    project: str
    description: str | None
    stars: int | None
    forks: int | None
    open_issues: int | None
    license: str | None
    scorecard: dict[str, Any] | None


@tool
def project_info(project: str) -> ProjectInfo:
    """Repository health from deps.dev: stars, forks, open issues, OpenSSF Scorecard.
    project looks like github.com/psf/requests."""
    data = get_json(f"{DEPS}/projects/{project}")
    if data is None:
        raise LookupError(f"project {project} not found on deps.dev")
    sc = data.get("scorecard")
    checks = None
    if sc:
        checks = {
            "date": sc.get("date"),
            "overall": sc.get("overallScore"),
            "checks": {c["name"]: c.get("score") for c in sc.get("checks", [])},
        }
    return ProjectInfo(
        project=project,
        description=data.get("description"),
        stars=data.get("starsCount"),
        forks=data.get("forksCount"),
        open_issues=data.get("openIssuesCount"),
        license=data.get("license"),
        scorecard=checks,
    )


# ------------------------------------------------------------------- registries


class PackageInfo(ToolResult):
    ecosystem: str
    package: str
    latest_version: str | None
    summary: str | None
    home_page: str | None
    repository: str | None
    author: str | None
    maintainers: list[str]
    first_release: str | None
    latest_release: str | None
    release_count: int
    recent_releases: list[dict[str, Any]]
    requested_version: dict[str, Any] | None

    def one_line(self) -> str:
        return f"{self.package}: latest {self.latest_version}, {self.release_count} releases"


def _pypi_repo_url(info: dict[str, Any]) -> str | None:
    urls = info.get("project_urls") or {}
    for k, v in urls.items():
        if any(s in k.lower() for s in ("source", "repository", "code", "github")):
            return v
    for v in urls.values():
        if "github.com" in (v or ""):
            return v
    return None


@tool
def package_info(ecosystem: str, package: str, version: str | None = None) -> PackageInfo:
    """Registry metadata (PyPI or npm): latest version, release history and dates,
    repository link, maintainers. Pass version to also get that version's details
    (upload time, yanked flag, declared dependencies)."""
    eco = ecosystem.lower()
    if eco == "pypi":
        data = get_json(f"{PYPI}/{package}/json")
        if data is None:
            raise LookupError(f"{package} not found on PyPI")
        info = data["info"]
        releases = data.get("releases", {})
        dated = []
        for ver, files in releases.items():
            if files:
                dated.append((min(f["upload_time_iso_8601"] for f in files), ver, files))
        dated.sort()
        requested = None
        if version:
            files = releases.get(version)
            if files is None:
                requested = {"version": version, "exists": False}
            else:
                requested = {
                    "version": version,
                    "exists": True,
                    "upload_time": min(f["upload_time_iso_8601"] for f in files) if files else None,
                    "yanked": any(f.get("yanked") for f in files),
                    "yanked_reason": next(
                        (f.get("yanked_reason") for f in files if f.get("yanked")), None
                    ),
                    "files": len(files),
                }
                if version == info.get("version"):
                    requested["requires_dist"] = (info.get("requires_dist") or [])[:60]
        return PackageInfo(
            ecosystem="pypi",
            package=package,
            latest_version=info.get("version"),
            summary=info.get("summary"),
            home_page=info.get("home_page") or None,
            repository=_pypi_repo_url(info),
            author=info.get("author") or info.get("author_email") or None,
            maintainers=[m for m in [info.get("maintainer"), info.get("maintainer_email")] if m],
            first_release=dated[0][0] if dated else None,
            latest_release=dated[-1][0] if dated else None,
            release_count=len(dated),
            recent_releases=[{"version": v, "date": d} for d, v, _ in dated[-8:]],
            requested_version=requested,
        )
    if eco == "npm":
        data = get_json(f"{NPM}/{package}")
        if data is None:
            raise LookupError(f"{package} not found on npm")
        times = {k: v for k, v in data.get("time", {}).items() if k not in ("created", "modified")}
        dated = sorted((t, v) for v, t in times.items())
        latest = data.get("dist-tags", {}).get("latest")
        repo = data.get("repository")
        repo_url = repo.get("url") if isinstance(repo, dict) else repo
        requested = None
        if version:
            vdata = data.get("versions", {}).get(version)
            requested = (
                {"version": version, "exists": False}
                if vdata is None
                else {
                    "version": version,
                    "exists": True,
                    "upload_time": times.get(version),
                    "deprecated": vdata.get("deprecated"),
                    "dependencies": vdata.get("dependencies", {}),
                    "has_install_scripts": any(
                        k in vdata.get("scripts", {})
                        for k in ("preinstall", "install", "postinstall")
                    ),
                }
            )
        return PackageInfo(
            ecosystem="npm",
            package=package,
            latest_version=latest,
            summary=data.get("description"),
            home_page=data.get("homepage"),
            repository=repo_url,
            author=(data.get("author") or {}).get("name")
            if isinstance(data.get("author"), dict)
            else data.get("author"),
            maintainers=[m.get("name") for m in data.get("maintainers", []) if m.get("name")],
            first_release=dated[0][0] if dated else None,
            latest_release=dated[-1][0] if dated else None,
            release_count=len(dated),
            recent_releases=[{"version": v, "date": d} for d, v in dated[-8:]],
            requested_version=requested,
        )
    raise ValueError("package_info supports ecosystem pypi or npm")


# ----------------------------------------------------------------------- GitHub


class RepoActivity(ToolResult):
    repo: str
    default_branch: str | None
    archived: bool | None
    pushed_at: str | None
    stars: int | None
    open_issues: int | None
    recent_commits: list[dict[str, Any]]
    days_since_last_push: int | None

    def one_line(self) -> str:
        return f"{self.repo}: last push {self.pushed_at}, archived={self.archived}"


@tool
def repo_activity(owner: str, repo: str) -> RepoActivity:
    """GitHub repository activity: archived flag, last push, last 10 commits with authors
    and dates. Set GITHUB_TOKEN for higher rate limits."""
    h = github_headers()
    r = get_json(f"{GH}/repos/{owner}/{repo}", headers=h)
    if r is None:
        raise LookupError(f"github repo {owner}/{repo} not found")
    commits = (
        get_json(f"{GH}/repos/{owner}/{repo}/commits", params={"per_page": 10}, headers=h) or []
    )
    pushed = r.get("pushed_at")
    days = None
    if pushed:
        dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
        days = (datetime.now(UTC) - dt).days
    return RepoActivity(
        repo=f"{owner}/{repo}",
        default_branch=r.get("default_branch"),
        archived=r.get("archived"),
        pushed_at=pushed,
        stars=r.get("stargazers_count"),
        open_issues=r.get("open_issues_count"),
        recent_commits=[
            {
                "sha": c["sha"][:10],
                "date": c["commit"]["author"]["date"],
                "author": c["commit"]["author"]["name"],
                "login": (c.get("author") or {}).get("login"),
                "message": c["commit"]["message"].splitlines()[0][:120],
            }
            for c in commits
        ],
        days_since_last_push=days,
    )


class Contributor(ToolResult):
    login: str
    account_created: str | None
    public_repos: int | None
    followers: int | None
    account_age_days: int | None
    contributions_to_repo: int | None


@tool
def contributor_profile(
    login: str, owner: str | None = None, repo: str | None = None
) -> Contributor:
    """GitHub account age and footprint for a contributor; optionally how many commits
    they have in a given repository. Useful for 'is this PR author trustworthy' questions."""
    h = github_headers()
    u = get_json(f"{GH}/users/{login}", headers=h)
    if u is None:
        raise LookupError(f"github user {login} not found")
    created = u.get("created_at")
    age = None
    if created:
        age = (datetime.now(UTC) - datetime.fromisoformat(created.replace("Z", "+00:00"))).days
    contribs = None
    if owner and repo:
        cs = (
            get_json(f"{GH}/repos/{owner}/{repo}/contributors", params={"per_page": 100}, headers=h)
            or []
        )
        contribs = next((c["contributions"] for c in cs if c.get("login") == login), 0)
    return Contributor(
        login=login,
        account_created=created,
        public_repos=u.get("public_repos"),
        followers=u.get("followers"),
        account_age_days=age,
        contributions_to_repo=contribs,
    )


TOOLS = [
    osv_query,
    osv_advisory,
    dependency_graph,
    version_info,
    project_info,
    package_info,
    repo_activity,
    contributor_profile,
]
