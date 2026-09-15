"""Supply-chain trust domain: is this package version safe to adopt?"""

from __future__ import annotations

import re

from verdict.domains.base import ClaimKind, Domain, register
from verdict.domains.supply_chain.tools import TOOLS
from verdict.graph.models import Question

CLAIM_KINDS = [
    ClaimKind(
        "vulnerable",
        "the version is affected by a published advisory",
        ["ecosystem", "package", "version", "advisory"],
    ),
    ClaimKind(
        "fixed_in",
        "the advisory is fixed from this version on",
        ["package", "advisory", "fixed_version"],
    ),
    ClaimKind(
        "no_known_vulnerabilities",
        "OSV returns no advisories for the version",
        ["ecosystem", "package", "version"],
    ),
    ClaimKind(
        "transitive_vulnerable",
        "a dependency of the version is affected by an advisory",
        ["package", "version", "dependency", "dependency_version", "advisory"],
    ),
    ClaimKind(
        "yanked", "the version was yanked/deprecated on the registry", ["package", "version"]
    ),
    ClaimKind(
        "version_missing", "the version does not exist on the registry", ["package", "version"]
    ),
    ClaimKind(
        "actively_maintained",
        "the project has recent releases or commits",
        ["package"],
    ),
    ClaimKind(
        "unmaintained",
        "no releases/commits for a long time, or repo archived",
        ["package"],
    ),
    ClaimKind(
        "install_scripts",
        "the version runs install-time scripts (npm) — a common attack vector",
        ["package", "version"],
    ),
    ClaimKind(
        "new_or_obscure",
        "package is very new or has very little history",
        ["package"],
    ),
    ClaimKind(
        "license",
        "the version is published under this license",
        ["package", "version", "license"],
    ),
    ClaimKind(
        "safe_to_adopt",
        "overall conclusion: adopting this exact version is acceptable",
        ["ecosystem", "package", "version"],
    ),
    ClaimKind(
        "not_safe_to_adopt",
        "overall conclusion: adopting this exact version is not acceptable",
        ["ecosystem", "package", "version"],
    ),
]

GUIDANCE = {
    "claimant": """
How to open a supply-chain case well:
1. package_info for the exact version (does it exist? yanked? when released? latest?).
2. osv_query for the exact version. Make one `vulnerable` claim per advisory, or one
   `no_known_vulnerabilities` claim if the list is empty.
3. version_info and project_info (use the repository from package_info,
   e.g. github.com/psf/requests).
4. repo_activity for maintenance signals.
5. Only then make ONE overall claim: safe_to_adopt or not_safe_to_adopt, citing the evidence
   that most directly supports it. The overall claim is what the adversary will attack.
Quote exact fragments from the JSON output, e.g. "advisories":[] or "yanked":true.
""",
    "adversary": """
Where claimants usually go wrong in supply-chain cases:
- They query the wrong version, or the latest instead of the requested one.
- They ignore transitive dependencies: call dependency_graph, then osv_query the direct
  dependencies (a handful of calls, do them in parallel). A vulnerable dependency justifies
  a `transitive_vulnerable` attack on safe_to_adopt.
- They say "maintained" from stars alone; check repo_activity dates.
- npm: check install scripts and deprecation in package_info's requested_version.
- They misread OSV ranges: an advisory in the list is not always "affected" if the events
  say the version is after `fixed`; check osv_advisory.
Attack safe_to_adopt / not_safe_to_adopt with specific cited claims. If everything holds, give up.
""",
    "investigator": """
For an unsupported claim, get the primary source: osv_query / osv_advisory for vulnerability
claims, package_info for registry claims, repo_activity for maintenance claims.
""",
}

_PKG_RE = re.compile(r"(?P<pkg>[A-Za-z0-9_.@/\-]+)[ =@]+(?P<ver>\d[\w.\-+]*)")


class SupplyChainDomain(Domain):
    def frame(self, question: Question) -> str:
        ctx = dict(question.context)
        if "package" not in ctx:
            m = _PKG_RE.search(question.text)
            if m:
                ctx.setdefault("package", m.group("pkg"))
                ctx.setdefault("version", m.group("ver"))
        ctx.setdefault("ecosystem", "pypi")
        lines = "\n".join(f"- {k}: {v}" for k, v in ctx.items())
        return (
            f"Question: {question.text}\nContext:\n{lines}\n"
            "Answer for the exact version given. Use the ecosystem value in tool calls.\n"
        )


DOMAIN = register(
    SupplyChainDomain(
        name="supply-chain",
        description="Decide whether a specific open-source package version is safe to adopt, "
        "using live data from OSV.dev, deps.dev, PyPI/npm and GitHub.",
        tools=TOOLS,
        claim_kinds=CLAIM_KINDS,
        guidance=GUIDANCE,
        conclusion_kinds=["safe_to_adopt", "not_safe_to_adopt"],
    )
)
from verdict.domains.base import _REGISTRY  # noqa: E402

_REGISTRY["supply_chain"] = DOMAIN  # alias
