"""Argument graph node types.

Everything an agent or a human does in the court ends up as one of these records.
The graph is append-only: there are no update or delete operations anywhere in the
codebase. A retraction is a new attack, a correction is a new claim.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    CLAIMANT = "claimant"
    INVESTIGATOR = "investigator"
    ADVERSARY = "adversary"
    AUDITOR = "auditor"
    HUMAN = "human"
    RUNTIME = "runtime"  # the tool runtime itself; the only author allowed to write Evidence


class Author(BaseModel):
    role: Role
    run_id: str | None = None  # agent episode id, None for humans
    name: str | None = None  # human user id / handle, or model name for agents

    def __str__(self) -> str:
        who = self.name or self.run_id or "?"
        return f"{self.role}:{who}"


class QuestionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class Question(BaseModel):
    id: str = Field(default_factory=lambda: new_id("q"))
    text: str
    domain: str
    context: dict[str, Any] = Field(default_factory=dict)
    status: QuestionStatus = QuestionStatus.OPEN
    created_at: datetime = Field(default_factory=now)


class Claim(BaseModel):
    """A proposition somebody asserts about the question.

    `kind` and `params` come from the domain vocabulary, so claims are machine-comparable
    (two agents asserting `vulnerable(requests, 2.28.0, GHSA-...)` produce the same key).
    `text` is the human-readable form.
    """

    id: str = Field(default_factory=lambda: new_id("c"))
    question_id: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    text: str
    author: Author
    created_at: datetime = Field(default_factory=now)

    @property
    def key(self) -> str:
        items = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.kind}({items})"


class Evidence(BaseModel):
    """The recorded result of one real tool call. Written only by the runtime."""

    id: str = Field(default_factory=lambda: new_id("e"))
    question_id: str
    tool: str
    args: dict[str, Any]
    ok: bool
    output: str  # raw tool output, serialized; this is what the auditor reads
    output_hash: str = ""
    summary: str = ""  # short, produced by the tool itself (not by an LLM)
    author: Author
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="after")
    def _seal(self) -> Evidence:
        if self.author.role != Role.RUNTIME:
            raise ValueError("Evidence can only be authored by the tool runtime")
        digest = hashlib.sha256(self.output.encode("utf-8")).hexdigest()
        if self.output_hash and self.output_hash != digest:
            raise ValueError("Evidence output_hash does not match output")
        object.__setattr__(self, "output_hash", digest)
        return self


class Support(BaseModel):
    """Evidence E backs claim C. `quote` must occur verbatim in E.output (the auditor checks)."""

    id: str = Field(default_factory=lambda: new_id("s"))
    claim_id: str
    evidence_id: str
    quote: str
    author: Author
    created_at: datetime = Field(default_factory=now)


class Attack(BaseModel):
    """Claim `attacker` undermines claim `target`. The attacker is itself a claim, so it can
    be attacked back and needs its own support to count."""

    id: str = Field(default_factory=lambda: new_id("a"))
    attacker_id: str
    target_id: str
    rationale: str
    author: Author
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="after")
    def _no_self_attack(self) -> Attack:
        if self.attacker_id == self.target_id:
            raise ValueError("a claim cannot attack itself")
        return self


class Audit(BaseModel):
    """Record that a support edge was checked against the raw evidence."""

    id: str = Field(default_factory=lambda: new_id("u"))
    support_id: str
    ok: bool
    note: str = ""
    author: Author
    created_at: datetime = Field(default_factory=now)


class Label(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNDECIDED = "undecided"
    UNSUPPORTED = "unsupported"


class Verdict(BaseModel):
    id: str = Field(default_factory=lambda: new_id("v"))
    question_id: str
    round: int
    labels: dict[str, Label]  # claim id -> label
    reason: str = ""  # why the loop stopped: fixed_point | budget | manual
    created_at: datetime = Field(default_factory=now)


class Graph(BaseModel):
    """Immutable snapshot of everything recorded for one question."""

    question: Question
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    supports: list[Support] = Field(default_factory=list)
    attacks: list[Attack] = Field(default_factory=list)
    audits: list[Audit] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)

    # --- convenience views -------------------------------------------------

    def claim(self, claim_id: str) -> Claim:
        for c in self.claims:
            if c.id == claim_id:
                return c
        raise KeyError(claim_id)

    def evidence_by_id(self, evidence_id: str) -> Evidence:
        for e in self.evidence:
            if e.id == evidence_id:
                return e
        raise KeyError(evidence_id)

    def supports_for(self, claim_id: str) -> list[Support]:
        return [s for s in self.supports if s.claim_id == claim_id]

    def audits_for(self, support_id: str) -> list[Audit]:
        return [a for a in self.audits if a.support_id == support_id]

    def attacks_on(self, claim_id: str) -> list[Attack]:
        return [a for a in self.attacks if a.target_id == claim_id]

    def attacks_by(self, claim_id: str) -> list[Attack]:
        return [a for a in self.attacks if a.attacker_id == claim_id]

    def is_supported(self, claim_id: str) -> bool:
        """A claim counts as supported if at least one support edge has not failed audit."""
        for s in self.supports_for(claim_id):
            audits = self.audits_for(s.id)
            if not audits or all(a.ok for a in audits):
                return True
        return False

    @property
    def latest_verdict(self) -> Verdict | None:
        return max(self.verdicts, key=lambda v: v.round, default=None)
