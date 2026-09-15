"""Append-only persistence for the argument graph.

SQLAlchemy Core, one table per node type, JSON columns for the flexible parts.
Works on SQLite out of the box and on Postgres by changing the URL. There are
deliberately no update or delete methods.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from verdict.graph.models import (
    Attack,
    Audit,
    Author,
    Claim,
    Evidence,
    Graph,
    Question,
    Support,
    Verdict,
)

metadata = MetaData()

questions = Table(
    "questions",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("text", Text, nullable=False),
    Column("domain", String(80), nullable=False),
    Column("context", Text, nullable=False),
    Column("status", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

claims = Table(
    "claims",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("question_id", String(40), nullable=False, index=True),
    Column("kind", String(120), nullable=False),
    Column("params", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("author", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

evidence = Table(
    "evidence",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("question_id", String(40), nullable=False, index=True),
    Column("tool", String(120), nullable=False),
    Column("args", Text, nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("output", Text, nullable=False),
    Column("output_hash", String(64), nullable=False),
    Column("summary", Text, nullable=False),
    Column("author", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

supports = Table(
    "supports",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("claim_id", String(40), nullable=False, index=True),
    Column("evidence_id", String(40), nullable=False, index=True),
    Column("quote", Text, nullable=False),
    Column("author", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

attacks = Table(
    "attacks",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("attacker_id", String(40), nullable=False, index=True),
    Column("target_id", String(40), nullable=False, index=True),
    Column("rationale", Text, nullable=False),
    Column("author", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

audits = Table(
    "audits",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("support_id", String(40), nullable=False, index=True),
    Column("ok", Boolean, nullable=False),
    Column("note", Text, nullable=False),
    Column("author", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

verdicts = Table(
    "verdicts",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("question_id", String(40), nullable=False, index=True),
    Column("round", Integer, nullable=False),
    Column("labels", Text, nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def _dump(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _author(a: Author) -> str:
    return a.model_dump_json()


class GraphStore:
    def __init__(self, url: str = "sqlite:///verdict.db") -> None:
        self.engine: Engine = create_engine(url, future=True)
        metadata.create_all(self.engine)

    # --- writes (append-only) ------------------------------------------------

    def add_question(self, q: Question) -> Question:
        with self.engine.begin() as cx:
            cx.execute(
                insert(questions).values(
                    id=q.id,
                    text=q.text,
                    domain=q.domain,
                    context=_dump(q.context),
                    status=str(q.status),
                    created_at=q.created_at,
                )
            )
        return q

    def add_claim(self, c: Claim) -> Claim:
        self._require_question(c.question_id)
        with self.engine.begin() as cx:
            cx.execute(
                insert(claims).values(
                    id=c.id,
                    question_id=c.question_id,
                    kind=c.kind,
                    params=_dump(c.params),
                    text=c.text,
                    author=_author(c.author),
                    created_at=c.created_at,
                )
            )
        return c

    def add_evidence(self, e: Evidence) -> Evidence:
        self._require_question(e.question_id)
        with self.engine.begin() as cx:
            cx.execute(
                insert(evidence).values(
                    id=e.id,
                    question_id=e.question_id,
                    tool=e.tool,
                    args=_dump(e.args),
                    ok=e.ok,
                    output=e.output,
                    output_hash=e.output_hash,
                    summary=e.summary,
                    author=_author(e.author),
                    created_at=e.created_at,
                )
            )
        return e

    def add_support(self, s: Support) -> Support:
        c = self._get_claim(s.claim_id)
        e = self._get_evidence(s.evidence_id)
        if c.question_id != e.question_id:
            raise ValueError("support must link a claim and evidence of the same question")
        with self.engine.begin() as cx:
            cx.execute(
                insert(supports).values(
                    id=s.id,
                    claim_id=s.claim_id,
                    evidence_id=s.evidence_id,
                    quote=s.quote,
                    author=_author(s.author),
                    created_at=s.created_at,
                )
            )
        return s

    def add_attack(self, a: Attack) -> Attack:
        src = self._get_claim(a.attacker_id)
        dst = self._get_claim(a.target_id)
        if src.question_id != dst.question_id:
            raise ValueError("attack must link two claims of the same question")
        with self.engine.begin() as cx:
            cx.execute(
                insert(attacks).values(
                    id=a.id,
                    attacker_id=a.attacker_id,
                    target_id=a.target_id,
                    rationale=a.rationale,
                    author=_author(a.author),
                    created_at=a.created_at,
                )
            )
        return a

    def add_audit(self, u: Audit) -> Audit:
        self._get_support(u.support_id)
        with self.engine.begin() as cx:
            cx.execute(
                insert(audits).values(
                    id=u.id,
                    support_id=u.support_id,
                    ok=u.ok,
                    note=u.note,
                    author=_author(u.author),
                    created_at=u.created_at,
                )
            )
        return u

    def add_verdict(self, v: Verdict) -> Verdict:
        self._require_question(v.question_id)
        with self.engine.begin() as cx:
            cx.execute(
                insert(verdicts).values(
                    id=v.id,
                    question_id=v.question_id,
                    round=v.round,
                    labels=_dump({k: str(lbl) for k, lbl in v.labels.items()}),
                    reason=v.reason,
                    created_at=v.created_at,
                )
            )
        return v

    # --- reads ---------------------------------------------------------------

    def list_questions(self) -> list[Question]:
        with self.engine.connect() as cx:
            rows = cx.execute(select(questions).order_by(questions.c.created_at)).mappings()
            return [self._q(r) for r in rows]

    def load(self, question_id: str) -> Graph:
        with self.engine.connect() as cx:
            qrow = (
                cx.execute(select(questions).where(questions.c.id == question_id))
                .mappings()
                .first()
            )
            if qrow is None:
                raise KeyError(question_id)
            crow = (
                cx.execute(select(claims).where(claims.c.question_id == question_id))
                .mappings()
                .all()
            )
            erow = (
                cx.execute(select(evidence).where(evidence.c.question_id == question_id))
                .mappings()
                .all()
            )
            claim_ids = [r["id"] for r in crow]
            srow = (
                cx.execute(select(supports).where(supports.c.claim_id.in_(claim_ids)))
                .mappings()
                .all()
                if claim_ids
                else []
            )
            arow = (
                cx.execute(select(attacks).where(attacks.c.target_id.in_(claim_ids)))
                .mappings()
                .all()
                if claim_ids
                else []
            )
            support_ids = [r["id"] for r in srow]
            urow = (
                cx.execute(select(audits).where(audits.c.support_id.in_(support_ids)))
                .mappings()
                .all()
                if support_ids
                else []
            )
            vrow = (
                cx.execute(select(verdicts).where(verdicts.c.question_id == question_id))
                .mappings()
                .all()
            )

        return Graph(
            question=self._q(qrow),
            claims=sorted(
                (
                    Claim(
                        id=r["id"],
                        question_id=r["question_id"],
                        kind=r["kind"],
                        params=json.loads(r["params"]),
                        text=r["text"],
                        author=Author.model_validate_json(r["author"]),
                        created_at=_ts(r["created_at"]),
                    )
                    for r in crow
                ),
                key=lambda x: x.created_at,
            ),
            evidence=sorted(
                (
                    Evidence(
                        id=r["id"],
                        question_id=r["question_id"],
                        tool=r["tool"],
                        args=json.loads(r["args"]),
                        ok=r["ok"],
                        output=r["output"],
                        output_hash=r["output_hash"],
                        summary=r["summary"],
                        author=Author.model_validate_json(r["author"]),
                        created_at=_ts(r["created_at"]),
                    )
                    for r in erow
                ),
                key=lambda x: x.created_at,
            ),
            supports=[
                Support(
                    id=r["id"],
                    claim_id=r["claim_id"],
                    evidence_id=r["evidence_id"],
                    quote=r["quote"],
                    author=Author.model_validate_json(r["author"]),
                    created_at=_ts(r["created_at"]),
                )
                for r in srow
            ],
            attacks=[
                Attack(
                    id=r["id"],
                    attacker_id=r["attacker_id"],
                    target_id=r["target_id"],
                    rationale=r["rationale"],
                    author=Author.model_validate_json(r["author"]),
                    created_at=_ts(r["created_at"]),
                )
                for r in arow
            ],
            audits=[
                Audit(
                    id=r["id"],
                    support_id=r["support_id"],
                    ok=r["ok"],
                    note=r["note"],
                    author=Author.model_validate_json(r["author"]),
                    created_at=_ts(r["created_at"]),
                )
                for r in urow
            ],
            verdicts=[
                Verdict(
                    id=r["id"],
                    question_id=r["question_id"],
                    round=r["round"],
                    labels=json.loads(r["labels"]),
                    reason=r["reason"],
                    created_at=_ts(r["created_at"]),
                )
                for r in vrow
            ],
        )

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _q(r: Any) -> Question:
        return Question(
            id=r["id"],
            text=r["text"],
            domain=r["domain"],
            context=json.loads(r["context"]),
            status=r["status"],
            created_at=_ts(r["created_at"]),
        )

    def _require_question(self, question_id: str) -> None:
        with self.engine.connect() as cx:
            if (
                cx.execute(select(questions.c.id).where(questions.c.id == question_id)).first()
                is None
            ):
                raise KeyError(f"unknown question {question_id}")

    def _get_claim(self, claim_id: str) -> Claim:
        with self.engine.connect() as cx:
            r = cx.execute(select(claims).where(claims.c.id == claim_id)).mappings().first()
        if r is None:
            raise KeyError(f"unknown claim {claim_id}")
        return Claim(
            id=r["id"],
            question_id=r["question_id"],
            kind=r["kind"],
            params=json.loads(r["params"]),
            text=r["text"],
            author=Author.model_validate_json(r["author"]),
            created_at=_ts(r["created_at"]),
        )

    def _get_evidence(self, evidence_id: str) -> Evidence:
        with self.engine.connect() as cx:
            r = cx.execute(select(evidence).where(evidence.c.id == evidence_id)).mappings().first()
        if r is None:
            raise KeyError(f"unknown evidence {evidence_id}")
        return Evidence(
            id=r["id"],
            question_id=r["question_id"],
            tool=r["tool"],
            args=json.loads(r["args"]),
            ok=r["ok"],
            output=r["output"],
            output_hash=r["output_hash"],
            summary=r["summary"],
            author=Author.model_validate_json(r["author"]),
            created_at=_ts(r["created_at"]),
        )

    def _get_support(self, support_id: str) -> None:
        with self.engine.connect() as cx:
            if cx.execute(select(supports.c.id).where(supports.c.id == support_id)).first() is None:
                raise KeyError(f"unknown support {support_id}")


def _ts(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)
