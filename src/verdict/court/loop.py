"""The court loop: fire roles from graph state until a fixed point or the budget ends.

There is no orchestrator deciding who speaks next. Each round:
  1. no claims yet            -> claimant opens the case
  2. claim without evidence,
     or a claim that attacks  -> investigator (finds support, or attacks it if evidence contradicts)
  3. support not yet audited  -> auditor (verbatim check, then entailment check)
  4. provisional verdict; accepted claims the adversary has not yet examined -> adversary
A round that writes nothing to the graph is a fixed point. Then the verdict is frozen.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from verdict.config import Settings
from verdict.court import roles
from verdict.domains.base import Domain
from verdict.engine.verdict import compute_verdict
from verdict.graph.models import (
    Attack,
    Audit,
    Author,
    Claim,
    Graph,
    Label,
    Question,
    Role,
    Support,
    Verdict,
    new_id,
)
from verdict.graph.store import GraphStore
from verdict.llm.client import LLM
from verdict.llm.episode import run_episode
from verdict.tools.registry import ToolRuntime

Event = Callable[[str], None]


@dataclass
class CourtStats:
    rounds: int = 0
    episodes: int = 0
    tool_calls: int = 0
    claims: int = 0
    supports: int = 0
    attacks: int = 0
    audits_failed: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = ""
    log: list[str] = field(default_factory=list)


class Court:
    def __init__(
        self,
        store: GraphStore,
        llm: LLM,
        domain: Domain,
        settings: Settings,
        on_event: Event | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.domain = domain
        self.settings = settings
        self.stats = CourtStats()
        self._on_event = on_event or (lambda _: None)

    # ------------------------------------------------------------------ public

    def run(self, question: Question) -> Verdict:
        s = self.settings
        runtime = ToolRuntime.for_question(self.store, question.id, self.domain.tools)
        investigated: set[str] = set()
        adversary_seen: set[str] = set()
        t0 = time.monotonic()
        reason = "max_rounds"

        for rnd in range(1, s.max_rounds + 1):
            self.stats.rounds = rnd
            wrote = 0
            self._event(f"round {rnd}")

            g = self.store.load(question.id)
            if not g.claims:
                wrote += self._claimant(runtime, g)
                g = self.store.load(question.id)

            targets = [
                c.id
                for c in g.claims
                if c.id not in investigated and (not g.is_supported(c.id) or g.attacks_by(c.id))
            ]
            for batch in _chunks(targets, 5):
                investigated.update(batch)
                wrote += self._investigator(runtime, self.store.load(question.id), batch)
                if self._over_budget(runtime, t0):
                    break

            wrote += self._audit(self.store.load(question.id))

            g = self.store.load(question.id)
            provisional = compute_verdict(g, rnd, "provisional")
            accepted = [
                cid
                for cid, lab in provisional.labels.items()
                if lab is Label.ACCEPTED and cid not in adversary_seen
            ]
            for batch in _chunks(accepted, 6):
                adversary_seen.update(batch)
                wrote += self._adversary(runtime, self.store.load(question.id), provisional, batch)
                if self._over_budget(runtime, t0):
                    break

            # attacks create new claims; audit them in the same round so the next
            # round's provisional verdict already reflects them
            wrote += self._audit(self.store.load(question.id))

            if wrote == 0:
                reason = "fixed_point"
                break
            if self._over_budget(runtime, t0):
                reason = "budget"
                break

        g = self.store.load(question.id)
        verdict = compute_verdict(g, self.stats.rounds, reason)
        self.store.add_verdict(verdict)
        self.stats.stop_reason = reason
        self.stats.tool_calls = runtime.calls
        self.stats.seconds = time.monotonic() - t0
        self.stats.prompt_tokens = self.llm.prompt_tokens
        self.stats.completion_tokens = self.llm.completion_tokens
        self._event(f"verdict: {reason} after {self.stats.rounds} rounds")
        return verdict

    # ------------------------------------------------------------------- roles

    def _claimant(self, runtime: ToolRuntime, g: Graph) -> int:
        spec = roles.claimant_spec(self.domain, g)
        out = self._episode(runtime, spec)
        if out is None:
            return 0
        author = self._author(Role.CLAIMANT)
        n = 0
        for pc in out.claims:
            cid = self._record_claim(pc, author, g.question.id)
            n += 1 if cid else 0
        return n

    def _investigator(self, runtime: ToolRuntime, g: Graph, claim_ids: list[str]) -> int:
        spec = roles.investigator_spec(self.domain, g, claim_ids)
        out = self._episode(runtime, spec)
        if out is None:
            return 0
        author = self._author(Role.INVESTIGATOR)
        n = 0
        fresh = self.store.load(g.question.id)
        known_evidence = {e.id for e in fresh.evidence}
        known_claims = {c.id for c in fresh.claims}
        for item in out.new_citations:
            cid, eid, quote = item.get("claim_id"), item.get("evidence_id"), item.get("quote", "")
            if cid in known_claims and eid in known_evidence and quote:
                self.store.add_support(
                    Support(claim_id=cid, evidence_id=eid, quote=quote, author=author)
                )
                self.stats.supports += 1
                n += 1
        for atk in out.attacks:
            n += self._record_attack(atk, author, g.question.id)
        return n

    def _adversary(
        self, runtime: ToolRuntime, g: Graph, provisional: Verdict, claim_ids: list[str]
    ) -> int:
        spec = roles.adversary_spec(self.domain, g, provisional.labels, claim_ids)
        out = self._episode(runtime, spec)
        if out is None:
            return 0
        author = self._author(Role.ADVERSARY)
        n = 0
        for atk in out.attacks:
            n += self._record_attack(atk, author, g.question.id)
        if out.gave_up:
            self._event(f"adversary gave up on {len(out.gave_up)} claim(s)")
        return n

    def _audit(self, g: Graph) -> int:
        n = 0
        for sup in g.supports:
            if g.audits_for(sup.id):
                continue
            ev = g.evidence_by_id(sup.evidence_id)
            claim = g.claim(sup.claim_id)
            author = self._author(Role.AUDITOR)
            if not _verbatim(sup.quote, ev.output):
                self.store.add_audit(
                    Audit(support_id=sup.id, ok=False, note="quote not verbatim", author=author)
                )
                self.stats.audits_failed += 1
                self._event(f"audit FAIL (not verbatim) on {claim.id}")
                n += 1
                continue
            if not ev.ok:
                self.store.add_audit(
                    Audit(
                        support_id=sup.id, ok=False, note="cites a failed tool call", author=author
                    )
                )
                self.stats.audits_failed += 1
                n += 1
                continue
            spec = roles.auditor_spec(claim.text, sup.quote, ev.tool, ev.args)
            empty = ToolRuntime.for_question(self.store, g.question.id, [])
            res = run_episode(self.llm, empty, spec.system, spec.task, spec.output, max_steps=2)
            self.stats.episodes += 1
            if res.output is None:
                ok, note = True, "entailment check unavailable; verbatim check passed"
            else:
                ok, note = res.output.entailed, res.output.note
            self.store.add_audit(Audit(support_id=sup.id, ok=ok, note=note, author=author))
            if not ok:
                self.stats.audits_failed += 1
                self._event(f"audit FAIL (not entailed) on {claim.id}: {note}")
            n += 1
        return n

    # ----------------------------------------------------------------- helpers

    def _episode(self, runtime: ToolRuntime, spec: roles.RoleSpec):
        self._event(f"{spec.role} episode")
        res = run_episode(
            self.llm,
            runtime,
            spec.system,
            spec.task,
            spec.output,
            max_steps=self.settings.max_episode_steps,
        )
        self.stats.episodes += 1
        if res.output is None:
            self._event(f"{spec.role} produced no result ({res.stopped})")
        return res.output

    def _record_claim(self, pc: roles.ProposedClaim, author: Author, qid: str) -> str | None:
        if pc.kind not in self.domain.kind_names():
            self._event(f"dropped claim with unknown kind {pc.kind!r}")
            return None
        g = self.store.load(qid)
        known_evidence = {e.id for e in g.evidence}
        existing = next(
            (
                c
                for c in g.claims
                if c.kind == pc.kind and c.params == pc.params and c.text == pc.text
            ),
            None,
        )
        claim = existing or self.store.add_claim(
            Claim(question_id=qid, kind=pc.kind, params=pc.params, text=pc.text, author=author)
        )
        if existing is None:
            self.stats.claims += 1
        for cit in pc.citations:
            if cit.evidence_id in known_evidence and cit.quote.strip():
                self.store.add_support(
                    Support(
                        claim_id=claim.id,
                        evidence_id=cit.evidence_id,
                        quote=cit.quote,
                        author=author,
                    )
                )
                self.stats.supports += 1
            else:
                self._event(f"dropped citation to unknown evidence {cit.evidence_id!r}")
        return claim.id

    def _record_attack(self, atk: roles.ProposedAttack, author: Author, qid: str) -> int:
        g = self.store.load(qid)
        if atk.target_claim_id not in {c.id for c in g.claims}:
            self._event(f"dropped attack on unknown claim {atk.target_claim_id!r}")
            return 0
        attacker_id = self._record_claim(atk.claim, author, qid)
        if attacker_id is None or attacker_id == atk.target_claim_id:
            return 0
        self.store.add_attack(
            Attack(
                attacker_id=attacker_id,
                target_id=atk.target_claim_id,
                rationale=atk.rationale,
                author=author,
            )
        )
        self.stats.attacks += 1
        return 1

    def _author(self, role: Role) -> Author:
        return Author(role=role, run_id=new_id("run"), name=self.llm.model)

    def _over_budget(self, runtime: ToolRuntime, t0: float) -> bool:
        s = self.settings
        return runtime.calls >= s.max_tool_calls or (time.monotonic() - t0) >= s.max_wall_clock_s

    def _event(self, msg: str) -> None:
        self.stats.log.append(msg)
        self._on_event(msg)


def _chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


_WS = re.compile(r"\s+")


def _verbatim(quote: str, output: str) -> bool:
    q = _WS.sub(" ", quote).strip().lower()
    o = _WS.sub(" ", output).lower()
    return bool(q) and q in o
