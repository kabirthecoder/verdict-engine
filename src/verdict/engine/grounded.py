"""Grounded semantics for abstract argumentation (Dung, 1995).

Given a set of arguments and an attack relation, the grounded extension is the least
fixed point of the characteristic function F(S) = {a | a is defended by S}, where S
defends a iff every attacker of a is attacked by some member of S.

It is unique, always exists, and is the most skeptical of the standard semantics:
an argument is accepted only if it must be, given the attacks on the table. That is
exactly the property a decision system should have — it never picks one of several
self-consistent stories; it says "undecided" instead.

The implementation is the standard iterative labelling: repeatedly mark IN every
argument whose attackers are all OUT, and OUT every argument with an IN attacker,
until nothing changes. What remains is UNDEC. Runs in O(|args| * |attacks|) worst case.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class Status(StrEnum):
    IN = "in"
    OUT = "out"
    UNDEC = "undec"


@dataclass(frozen=True)
class Framework:
    arguments: frozenset[Hashable]
    attacks: frozenset[tuple[Hashable, Hashable]]  # (attacker, target)

    def __post_init__(self) -> None:
        for a, b in self.attacks:
            if a not in self.arguments or b not in self.arguments:
                raise ValueError(f"attack ({a!r}, {b!r}) references unknown argument")

    @classmethod
    def build(
        cls, arguments: Iterable[Hashable], attacks: Iterable[tuple[Hashable, Hashable]]
    ) -> Framework:
        return cls(frozenset(arguments), frozenset(attacks))


@dataclass
class Labelling:
    status: dict[Hashable, Status] = field(default_factory=dict)

    @property
    def accepted(self) -> set[Hashable]:
        return {a for a, s in self.status.items() if s is Status.IN}

    @property
    def rejected(self) -> set[Hashable]:
        return {a for a, s in self.status.items() if s is Status.OUT}

    @property
    def undecided(self) -> set[Hashable]:
        return {a for a, s in self.status.items() if s is Status.UNDEC}


def grounded(fw: Framework) -> Labelling:
    attackers: dict[Hashable, set[Hashable]] = defaultdict(set)
    targets: dict[Hashable, set[Hashable]] = defaultdict(set)
    for a, b in fw.attacks:
        attackers[b].add(a)
        targets[a].add(b)

    status: dict[Hashable, Status] = dict.fromkeys(fw.arguments, Status.UNDEC)

    changed = True
    while changed:
        changed = False
        for arg in fw.arguments:
            if status[arg] is not Status.UNDEC:
                continue
            atk = attackers.get(arg, set())
            if all(status[x] is Status.OUT for x in atk):
                status[arg] = Status.IN
                changed = True
                for t in targets.get(arg, ()):
                    if status[t] is Status.UNDEC:
                        status[t] = Status.OUT
            elif any(status[x] is Status.IN for x in atk):
                status[arg] = Status.OUT
                changed = True

    return Labelling(status)


def explain(fw: Framework, lab: Labelling, arg: Hashable) -> str:
    """One-line, human-readable reason for an argument's label."""
    attackers = [a for a, b in fw.attacks if b == arg]
    s = lab.status[arg]
    if s is Status.IN:
        if not attackers:
            return "accepted: no attacks"
        return "accepted: every attacker is rejected (" + ", ".join(map(str, attackers)) + ")"
    if s is Status.OUT:
        winners = [a for a in attackers if lab.status[a] is Status.IN]
        return "rejected: attacked by accepted " + ", ".join(map(str, winners))
    open_ = [a for a in attackers if lab.status[a] is Status.UNDEC]
    return "undecided: unresolved attack from " + ", ".join(map(str, open_))
