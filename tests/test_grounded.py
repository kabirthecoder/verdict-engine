"""Textbook cases for grounded semantics."""

from verdict.engine import Framework, Status, explain, grounded


def lab(args, attacks):
    return grounded(Framework.build(args, attacks)).status


def test_unattacked_is_in():
    assert lab("a", []) == {"a": Status.IN}


def test_simple_attack():
    s = lab("ab", [("a", "b")])
    assert s == {"a": Status.IN, "b": Status.OUT}


def test_reinstatement_chain():
    # a -> b -> c : a defends c
    s = lab("abc", [("a", "b"), ("b", "c")])
    assert s == {"a": Status.IN, "b": Status.OUT, "c": Status.IN}


def test_mutual_attack_is_undecided():
    s = lab("ab", [("a", "b"), ("b", "a")])
    assert s == {"a": Status.UNDEC, "b": Status.UNDEC}


def test_undecided_propagates():
    # a <-> b, and b -> c : c cannot be accepted because b is not rejected
    s = lab("abc", [("a", "b"), ("b", "a"), ("b", "c")])
    assert s["c"] is Status.UNDEC


def test_self_attack_is_undecided_and_blocks():
    s = lab("ab", [("a", "a"), ("a", "b")])
    assert s == {"a": Status.UNDEC, "b": Status.UNDEC}


def test_odd_cycle_undecided_even_cycle_undecided():
    assert set(lab("abc", [("a", "b"), ("b", "c"), ("c", "a")]).values()) == {Status.UNDEC}
    assert set(lab("abcd", [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]).values()) == {
        Status.UNDEC
    }


def test_defended_against_cycle():
    # d -> a, a <-> b, b -> c : d kills a, so b is IN, so c is OUT
    s = lab("abcd", [("d", "a"), ("a", "b"), ("b", "a"), ("b", "c")])
    assert s == {"d": Status.IN, "a": Status.OUT, "b": Status.IN, "c": Status.OUT}


def test_unknown_argument_in_attack_rejected():
    import pytest

    with pytest.raises(ValueError):
        Framework.build("a", [("a", "z")])


def test_explain():
    fw = Framework.build("abc", [("a", "b"), ("b", "c")])
    l_ = grounded(fw)
    assert explain(fw, l_, "a") == "accepted: no attacks"
    assert explain(fw, l_, "b") == "rejected: attacked by accepted a"
    assert explain(fw, l_, "c").startswith("accepted: every attacker is rejected")


def test_scales():
    # long reinstatement chain of 5000 arguments
    n = 5000
    args = list(range(n))
    attacks = [(i, i + 1) for i in range(n - 1)]
    s = lab(args, attacks)
    assert (
        s[0] is Status.IN
        and s[1] is Status.OUT
        and s[n - 1] is (Status.IN if n % 2 else Status.OUT)
    )
