from __future__ import annotations

import random

from snapshot_tool.transitions import compute_transitions


def test_compute_transitions_basic_identity():
    baseline = {
        "A": "pass",
        "B": "fail",
        "C": "skip",
    }
    verify = {
        "A": "pass",
        "B": "fail",
        "C": "fail",
    }
    out = compute_transitions(baseline, verify)
    # Expect all nine keys initialized with 3x3 states
    assert out.get("pass-to-pass", 0) == 1
    assert out.get("fail-to-fail", 0) == 1
    assert out.get("skip-to-skip", 0) == 0
    assert out.get("skip-to-fail", 0) == 1
    # Others should be zero
    zeros = [
        "pass-to-fail",
        "pass-to-skip",
        "fail-to-pass",
        "fail-to-skip",
        "skip-to-pass",
    ]
    for k in zeros:
        assert out.get(k, 0) == 0


def test_compute_transitions_mixed_and_legacy():
    baseline = {
        "A": "pass",
        "B": "fail",
        "C": "failed_to_pass",  # legacy
        "D": "skip",
        "E": "pass",
    }
    verify = {
        "A": "fail",  # pass->fail
        "B": "pass",  # fail->pass
        "C": "pass",  # legacy fail->pass
        "D": "fail",  # skip->fail
        "F": "pass",  # new test not in baseline; ignored
    }
    out = compute_transitions(baseline, verify)
    assert out.get("pass-to-fail", 0) == 1
    assert out.get("fail-to-pass", 0) == 2
    assert out.get("skip-to-fail", 0) == 1
    # Everything else zero
    expected_zero = [
        "pass-to-pass",
        "pass-to-skip",
        "fail-to-fail",
        "fail-to-skip",
        "skip-to-pass",
        "skip-to-skip",
    ]
    for k in expected_zero:
        assert out.get(k, 0) == 0


def test_compute_transitions_randomized():
    random.seed(42)
    # Create a set of test ids
    n = 200
    ids = [f"T{i:04d}" for i in range(n)]

    # Allow baseline to produce a legacy value sometimes
    base_choices = ["pass", "fail", "skip", "failed_to_pass"]
    verify_choices = ["pass", "fail", "skip"]

    baseline = {tid: random.choice(base_choices) for tid in ids}
    verify = {tid: random.choice(verify_choices) for tid in ids}

    # Compute via library
    out = compute_transitions(baseline, verify)

    # Build expected using the same normalization rules
    def norm(s: str) -> str:
        return "fail" if s == "failed_to_pass" else s

    baseline_states = set(norm(s) for s in baseline.values())
    verify_states = set(norm(verify[tid]) for tid in ids)

    expected = {f"{a}-to-{b}": 0 for a in baseline_states for b in verify_states}
    for tid in ids:
        a = norm(baseline[tid])
        b = norm(verify[tid])
        expected[f"{a}-to-{b}"] += 1

    # Totals match number of ids (all overlap)
    assert sum(out.values()) == n
    assert sum(expected.values()) == n
    # Exact dictionary match
    assert out == expected
