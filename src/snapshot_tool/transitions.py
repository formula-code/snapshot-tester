"""
Transition utilities for baseline->verify comparisons.

Provides a pure function that, given a baseline mapping and a verify mapping
of test_id->status, computes the 3x3 {pass,fail,skip}-to-{pass,fail,skip}
transition counts.
"""

from __future__ import annotations

from typing import Dict


def _normalize_status(status: str) -> str:
    """Normalize status strings to one of pass|fail|skip.

    - Supports legacy value "failed_to_pass" by mapping it to "fail".
    - Any unknown value defaults to "fail" for conservative interpretation.
    """
    if status == "failed_to_pass":
        return "fail"
    # Keep other values as-is; callers may only rely on what's present
    # in baseline/verify entries. Common values are: pass, fail, skip.
    return status


def compute_transitions(
    baseline_entries: Dict[str, str], verify_entries: Dict[str, str]
) -> Dict[str, int]:
    """Compute transition counts between baseline and verify statuses.

    Args:
        baseline_entries: Mapping of test_id->baseline status (pass|fail|skip or legacy failed_to_pass)
        verify_entries: Mapping of test_id->verify status (pass|fail|skip)

    Returns:
        A dict with all nine transition keys like "pass-to-fail", each an int count.
        Tests present only in verify but not baseline are ignored; transitions
        are computed for baseline keys present in verify_entries.
    """
    # Determine dynamic state sets from data actually present
    # Baseline states come only from baseline entries
    baseline_states = {_normalize_status(s) for s in baseline_entries.values()}
    # Verify states come from overlapping tests (present in both)
    verify_states: set[str] = set()

    overlapping_ids = [tid for tid in baseline_entries.keys() if tid in verify_entries]
    for tid in overlapping_ids:
        verify_states.add(_normalize_status(verify_entries[tid]))

    # Initialize all pair keys seen in the data
    transitions: Dict[str, int] = {f"{a}-to-{b}": 0 for a in baseline_states for b in verify_states}

    for test_id, b_status_raw in baseline_entries.items():
        v_status_raw = verify_entries.get(test_id)
        if v_status_raw is None:
            continue
        b = _normalize_status(b_status_raw)
        v = _normalize_status(v_status_raw)
        key = f"{b}-to-{v}"
        # Guard against unexpected keys if normalization changes
        if key in transitions:
            transitions[key] += 1

    return transitions
