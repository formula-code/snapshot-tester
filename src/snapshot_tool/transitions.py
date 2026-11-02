"""Count baseline-to-verify status transitions such as ``pass-to-fail``."""

from __future__ import annotations

from typing import Dict


def _normalize_status(status: str) -> str:
    """Map "failed_to_pass" to "fail"; return any other status unchanged."""
    if status == "failed_to_pass":
        return "fail"
    return status


def compute_transitions(
    baseline_entries: Dict[str, str], verify_entries: Dict[str, str]
) -> Dict[str, int]:
    """Compute transition counts between baseline and verify statuses.

    Args:
        baseline_entries: Mapping of test_id->baseline status (pass|fail|skip or failed_to_pass)
        verify_entries: Mapping of test_id->verify status (pass|fail|skip)

    Returns:
        A dict of "<baseline>-to-<verify>" counts. Keys exist for every pair of a baseline status
        and a status seen in verify, so absent pairs can be missing. Only test_ids in both
        mappings are counted.
    """
    baseline_states = {_normalize_status(s) for s in baseline_entries.values()}
    verify_states: set[str] = set()

    overlapping_ids = [tid for tid in baseline_entries.keys() if tid in verify_entries]
    for tid in overlapping_ids:
        verify_states.add(_normalize_status(verify_entries[tid]))

    transitions: Dict[str, int] = {f"{a}-to-{b}": 0 for a in baseline_states for b in verify_states}

    for test_id, b_status_raw in baseline_entries.items():
        v_status_raw = verify_entries.get(test_id)
        if v_status_raw is None:
            continue
        b = _normalize_status(b_status_raw)
        v = _normalize_status(v_status_raw)
        key = f"{b}-to-{v}"
        if key in transitions:
            transitions[key] += 1

    return transitions
