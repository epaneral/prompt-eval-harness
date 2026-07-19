"""Gate suite: evaluate_gates, failing_case_ids, refresh_cases.

Design decisions under test are documented in harness/gates.py's module
docstring. The rules exercised here:

  * every gate is AND-ed: schema validity == 100%, per-type recall == 1.0,
    trap false-extraction rate == 0% over schema-valid traps, and no metric
    below the pinned baseline;
  * undefined metrics (None) FAIL their gate -- "no data" is never a pass,
    which is what stops a zero-valid-output run from sailing through the
    trap gate on a flattering 0%;
  * failing_case_ids is the resample set: the union of schema-invalid and
    non-exact-match case ids, sorted and de-duplicated;
  * refresh_cases deletes exactly the named cases' cache entries.

No network is ever touched: no real anthropic client is constructed (the
client is always None) and harness.runner._call_api is monkeypatched with a
scripted stub everywhere a call would otherwise happen. RunResult and Report
objects are produced by driving the real run_corpus against a small fake
corpus in tmp_path, and baselines are produced the way write_baseline shapes
them (built by calling write_baseline and json.loads-ing the file).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import harness.runner as runner
from harness.gates import (
    GateResult,
    evaluate_gates,
    failing_case_ids,
    refresh_cases,
)
from harness.grader import INDICATOR_TYPES
from harness.runner import (
    CachedResponse,
    RunResult,
    cache_path,
    prompt_sha256,
    run_corpus,
    write_baseline,
)

# A positive case whose expected extraction populates EVERY indicator type,
# paired with a trap that expects nothing. The positive carrying all four
# types is what makes every recall_floor_<type> gate DEFINED (recall over a
# nonzero denominator) rather than fail-closed on a None. The values are
# already in canonical defanged-at-rest form, so a stub echoing them back is
# an exact match through the grader.
_FULL_EXTRACTION = {
    "ipv4": ["45[.]129[.]14[.]203"],
    "domains": ["evil[.]com"],
    "urls": ["hxxp://evil[.]com/payload"],
    "hashes": ["d41d8cd98f00b204e9800998ecf8427e"],
}
_EMPTY_EXTRACTION = {"ipv4": [], "domains": [], "urls": [], "hashes": []}

# Ceiling response for the positive case (all types) and the clean trap
# response (nothing extracted). compact separators keep them byte-stable.
_FULL_JSON = json.dumps(_FULL_EXTRACTION)
_EMPTY_JSON = json.dumps(_EMPTY_EXTRACTION)

# A positive response that DROPS the domain -> one recall miss on `domains`.
_DEGRADED_EXTRACTION = {
    "ipv4": ["45[.]129[.]14[.]203"],
    "domains": [],
    "urls": ["hxxp://evil[.]com/payload"],
    "hashes": ["d41d8cd98f00b204e9800998ecf8427e"],
}
_DEGRADED_JSON = json.dumps(_DEGRADED_EXTRACTION)

# A trap response that leaks one indicator not present in its (empty)
# expected -> a false extraction the trap gate must catch.
_TRAP_LEAK_JSON = json.dumps({"ipv4": [], "domains": ["leak[.]net"], "urls": [], "hashes": []})

# Distinct input text per case so a stub can key its response on input_text.
_POS_INPUT = "Positive sample. Full IOC set present.\n"
_TRAP_INPUT = "Trap sample. Near-miss, nothing to extract.\n"

_MANIFEST = """\
cases:
- id: full_01_pos
  category: full
  role: positive
  twin_id: full_01_trap
  input: inputs/full_01_pos.txt
  expected: expected/full_01_pos.json
- id: full_01_trap
  category: full
  role: trap
  twin_id: full_01_pos
  input: inputs/full_01_trap.txt
  expected: expected/full_01_trap.json
"""


def _build_corpus(root: Path) -> Path:
    """Write a minimal two-case corpus (all-type positive + empty trap)."""
    corpus = root / "corpus"
    (corpus / "inputs").mkdir(parents=True)
    (corpus / "expected").mkdir(parents=True)
    (corpus / "manifest.yaml").write_text(_MANIFEST, encoding="utf-8")

    (corpus / "inputs" / "full_01_pos.txt").write_text(_POS_INPUT, encoding="utf-8")
    (corpus / "inputs" / "full_01_trap.txt").write_text(_TRAP_INPUT, encoding="utf-8")
    (corpus / "expected" / "full_01_pos.json").write_text(
        json.dumps(_FULL_EXTRACTION, indent=2), encoding="utf-8"
    )
    (corpus / "expected" / "full_01_trap.json").write_text(
        json.dumps(_EMPTY_EXTRACTION, indent=2), encoding="utf-8"
    )
    return corpus


class _KeyedCall:
    """Stub for _call_api: returns a response chosen by the case input text.

    Both corpus inputs are distinct, so the response for each case is selected
    deterministically without relying on manifest ordering. Any input not in
    the map raises, so a mis-wired test fails loudly rather than silently.
    """

    def __init__(self, by_input: dict[str, str]) -> None:
        self._by_input = by_input
        self.calls = 0

    def __call__(
        self, client: object, model: str, system_prompt: str, input_text: str
    ) -> CachedResponse:
        self.calls += 1
        return CachedResponse(self._by_input[input_text], input_tokens=100, output_tokens=40)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pos_response: str,
    trap_response: str,
    *,
    model: str = "claude-sonnet-4-6",
    prompt_text: str = "prompt v1",
) -> RunResult:
    """Drive run_corpus against the fake corpus with per-case scripted output."""
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    stub = _KeyedCall({_POS_INPUT: pos_response, _TRAP_INPUT: trap_response})
    monkeypatch.setattr(runner, "_call_api", stub)
    return run_corpus(None, model, prompt_text, corpus)


def _baseline_from(
    tmp_path: Path,
    result: RunResult,
    *,
    model: str = "claude-sonnet-4-6",
    prompt_text: str = "prompt v1",
) -> dict:
    """Pin `result` via write_baseline and read the file back (its exact shape).

    Reading the file back (rather than hand-shaping a dict) keeps the baseline
    exactly as gates.py will encounter it on disk, including the nested
    asdict(report) structure the no-regression gate reads. BASELINES_DIR is
    redirected under tmp_path and restored so nothing leaks between tests.
    """
    prompt_path = tmp_path / "v1.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    original = runner.BASELINES_DIR
    runner.BASELINES_DIR = tmp_path / "baselines"
    try:
        out = write_baseline(prompt_path, prompt_text, model, result)
    finally:
        runner.BASELINES_DIR = original
    return json.loads(out.read_text(encoding="utf-8"))


def _gates_by_name(gates: list[GateResult]) -> dict[str, GateResult]:
    return {g.name: g for g in gates}


# --- 1. all-ceiling run vs its own baseline: every gate passes --------------


def test_all_ceiling_run_passes_every_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run(monkeypatch, tmp_path, _FULL_JSON, _EMPTY_JSON)
    baseline = _baseline_from(tmp_path, result)

    gates = evaluate_gates(result, baseline)
    by_name = _gates_by_name(gates)

    # The full set of gate names must be present.
    expected_names = {"schema_validity_100", "trap_ceiling", "no_regression_vs_baseline"}
    expected_names.update(f"recall_floor_{t}" for t in INDICATOR_TYPES)
    assert set(by_name) == expected_names

    # And every one passes.
    assert all(g.passed for g in gates), [
        (g.name, g.detail) for g in gates if not g.passed
    ]


# --- 2. schema failure: invalid output fails validity + affected recall -----


def test_schema_failure_fails_validity_and_recall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The positive case returns prose (schema-invalid); the trap stays clean.
    prose = "Sure! I found evil[.]com and a hash in the text."
    result = _run(monkeypatch, tmp_path, prose, _EMPTY_JSON)
    baseline = _baseline_from(tmp_path, result)

    by_name = _gates_by_name(evaluate_gates(result, baseline))

    # schema_validity_100 fails, and its detail names the offending case id.
    validity = by_name["schema_validity_100"]
    assert validity.passed is False
    assert "full_01_pos" in validity.detail

    # Invalid graded as empty -> the positive case (which carried every
    # expected indicator) now misses them all, so each recall floor fails.
    for t in INDICATOR_TYPES:
        assert by_name[f"recall_floor_{t}"].passed is False, t


# --- 3. trap leak: an extra indicator on a trap fails trap_ceiling ----------


def test_trap_leak_fails_trap_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Positive stays at ceiling; the trap leaks a domain not in its expected.
    result = _run(monkeypatch, tmp_path, _FULL_JSON, _TRAP_LEAK_JSON)
    baseline = _baseline_from(tmp_path, result)

    trap_gate = _gates_by_name(evaluate_gates(result, baseline))["trap_ceiling"]
    assert trap_gate.passed is False
    # The trap rate is defined (there IS a schema-valid trap) but nonzero, and
    # the detail names the offending trap case.
    assert result.trap_fp_case_rate_valid == 1.0
    assert "full_01_trap" in trap_gate.detail


# --- 4. fail-closed on undefined trap rate: no valid traps at all -----------


def test_trap_ceiling_fails_closed_when_no_valid_traps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every response is prose -> zero schema-valid outputs, so there is no
    # schema-valid trap and the trap rate is undefined (None). The gate must
    # FAIL closed, never report a flattering passing 0%.
    prose = "No JSON here, just narration about evil[.]com."
    result = _run(monkeypatch, tmp_path, prose, prose)
    baseline = _baseline_from(tmp_path, result)

    trap_gate = _gates_by_name(evaluate_gates(result, baseline))["trap_ceiling"]
    assert result.trap_fp_case_rate_valid is None
    assert result.n_valid_traps == 0
    assert trap_gate.passed is False
    assert "no schema-valid trap" in trap_gate.detail


# --- 5. no_regression_vs_baseline: degraded vs ceiling vs vs-itself ---------


def test_no_regression_fails_against_ceiling_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A ceiling baseline (all metrics at ceiling), pinned from a clean run.
    ceiling = _run(monkeypatch, tmp_path, _FULL_JSON, _EMPTY_JSON)
    ceiling_baseline = _baseline_from(tmp_path, ceiling)

    # A fresh degraded run in its own tmp corpus: the positive drops one
    # domain, so domains.recall falls below the ceiling baseline.
    degraded = _run(monkeypatch, tmp_path / "b", _DEGRADED_JSON, _EMPTY_JSON)

    gate = _gates_by_name(evaluate_gates(degraded, ceiling_baseline))[
        "no_regression_vs_baseline"
    ]
    assert gate.passed is False
    # The detail lists at least one concrete metric comparison naming the
    # regressed indicator type.
    assert "domains" in gate.detail
    assert "<" in gate.detail


def test_no_regression_passes_against_baseline_from_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The SAME degraded result compared to a baseline pinned FROM ITSELF: every
    # metric equals its own baseline, so no_regression passes (>= is satisfied).
    degraded = _run(monkeypatch, tmp_path, _DEGRADED_JSON, _EMPTY_JSON)
    self_baseline = _baseline_from(tmp_path, degraded)

    gate = _gates_by_name(evaluate_gates(degraded, self_baseline))[
        "no_regression_vs_baseline"
    ]
    assert gate.passed is True
    assert gate.detail == "no metric below baseline"


# --- 6. failing_case_ids: sorted, de-duplicated union -----------------------


def test_failing_case_ids_union_sorted_deduped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Positive -> prose (schema-invalid AND non-exact-match, so it is in BOTH
    # contributing sets); trap -> leaks a domain (valid but non-exact-match).
    prose = "Prose, not JSON."
    result = _run(monkeypatch, tmp_path, prose, _TRAP_LEAK_JSON)

    ids = failing_case_ids(result)

    # Union of schema-invalid and non-exact-match ids, sorted, no duplicates.
    assert ids == ["full_01_pos", "full_01_trap"]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    # The positive appears once despite being in both contributing sets.
    assert ids.count("full_01_pos") == 1


# --- 7. refresh_cases: deletes named cache files, leaves the rest -----------


def test_refresh_cases_deletes_named_and_tolerates_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt_text = "prompt v1"
    model = "claude-sonnet-4-6"
    # A clean run populates the (patched, tmp) cache for both cases.
    _run(monkeypatch, tmp_path, _FULL_JSON, _EMPTY_JSON, model=model, prompt_text=prompt_text)

    prompt_hash = prompt_sha256(prompt_text)
    pos_path = cache_path(model, prompt_hash, "full_01_pos")
    trap_path = cache_path(model, prompt_hash, "full_01_trap")
    assert pos_path.exists() and trap_path.exists()

    # Refresh only the positive: its file is deleted, the trap's is untouched.
    refresh_cases(model, prompt_text, ["full_01_pos"])
    assert not pos_path.exists()
    assert trap_path.exists()

    # Refreshing an id with no cache file must not raise (unlink missing_ok).
    refresh_cases(model, prompt_text, ["never_cached"])
    # And the still-cached trap file is left in place by that no-op call.
    assert trap_path.exists()
