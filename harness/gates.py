"""Regression gates: mechanical threshold checks against the pinned baseline.

Every threshold below was ratified at Checkpoint 2 from baseline evidence
(v1 on claude-sonnet-4-6: all metrics at ceiling) plus a drift study
(baselines/drift_v1_claude-sonnet-4-6.json: 5 independent uncached runs,
220 case-samples, zero flips). Nothing here is an invented number.

Gates, all AND-ed:
  1. schema validity == 100%, judged under the model's recorded fence policy
  2. recall == 1.0 for EVERY indicator type
  3. trap false-extraction rate over schema-valid traps == 0%
  4. no metric below the pinned baseline (redundant while the baseline is
     at ceiling; becomes independent the day a below-ceiling baseline is
     deliberately pinned)

Undefined metrics (None / "n/a") FAIL their gate: "no data" is not a pass.
This is what stops a run with zero valid outputs from sailing through the
trap gate on a flattering 0%.

A red counts as a regression only after it reproduces on fresh samples
(ratified rule; drift study says this should be rare). --refresh-failing
automates it: on a first failure, the failing cases' cache entries are
deleted and the run repeats with new API samples. A failure that repeats
is a confirmed regression (exit 1); one that vanishes is recorded as a
drift flake (exit 0, loudly noted). CI runs with this flag so the rule is
applied mechanically, not by human memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic

from harness.grader import INDICATOR_TYPES
from harness.runner import (
    BASELINES_DIR,
    MODEL_PRICING,
    PRIMARY_MODEL,
    ROOT,
    RunResult,
    cache_path,
    prompt_sha256,
    run_corpus,
)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def _check_floor(current: float | None, floor: float, none_detail: str) -> tuple[bool, str]:
    if current is None:
        return False, none_detail
    return current >= floor, f"{current:.3f}"


def evaluate_gates(result: RunResult, baseline: dict) -> list[GateResult]:
    gates: list[GateResult] = []

    # Gate 1: schema validity == 100% (fence policy already applied upstream).
    invalid = ", ".join(result.schema_invalid_cases) or "none"
    gates.append(
        GateResult(
            "schema_validity_100",
            result.schema_validity_rate == 1.0,
            f"{result.schema_validity_rate:.1%} valid; invalid: {invalid}",
        )
    )

    # Gate 2: per-type recall floor 1.0, fail-closed on undefined.
    for t in INDICATOR_TYPES:
        recall = result.report.per_type[t].recall
        passed, detail = _check_floor(recall, 1.0, "recall undefined (fail-closed)")
        gates.append(GateResult(f"recall_floor_{t}", passed, f"recall {detail}"))

    # Gate 3: trap ceiling 0% over schema-valid traps, fail-closed on undefined.
    trap_rate = result.trap_fp_case_rate_valid
    if trap_rate is None:
        gates.append(
            GateResult(
                "trap_ceiling",
                False,
                "no schema-valid trap outputs (fail-closed)",
            )
        )
    else:
        offenders = ", ".join(result.report.trap_fp_cases) or "none"
        gates.append(
            GateResult(
                "trap_ceiling",
                trap_rate == 0.0,
                f"{trap_rate:.1%} of {result.n_valid_traps} valid traps; offending: {offenders}",
            )
        )

    # Gate 4: no metric below the pinned baseline.
    violations: list[str] = []
    base_report = baseline["report"]
    if result.report.exact_match_rate < base_report["exact_match_rate"]:
        violations.append(
            f"exact_match {result.report.exact_match_rate:.3f} < "
            f"baseline {base_report['exact_match_rate']:.3f}"
        )
    if result.schema_validity_rate < baseline["schema_validity_rate"]:
        violations.append(
            f"schema_validity {result.schema_validity_rate:.3f} < "
            f"baseline {baseline['schema_validity_rate']:.3f}"
        )
    for t in INDICATOR_TYPES:
        for metric in ("precision", "recall", "f1"):
            base_value = base_report["per_type"][t][metric]
            if base_value is None:
                continue  # no baseline basis for comparison
            current = getattr(result.report.per_type[t], metric)
            if current is None or current < base_value:
                shown = "n/a" if current is None else f"{current:.3f}"
                violations.append(f"{t}.{metric} {shown} < baseline {base_value:.3f}")
    base_trap = baseline.get("trap_fp_case_rate_valid")
    if base_trap is not None and (trap_rate is None or trap_rate > base_trap):
        shown = "n/a" if trap_rate is None else f"{trap_rate:.3f}"
        violations.append(f"trap_rate {shown} > baseline {base_trap:.3f}")
    gates.append(
        GateResult(
            "no_regression_vs_baseline",
            not violations,
            "; ".join(violations) or "no metric below baseline",
        )
    )
    return gates


def failing_case_ids(result: RunResult) -> list[str]:
    """Cases to resample when confirming a red: anything not exactly right."""
    ids = set(result.schema_invalid_cases)
    ids.update(g.case_id for g in result.report.cases if not g.exact_match)
    return sorted(ids)


def refresh_cases(model: str, prompt_text: str, case_ids: list[str]) -> None:
    """Delete the cache entries for these cases so the next run resamples them."""
    prompt_hash = prompt_sha256(prompt_text)
    for case_id in case_ids:
        cache_path(model, prompt_hash, case_id).unlink(missing_ok=True)


def _print_verdict(gates: list[GateResult], label: str) -> bool:
    print(f"--- gates: {label} ---")
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"  {status}  {gate.name}: {gate.detail}")
    all_passed = all(g.passed for g in gates)
    print(f"  => {'ALL GATES PASS' if all_passed else 'GATE FAILURE'}")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate regression gates.")
    parser.add_argument("--model", default=PRIMARY_MODEL, choices=sorted(MODEL_PRICING))
    parser.add_argument("--prompt", default=str(ROOT / "prompts" / "v1.md"))
    parser.add_argument(
        "--refresh-failing",
        action="store_true",
        help="on a red, resample the failing cases and require the failure to reproduce",
    )
    args = parser.parse_args()

    prompt_path = Path(args.prompt)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    baseline_path = BASELINES_DIR / f"{prompt_path.stem}_{args.model}.json"
    if not baseline_path.exists():
        print(f"no pinned baseline at {baseline_path}; run the runner and pin one first")
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    client = anthropic.Anthropic()
    result = run_corpus(client, args.model, prompt_text)
    if _print_verdict(evaluate_gates(result, baseline), "first evaluation"):
        return 0

    if not args.refresh_failing:
        print("red without --refresh-failing: rerun with the flag to apply the")
        print("ratified rule (a regression must reproduce on fresh samples)")
        return 1

    to_refresh = failing_case_ids(result)
    print(f"resampling {len(to_refresh)} failing case(s): {', '.join(to_refresh)}")
    refresh_cases(args.model, prompt_text, to_refresh)
    fresh = run_corpus(client, args.model, prompt_text)
    if _print_verdict(evaluate_gates(fresh, baseline), "fresh-sample confirmation"):
        print("NOTE: first red did not reproduce on fresh samples -- recorded as a")
        print("drift flake per the ratified rule, not a regression")
        return 0
    print("regression CONFIRMED: failure reproduced on fresh samples")
    return 1


if __name__ == "__main__":
    sys.exit(main())
