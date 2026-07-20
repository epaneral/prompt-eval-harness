"""Committed-cache integrity: the keyless-CI property, pinned as tests.

The gates job runs entirely from the committed response cache -- no API
key needed -- only while every pinned baseline's (prompt, model) pair has
a complete, input-fresh cache entry for every corpus case. Before these
tests, a violation (a corpus edit without a cache refresh, a deleted or
hand-mangled entry) was only discovered at gates runtime, by spending API
money. Here it is a pytest failure on every push.

Scope note: entries are checked against each baseline's own recorded
prompt_sha256, NOT against the current prompt file. A branch that edits
the prompt (the regression-demo exhibit) leaves the old prompt's cache
untouched and therefore still passes; pinning a new baseline is what
extends the property to the new prompt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.runner import BASELINES_DIR, cache_path
from harness.schema import CaseSpec

# Every pinned baseline participates; drift studies have their own shape
# and record no cache provenance, so they are excluded by prefix.
_BASELINES = sorted(
    p for p in BASELINES_DIR.glob("*.json") if not p.name.startswith("drift_")
)


def test_at_least_one_baseline_is_pinned() -> None:
    # Guards the parametrized test below from passing vacuously on an
    # empty glob (e.g. after a directory rename).
    assert _BASELINES, f"no pinned baselines found under {BASELINES_DIR}"


@pytest.mark.parametrize("baseline_path", _BASELINES, ids=lambda p: p.stem)
def test_committed_cache_complete_and_fresh(
    baseline_path: Path, manifest: list[CaseSpec], corpus_root: Path
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    model = baseline["model"]
    prompt_hash = baseline["prompt_sha256"]

    for spec in manifest:
        entry_path = cache_path(model, prompt_hash, spec.id)
        assert entry_path.exists(), (
            f"{baseline_path.name}: missing cache entry for {spec.id} "
            f"({entry_path}) -- keyless gates runs need the full committed cache"
        )
        entry = json.loads(entry_path.read_text(encoding="utf-8"))

        input_hash = hashlib.sha256(
            (corpus_root / spec.input).read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        assert entry["input_sha256"] == input_hash, (
            f"{baseline_path.name}: stale cache entry for {spec.id} -- the corpus "
            f"input was edited without refreshing the cache (and the baseline)"
        )
        # Provenance fields must agree with where the entry sits.
        assert entry["prompt_sha256"] == prompt_hash, spec.id
        assert entry["model"] == model, spec.id
        assert entry["case_id"] == spec.id, spec.id
