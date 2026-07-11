"""Runner suite: hashing, caching, cost math, strict parsing, and the run loop.

Every design decision encoded here is documented in harness/runner.py's
module docstring. The rules under test:

  * caching is structural -- a cache hit costs nothing and skips the API;
  * parsing applies ONE documented transport normalization (a whole-payload
    markdown fence is stripped and counted, not failed -- owner decision at
    Checkpoint 2) and is otherwise strict: prose, partial fences, and
    schema violations still fail;
  * a schema-invalid response grades as an EMPTY extraction (every
    expected indicator becomes a miss);
  * the cost cap is a hard abort on cumulative NEW spend;
  * a model with no pricing entry is rejected before any call.

No network is ever touched: no real anthropic client is constructed
(the client is always None) and harness.runner._call_api is monkeypatched
with a counting stub everywhere a call would otherwise happen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import harness.runner as runner
from harness.runner import (
    CachedResponse,
    CostCapExceeded,
    RunResult,
    cache_path,
    cost_usd,
    fetch_response,
    parse_output,
    prompt_sha256,
    run_corpus,
    strip_transport_fence,
    write_baseline,
)

# A canonical, defanged, schema-valid response body. Domains are bracketed
# per the pinned canonical defang style so it round-trips through the grader.
VALID_JSON = json.dumps(
    {"ipv4": [], "domains": ["evil[.]com"], "urls": [], "hashes": []}
)


# --- 1. prompt_sha256 -------------------------------------------------------


def test_prompt_sha256_stable_for_same_input() -> None:
    assert prompt_sha256("system prompt v1") == prompt_sha256("system prompt v1")


def test_prompt_sha256_differs_for_different_input() -> None:
    assert prompt_sha256("prompt a") != prompt_sha256("prompt b")


def test_prompt_sha256_is_64_hex_chars() -> None:
    digest = prompt_sha256("anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# --- 2. cache_path shape ----------------------------------------------------


def test_cache_path_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path)
    prompt_hash = "0123456789abcdef0123456789abcdef"
    path = cache_path("claude-sonnet-4-6", prompt_hash, "case_42")
    # CACHE_DIR / model / hash[:12] / case_id.json
    assert path == tmp_path / "claude-sonnet-4-6" / prompt_hash[:12] / "case_42.json"


# --- 3. cost_usd exact math -------------------------------------------------


def test_cost_usd_sonnet_1k_in_1k_out() -> None:
    # (1000 * 3.00 + 1000 * 15.00) / 1_000_000 == 0.018
    assert cost_usd("claude-sonnet-4-6", 1000, 1000) == pytest.approx(0.018)


def test_cost_usd_haiku_1k_in_1k_out() -> None:
    # (1000 * 1.00 + 1000 * 5.00) / 1_000_000 == 0.006
    assert cost_usd("claude-haiku-4-5", 1000, 1000) == pytest.approx(0.006)


def test_cost_usd_sonnet_asymmetric_tokens() -> None:
    # (500 * 3.00 + 200 * 15.00) / 1_000_000 == 0.0045
    assert cost_usd("claude-sonnet-4-6", 500, 200) == pytest.approx(0.0045)


def test_cost_usd_haiku_zero_tokens_is_zero() -> None:
    assert cost_usd("claude-haiku-4-5", 0, 0) == 0.0


def test_cost_usd_opus_1k_in_1k_out() -> None:
    # (1000 * 5.00 + 1000 * 25.00) / 1_000_000 == 0.030
    assert cost_usd("claude-opus-4-8", 1000, 1000) == pytest.approx(0.030)


# --- 3b. _call_api request shape per model -----------------------------------


class _RecordingClient:
    """Stands in for anthropic.Anthropic: captures messages.create kwargs."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self.messages = self

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs

        class _Block:
            type = "text"
            text = VALID_JSON

        class _Usage:
            input_tokens = 10
            output_tokens = 5

        class _Response:
            content = [_Block()]
            usage = _Usage()

        return _Response()


def test_call_api_sends_temperature_for_sonnet() -> None:
    client = _RecordingClient()
    runner._call_api(client, "claude-sonnet-4-6", "sys prompt", "input")
    assert client.kwargs is not None
    assert client.kwargs["temperature"] == 0.0
    assert client.kwargs["model"] == "claude-sonnet-4-6"
    assert client.kwargs["system"] == "sys prompt"


def test_call_api_omits_temperature_for_opus() -> None:
    # Opus 4.7+ rejects sampling params outright; the runner must not send them.
    client = _RecordingClient()
    runner._call_api(client, "claude-opus-4-8", "sys prompt", "input")
    assert client.kwargs is not None
    assert "temperature" not in client.kwargs
    assert client.kwargs["model"] == "claude-opus-4-8"


# --- 4. parse_output strictness --------------------------------------------


def test_parse_output_accepts_canonical_object() -> None:
    result = parse_output(VALID_JSON)
    assert result is not None
    assert result.domains == ["evil[.]com"]
    assert result.ipv4 == [] and result.urls == [] and result.hashes == []


def test_parse_output_tolerates_surrounding_whitespace() -> None:
    assert parse_output("  \n\t" + VALID_JSON + "\n  ") is not None


def test_parse_output_alone_rejects_json_fence() -> None:
    # parse_output itself stays strict; fence handling is a separate,
    # documented normalization layer applied before it.
    fenced = "```json\n" + VALID_JSON + "\n```"
    assert parse_output(fenced) is None


# --- 4b. strip_transport_fence: the one documented normalization -------------


def test_strip_fence_whole_payload_json_tag() -> None:
    body, fenced = strip_transport_fence("```json\n" + VALID_JSON + "\n```")
    assert fenced is True
    assert body == VALID_JSON
    assert parse_output(body) is not None


def test_strip_fence_whole_payload_bare_fence() -> None:
    body, fenced = strip_transport_fence("```\n" + VALID_JSON + "\n```")
    assert fenced is True
    assert body == VALID_JSON


def test_strip_fence_tolerates_surrounding_whitespace() -> None:
    body, fenced = strip_transport_fence("\n  ```json\n" + VALID_JSON + "\n```  \n")
    assert fenced is True
    assert body == VALID_JSON


def test_strip_fence_leaves_unfenced_text_alone() -> None:
    body, fenced = strip_transport_fence(VALID_JSON)
    assert fenced is False
    assert body == VALID_JSON


def test_strip_fence_does_not_normalize_prose_around_fence() -> None:
    # The line is precise: only a fence wrapping the ENTIRE payload counts.
    text = "Here you go:\n```json\n" + VALID_JSON + "\n```"
    body, fenced = strip_transport_fence(text)
    assert fenced is False
    assert body == text


def test_strip_fence_does_not_normalize_partial_fence() -> None:
    text = "```json\n" + VALID_JSON  # opening fence only
    body, fenced = strip_transport_fence(text)
    assert fenced is False
    assert body == text


def test_parse_output_rejects_prose_wrapped_json() -> None:
    assert parse_output("Here is the extraction: " + VALID_JSON) is None


def test_parse_output_rejects_missing_key() -> None:
    missing = json.dumps({"ipv4": [], "domains": [], "urls": []})
    assert parse_output(missing) is None


def test_parse_output_rejects_extra_key() -> None:
    extra = json.dumps(
        {"ipv4": [], "domains": [], "urls": [], "hashes": [], "note": "x"}
    )
    assert parse_output(extra) is None


def test_parse_output_rejects_empty_string() -> None:
    assert parse_output("") is None


# --- 5. fetch_response cache round-trip -------------------------------------


class _CallCounter:
    """Stub for _call_api: counts invocations, returns a fixed response."""

    def __init__(self, response: CachedResponse) -> None:
        self.response = response
        self.calls = 0

    def __call__(
        self, client: object, model: str, system_prompt: str, input_text: str
    ) -> CachedResponse:
        self.calls += 1
        return self.response


def test_fetch_response_cache_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path)
    stub = _CallCounter(CachedResponse(VALID_JSON, input_tokens=120, output_tokens=34))
    monkeypatch.setattr(runner, "_call_api", stub)

    # First call: a miss -> the stub runs and the cache file is written.
    first, from_cache = fetch_response(None, "claude-sonnet-4-6", "prompt", "case_a", "input text")
    assert from_cache is False
    assert stub.calls == 1
    assert first == CachedResponse(VALID_JSON, 120, 34)

    path = cache_path("claude-sonnet-4-6", prompt_sha256("prompt"), "case_a")
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written) == {"model", "prompt_sha256", "case_id", "response_text", "usage"}
    assert written["model"] == "claude-sonnet-4-6"
    assert written["prompt_sha256"] == prompt_sha256("prompt")
    assert written["case_id"] == "case_a"
    assert written["response_text"] == VALID_JSON
    assert written["usage"] == {"input_tokens": 120, "output_tokens": 34}

    # Second call: a hit -> served from disk, stub NOT invoked again.
    second, from_cache = fetch_response(None, "claude-sonnet-4-6", "prompt", "case_a", "input text")
    assert from_cache is True
    assert stub.calls == 1  # unchanged
    assert second == first


# --- 6. run_corpus end-to-end ----------------------------------------------

# One twin pair (one positive, one trap), both expecting a single defanged
# domain, so a stub returning that exact object is an exact match for both.
_MANIFEST = """\
cases:
- id: defanged_01_pos
  category: defanged
  role: positive
  twin_id: defanged_01_trap
  input: inputs/defanged_01_pos.txt
  expected: expected/defanged_01_pos.json
- id: defanged_01_trap
  category: defanged
  role: trap
  twin_id: defanged_01_pos
  input: inputs/defanged_01_trap.txt
  expected: expected/defanged_01_trap.json
"""

_EXPECTED = json.dumps(
    {"ipv4": [], "domains": ["evil[.]com"], "urls": [], "hashes": []}, indent=2
)


def _build_corpus(root: Path) -> Path:
    """Write a minimal two-case corpus under root; return the corpus dir."""
    corpus = root / "corpus"
    (corpus / "inputs").mkdir(parents=True)
    (corpus / "expected").mkdir(parents=True)
    (corpus / "manifest.yaml").write_text(_MANIFEST, encoding="utf-8")
    for case in ("defanged_01_pos", "defanged_01_trap"):
        (corpus / "inputs" / f"{case}.txt").write_text(
            "Indicator seen: evil[.]com\n", encoding="utf-8"
        )
        (corpus / "expected" / f"{case}.json").write_text(_EXPECTED, encoding="utf-8")
    return corpus


class _ScriptedCall:
    """Stub for _call_api returning a fixed CachedResponse, counting calls."""

    def __init__(self, response: CachedResponse) -> None:
        self.response = response
        self.calls = 0

    def __call__(
        self, client: object, model: str, system_prompt: str, input_text: str
    ) -> CachedResponse:
        self.calls += 1
        return self.response


def test_run_corpus_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    # 100 in / 40 out per call, both cases -> both match expected exactly.
    stub = _ScriptedCall(CachedResponse(VALID_JSON, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    result = run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)

    assert isinstance(result, RunResult)
    assert result.api_calls == 2
    assert result.cache_hits == 0
    assert stub.calls == 2
    assert result.schema_validity_rate == 1.0
    assert result.schema_invalid_cases == []
    assert result.fence_normalized_cases == []
    assert result.report.exact_match_rate == 1.0
    # Hand-computed: two calls at cost_usd(sonnet, 100, 40) each.
    per_call = cost_usd("claude-sonnet-4-6", 100, 40)
    assert result.total_cost_usd == pytest.approx(2 * per_call)


def test_run_corpus_rerun_is_all_cache_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    stub = _ScriptedCall(CachedResponse(VALID_JSON, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)  # populate cache
    assert stub.calls == 2

    second = run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)
    assert second.api_calls == 0
    assert second.cache_hits == 2
    assert stub.calls == 2  # no further calls
    assert second.total_cost_usd == 0.0
    # Cache hits still grade correctly.
    assert second.report.exact_match_rate == 1.0
    assert second.schema_validity_rate == 1.0


def test_run_corpus_schema_invalid_grades_as_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    # Both input files are identical here, so we cannot key the stub on input.
    # Use a stateful stub that returns prose first, then valid JSON second.
    scripted: list[CachedResponse] = [
        CachedResponse("Sure! I found evil[.]com in the text.", 100, 40),  # prose
        CachedResponse(VALID_JSON, 100, 40),  # valid
    ]
    call_index = {"n": 0}

    def stub(client: object, model: str, system_prompt: str, input_text: str) -> CachedResponse:
        resp = scripted[call_index["n"]]
        call_index["n"] += 1
        return resp

    monkeypatch.setattr(runner, "_call_api", stub)

    result = run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)

    # Manifest order is pos then trap; the prose response lands on the pos case.
    assert result.schema_invalid_cases == ["defanged_01_pos"]
    assert result.schema_validity_rate == 0.5

    # The schema-invalid case grades as an EMPTY extraction: its expected
    # single domain becomes a false negative, and it is not an exact match.
    by_id = {g.case_id: g for g in result.report.cases}
    invalid_grade = by_id["defanged_01_pos"]
    assert invalid_grade.exact_match is False
    assert invalid_grade.counts["domains"].fn == 1
    assert invalid_grade.counts["domains"].tp == 0
    assert invalid_grade.counts["domains"].fp == 0
    assert invalid_grade.missed["domains"] == ["evil[.]com"]

    # The valid case still matches exactly.
    assert by_id["defanged_01_trap"].exact_match is True


def test_run_corpus_fenced_response_normalized_and_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A whole-payload fence is stripped, graded normally, and counted."""
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    fenced = "```json\n" + VALID_JSON + "\n```"
    stub = _ScriptedCall(CachedResponse(fenced, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    result = run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)

    assert result.schema_validity_rate == 1.0
    assert result.schema_invalid_cases == []
    assert result.fence_normalized_cases == ["defanged_01_pos", "defanged_01_trap"]
    assert result.report.exact_match_rate == 1.0


def test_run_corpus_cost_cap_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    per_call = cost_usd("claude-sonnet-4-6", 100, 40)
    # Cap just below a single call's cost -> first NEW call trips the abort.
    monkeypatch.setattr(runner, "COST_CAP_USD", per_call / 2)
    stub = _ScriptedCall(CachedResponse(VALID_JSON, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    with pytest.raises(CostCapExceeded):
        run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)


def test_run_corpus_unknown_model_rejected_before_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    corpus = _build_corpus(tmp_path)
    stub = _ScriptedCall(CachedResponse(VALID_JSON, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    with pytest.raises(ValueError):
        run_corpus(None, "claude-opus-9-9", "prompt v1", corpus)
    # ValueError is raised before the manifest is loaded or any call is made.
    assert stub.calls == 0


# --- 7. write_baseline ------------------------------------------------------


def test_write_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "BASELINES_DIR", tmp_path / "baselines")
    corpus = _build_corpus(tmp_path)
    stub = _ScriptedCall(CachedResponse(VALID_JSON, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    result = run_corpus(None, "claude-sonnet-4-6", "prompt v1", corpus)

    prompt_path = tmp_path / "v1.md"
    prompt_path.write_text("prompt v1", encoding="utf-8")
    out = write_baseline(prompt_path, "prompt v1", "claude-sonnet-4-6", result)

    # Path is named {stem}_{model}.json under the patched baselines dir.
    assert out == (tmp_path / "baselines" / "v1_claude-sonnet-4-6.json")
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["prompt_version"] == "v1"
    assert payload["prompt_sha256"] == prompt_sha256("prompt v1")
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["schema_validity_rate"] == 1.0
    assert payload["fence_normalized_cases"] == []

    report = payload["report"]
    assert "per_type" in report
    assert "exact_match_rate" in report
    assert "trap_fp_case_rate" in report


def test_write_baseline_temperature_null_for_opus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An Opus baseline must not claim a temperature that was never sent."""
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "BASELINES_DIR", tmp_path / "baselines")
    corpus = _build_corpus(tmp_path)
    stub = _ScriptedCall(CachedResponse(VALID_JSON, input_tokens=100, output_tokens=40))
    monkeypatch.setattr(runner, "_call_api", stub)

    result = run_corpus(None, "claude-opus-4-8", "prompt v1", corpus)
    prompt_path = tmp_path / "v1.md"
    prompt_path.write_text("prompt v1", encoding="utf-8")
    out = write_baseline(prompt_path, "prompt v1", "claude-opus-4-8", result)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["temperature"] is None
    assert payload["model"] == "claude-opus-4-8"
