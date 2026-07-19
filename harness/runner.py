"""API runner: execute the corpus against a prompt version, cache responses, pin baselines.

Design notes (Checkpoint 2 review artifacts):

Caching is structural, not an optimization. Responses are committed to
``cache/`` keyed on (prompt_sha256, model, case_id) so that a fresh clone
grades from fixtures with no API key, and CI only spends money when the
prompt or corpus actually changed. A cache hit costs nothing and cannot
trip the cost cap.

Fence handling is PER-MODEL and baseline-derived (owner decision,
Checkpoint 2). A markdown fence wrapping the entire payload is transport
framing; whether it counts as a validity failure depends on the model's
own pinned baseline behavior: a model whose baseline established fencing
as its native framing (Haiku: 44/44 fenced) has the fence normalized
away before validation, while a model whose baseline is fence-free
(Sonnet: 0/44) is parsed strictly -- a fenced response there is a
deviation from its own baseline and fails schema validity. The
OBSERVATION is uniform either way: fenced responses are counted and
listed for every model, so framing behavior stays comparable across
models regardless of policy. Nothing else is ever repaired: prose
around the JSON, partial fences, and commentary fail for every model.

A schema-invalid response grades as an EMPTY extraction: an output the
harness cannot parse extracts nothing, so every expected indicator counts
as a miss. Schema validity is also reported separately (it is its own
gate). This choice is flagged for owner ratification at Checkpoint 2.

Denominators are labeled, not blended (owner direction). Exact-match is
reported twice -- over all cases (invalid counts as a miss; the fixed
denominator the no-regression gate needs) and over schema-valid outputs
only (the pure extraction-quality view). The trap metric is CONDITIONED
on validity: an unparseable trap response is not a trap "success", so
the rate is leaking traps / schema-valid traps, n/a (fail-closed) when
no trap produced valid output. Recall and per-type metrics deliberately
stay unconditional: computed over valid outputs only, they could be
improved by breaking formatting on hard cases -- the excluded-denominator
failure mode this design rejects.

The cost cap is a hard abort on cumulative NEW spend within one run
(cache hits are free). It is a runaway brake, not a budget plan.

The prompt file is the system prompt, sent verbatim; the case input text
is the user message. Version identity = filename + git history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from pydantic import ValidationError

from harness.grader import Report, grade_corpus
from harness.schema import IOCExtraction, load_manifest

# USD per million tokens (input, output). A model must have a pricing entry
# before the runner will call it: the cost cap cannot be enforced otherwise.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
}
# Opus 4.7+ removed sampling parameters: a request that sends temperature is
# rejected outright. The runner omits it for these models and the baseline
# records temperature as null -- the nondeterminism policy degrades from
# "temperature 0" to "no sampling controls exposed; assume drift anyway".
NO_SAMPLING_PARAMS = frozenset({"claude-opus-4-8"})
# Baseline-derived fence policy: only models whose pinned baseline showed
# fencing as their native framing get the fence normalized before validation.
# Adding a model here requires probe-run evidence, like any other threshold.
FENCE_NORMALIZING_MODELS = frozenset({"claude-haiku-4-5"})
PRIMARY_MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
COST_CAP_USD = 0.50

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
BASELINES_DIR = ROOT / "baselines"


class CostCapExceeded(RuntimeError):
    """Raised mid-run when cumulative new API spend crosses COST_CAP_USD."""


@dataclass(frozen=True)
class CachedResponse:
    response_text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RunResult:
    report: Report
    schema_invalid_cases: list[str]
    schema_validity_rate: float
    # Observation, not policy: every fenced response is recorded here for
    # every model, whether or not the fence was normalized away.
    fenced_cases: list[str]
    # Conditioned views: denominators restricted to schema-valid outputs,
    # None (never coerced) when the denominator is empty.
    n_valid_cases: int
    n_valid_traps: int
    exact_match_rate_valid: float | None
    trap_fp_case_rate_valid: float | None
    total_cost_usd: float
    api_calls: int
    cache_hits: int


def prompt_sha256(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def cache_path(model: str, prompt_hash: str, case_id: str) -> Path:
    # First 12 hex chars keep paths short; the full hash is stored inside
    # the cache file and the baseline for exact provenance.
    return CACHE_DIR / model / prompt_hash[:12] / f"{case_id}.json"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = MODEL_PRICING[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def _call_api(
    client: anthropic.Anthropic, model: str, system_prompt: str, input_text: str
) -> CachedResponse:
    sampling: dict[str, float] = (
        {} if model in NO_SAMPLING_PARAMS else {"temperature": TEMPERATURE}
    )
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": input_text}],
        **sampling,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return CachedResponse(
        response_text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def fetch_response(
    client: anthropic.Anthropic,
    model: str,
    prompt_text: str,
    case_id: str,
    input_text: str,
) -> tuple[CachedResponse, bool]:
    """Return (response, from_cache), writing the cache on a miss."""
    prompt_hash = prompt_sha256(prompt_text)
    path = cache_path(model, prompt_hash, case_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        usage = data["usage"]
        return (
            CachedResponse(data["response_text"], usage["input_tokens"], usage["output_tokens"]),
            True,
        )
    response = _call_api(client, model, prompt_text, input_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "prompt_sha256": prompt_hash,
        "case_id": case_id,
        "response_text": response.response_text,
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return response, False


# Exactly one fence wrapping the entire payload, optional info string
# (```json). Anything less total -- prose before/after, partial fences --
# deliberately does not match and remains schema-invalid.
_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*[ \t]*\n(.*?)\n?```$", re.DOTALL)


def strip_transport_fence(text: str) -> tuple[str, bool]:
    """Remove one whole-payload markdown fence; report whether one was found."""
    match = _FENCE.match(text.strip())
    if match:
        return match.group(1), True
    return text, False


def parse_output(text: str) -> IOCExtraction | None:
    """Strictly parse a (transport-normalized) response body, else None."""
    try:
        return IOCExtraction.model_validate_json(text.strip())
    except ValidationError:
        return None


def run_corpus(
    client: anthropic.Anthropic,
    model: str,
    prompt_text: str,
    corpus_dir: Path = ROOT / "corpus",
) -> RunResult:
    if model not in MODEL_PRICING:
        raise ValueError(f"no pricing entry for {model!r}; cost cap cannot be enforced")
    specs = load_manifest(corpus_dir / "manifest.yaml")
    triples = []
    schema_invalid: list[str] = []
    fenced_cases: list[str] = []
    spent = 0.0
    api_calls = 0
    cache_hits = 0
    for spec in specs:
        input_text = (corpus_dir / spec.input).read_text(encoding="utf-8")
        expected = IOCExtraction.model_validate_json(
            (corpus_dir / spec.expected).read_text(encoding="utf-8")
        )
        response, from_cache = fetch_response(client, model, prompt_text, spec.id, input_text)
        if from_cache:
            cache_hits += 1
        else:
            api_calls += 1
            spent += cost_usd(model, response.input_tokens, response.output_tokens)
            if spent > COST_CAP_USD:
                raise CostCapExceeded(
                    f"spent ${spent:.4f} after {api_calls} calls; cap is ${COST_CAP_USD:.2f}"
                )
        body, fenced = strip_transport_fence(response.response_text)
        if fenced:
            fenced_cases.append(spec.id)
        if model not in FENCE_NORMALIZING_MODELS:
            # Strict model: fencing deviates from its baseline, so the raw
            # text is validated as-is and a fenced response fails.
            body = response.response_text
        actual = parse_output(body)
        if actual is None:
            schema_invalid.append(spec.id)
            actual = IOCExtraction(ipv4=[], domains=[], urls=[], hashes=[])
        triples.append((spec, expected, actual))
    report = grade_corpus(triples)
    invalid_ids = set(schema_invalid)
    valid_grades = [g for g in report.cases if g.case_id not in invalid_ids]
    valid_traps = [g for g in valid_grades if g.role == "trap"]
    # trap_fp_cases can only contain valid cases (an empty extraction cannot
    # falsely extract), so the conditioned numerator is the same list.
    return RunResult(
        report=report,
        schema_invalid_cases=sorted(schema_invalid),
        schema_validity_rate=1 - len(schema_invalid) / len(specs),
        fenced_cases=sorted(fenced_cases),
        n_valid_cases=len(valid_grades),
        n_valid_traps=len(valid_traps),
        exact_match_rate_valid=(
            sum(g.exact_match for g in valid_grades) / len(valid_grades)
            if valid_grades
            else None
        ),
        trap_fp_case_rate_valid=(
            len(report.trap_fp_cases) / len(valid_traps) if valid_traps else None
        ),
        total_cost_usd=spent,
        api_calls=api_calls,
        cache_hits=cache_hits,
    )


def write_baseline(prompt_path: Path, prompt_text: str, model: str, result: RunResult) -> Path:
    """Pin a run's full results to baselines/<prompt_version>_<model>.json."""
    payload = {
        "prompt_version": prompt_path.stem,
        "prompt_sha256": prompt_sha256(prompt_text),
        "model": model,
        "temperature": None if model in NO_SAMPLING_PARAMS else TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_validity_rate": result.schema_validity_rate,
        "schema_invalid_cases": result.schema_invalid_cases,
        "fenced_cases": result.fenced_cases,
        "fence_policy": "normalize" if model in FENCE_NORMALIZING_MODELS else "strict",
        "n_valid_cases": result.n_valid_cases,
        "n_valid_traps": result.n_valid_traps,
        "exact_match_rate_valid": result.exact_match_rate_valid,
        "trap_fp_case_rate_valid": result.trap_fp_case_rate_valid,
        "total_cost_usd": round(result.total_cost_usd, 6),
        "api_calls": result.api_calls,
        "cache_hits": result.cache_hits,
        "report": asdict(result.report),
    }
    BASELINES_DIR.mkdir(exist_ok=True)
    path = BASELINES_DIR / f"{prompt_path.stem}_{model}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(result: RunResult, model: str, prompt_path: Path) -> None:
    report = result.report
    print(f"prompt: {prompt_path.name}   model: {model}")
    print(
        f"cases: {len(report.cases)}   api calls: {result.api_calls}   "
        f"cache hits: {result.cache_hits}   new spend: ${result.total_cost_usd:.4f}"
    )
    invalid = ", ".join(result.schema_invalid_cases) or "none"
    print(f"schema validity: {result.schema_validity_rate:.1%}   invalid: {invalid}")
    policy = "normalize" if model in FENCE_NORMALIZING_MODELS else "strict"
    print(
        f"fenced responses observed: {len(result.fenced_cases)}/{len(report.cases)}"
        f"   (fence policy for this model: {policy})"
    )
    print(f"exact-match (all cases; invalid counts as miss): {report.exact_match_rate:.1%}")
    valid_em = (
        "n/a"
        if result.exact_match_rate_valid is None
        else f"{result.exact_match_rate_valid:.1%}"
    )
    print(
        f"exact-match (schema-valid outputs, n={result.n_valid_cases}/{len(report.cases)}): "
        f"{valid_em}"
    )
    print("per-type (micro-averaged, all cases -- see docstring):")
    for t, m in report.per_type.items():
        print(f"  {t:<8} P={_fmt(m.precision)}  R={_fmt(m.recall)}  F1={_fmt(m.f1)}")
    total_traps = sum(1 for g in report.cases if g.role == "trap")
    rate = (
        "n/a"
        if result.trap_fp_case_rate_valid is None
        else f"{result.trap_fp_case_rate_valid:.1%}"
    )
    offenders = ", ".join(report.trap_fp_cases) or "none"
    print(
        f"trap false-extraction rate (schema-valid traps, n={result.n_valid_traps}/{total_traps}): "
        f"{rate}   offending: {offenders}"
    )
    print("failure concentration by category (fp/fn summed over types):")
    for category, counts in report.by_category.items():
        fp = sum(c.fp for c in counts.values())
        fn = sum(c.fn for c in counts.values())
        print(f"  {category:<24} fp={fp}  fn={fn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IOC-extraction eval corpus.")
    parser.add_argument("--model", default=PRIMARY_MODEL, choices=sorted(MODEL_PRICING))
    parser.add_argument("--prompt", default=str(ROOT / "prompts" / "v1.md"))
    args = parser.parse_args()

    prompt_path = Path(args.prompt)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    client = anthropic.Anthropic()

    result = run_corpus(client, args.model, prompt_text)
    print_summary(result, args.model, prompt_path)
    baseline = write_baseline(prompt_path, prompt_text, args.model, result)
    print(f"baseline pinned: {baseline.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
