"""Normalization and scoring for IOC-extraction outputs.

Canonicalization policy (Checkpoint 1 review artifact)
======================================================

Principle: the grader normalizes REPRESENTATION only; all extraction
JUDGMENT (what counts as an indicator) lives in the prompt and the
expected files. If the grader filtered private IPs or example.com, it
would grade away exactly the mistakes the near-miss twin cases exist
to catch. Canonicalization changes form, never content.

Canonical form is DEFANGED AT REST (owner decision): expected files,
and both sides of every comparison, use one fixed defanged style so
the repository never contains live-looking indicators.

Normalization pipeline, applied identically to model output and
expected files:

    strip whitespace
    -> refang (closed variant table below)
    -> lowercase (domains, hashes, URL scheme+host)
    -> strip one trailing dot (domains)
    -> apply canonical defang
    -> dedupe (set semantics per indicator type)

Refang variant table (closed -- additions require owner sign-off):

    hxxp / hxxps (any case)   -> http / https
    [.]                       -> .
    (.)                       -> .
    [://]                     -> ://

Canonical defang style (pinned):

    domains   every dot bracketed          evil[.]com
    ipv4      every dot bracketed          45[.]129[.]14[.]203
    urls      http->hxxp, https->hxxps; host dots bracketed;
              path/query/fragment verbatim (dots NOT bracketed)

URL normalization is deliberately minimal: lowercase scheme+host only;
path/query/fragment stay case-sensitive verbatim (RFC 3986); no
percent-decoding, no default-port removal. One equivalence exception:
a bare-host trailing slash is stripped (hxxp://x[.]com/ == hxxp://x[.]com)
only when there is no other path, query, or fragment.

Duplicates in model output are never penalized (dedupe precedes
comparison). Out of scope for v1: IPv6, IDN/punycode, domain:port
outside URLs.

Judgment policies (enforced in prompts/v1.md and the expected files,
NEVER by this module): non-global IPv4s excluded; RFC 2606/6761 names
excluded; incidental benign-context domains excluded; a URL's host is
also extracted into domains/ipv4; version strings, git SHAs, and UUIDs
are not indicators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from harness.schema import CaseSpec, IOCExtraction

INDICATOR_TYPES: tuple[str, ...] = ("ipv4", "domains", "urls", "hashes")

# hxxps must be replaced before hxxp: a single hxxp->http pass would leave
# the trailing "S" of an uppercase "HXXPS" untouched ("httpS").
_HXXPS = re.compile(r"hxxps", re.IGNORECASE)
_HXXP = re.compile(r"hxxp", re.IGNORECASE)

_DEFANG_VARIANTS: tuple[tuple[str, str], ...] = (
    ("[.]", "."),
    ("(.)", "."),
    ("[://]", "://"),
)

_SCHEME_DEFANG = {"http": "hxxp", "https": "hxxps"}


def refang(s: str) -> str:
    """Collapse every defang style in the closed table to live form.

    Applied to the whole string: normalizers re-defang afterwards, so
    refang only needs to erase stylistic variation, not preserve it.
    """
    s = _HXXPS.sub("https", s)
    s = _HXXP.sub("http", s)
    for defanged, live in _DEFANG_VARIANTS:
        s = s.replace(defanged, live)
    return s


def _defang_dots(s: str) -> str:
    return s.replace(".", "[.]")


def normalize_ipv4(s: str) -> str:
    return _defang_dots(refang(s.strip()))


def normalize_domain(s: str) -> str:
    live = refang(s.strip()).lower()
    return _defang_dots(live.removesuffix("."))


def normalize_hash(s: str) -> str:
    return s.strip().lower()


def normalize_url(s: str) -> str:
    """Canonicalize a URL: refang, lowercase scheme+host, defang scheme+host.

    Splitting happens on the refanged (live) form so the parser sees a
    syntactically ordinary URL; the canonical defang is applied to the
    scheme and netloc only, leaving path/query/fragment verbatim.
    """
    parts = urlsplit(refang(s.strip()))
    netloc = parts.netloc.lower()
    path = parts.path
    if path == "/" and not parts.query and not parts.fragment:
        path = ""
    scheme = _SCHEME_DEFANG.get(parts.scheme.lower(), parts.scheme.lower())
    return urlunsplit((scheme, _defang_dots(netloc), path, parts.query, parts.fragment))


def normalize_extraction(x: IOCExtraction) -> dict[str, set[str]]:
    """Map an extraction to canonical, deduped per-type sets."""
    return {
        "ipv4": {normalize_ipv4(v) for v in x.ipv4},
        "domains": {normalize_domain(v) for v in x.domains},
        "urls": {normalize_url(v) for v in x.urls},
        "hashes": {normalize_hash(v) for v in x.hashes},
    }


@dataclass(frozen=True)
class TypeCounts:
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class CaseGrade:
    case_id: str
    category: str
    role: str
    counts: dict[str, TypeCounts]
    # The offending values themselves (sorted, for determinism): the
    # failure breakdown must show WHERE failures concentrate, not just counts.
    false_extractions: dict[str, list[str]]
    missed: dict[str, list[str]]
    exact_match: bool


@dataclass(frozen=True)
class Metrics:
    # None means undefined (zero denominator), deliberately never coerced
    # to 0.0 or 1.0: a downstream gate must handle "no data" explicitly
    # rather than silently passing or failing.
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class Report:
    per_type: dict[str, Metrics]
    exact_match_rate: float
    by_category: dict[str, dict[str, TypeCounts]]
    # Fraction of trap cases with >= 1 false positive of any type; the
    # future false-extraction gate reads this. Case-level rather than
    # token-level so one pathological case cannot dominate the rate.
    trap_fp_case_rate: float | None
    trap_fp_cases: list[str]
    cases: list[CaseGrade]


def grade_case(spec: CaseSpec, expected: IOCExtraction, actual: IOCExtraction) -> CaseGrade:
    exp = normalize_extraction(expected)
    act = normalize_extraction(actual)
    counts: dict[str, TypeCounts] = {}
    false_extractions: dict[str, list[str]] = {}
    missed: dict[str, list[str]] = {}
    for t in INDICATOR_TYPES:
        fp = act[t] - exp[t]
        fn = exp[t] - act[t]
        counts[t] = TypeCounts(tp=len(act[t] & exp[t]), fp=len(fp), fn=len(fn))
        false_extractions[t] = sorted(fp)
        missed[t] = sorted(fn)
    exact = all(c.fp == 0 and c.fn == 0 for c in counts.values())
    return CaseGrade(
        case_id=spec.id,
        category=spec.category,
        role=spec.role,
        counts=counts,
        false_extractions=false_extractions,
        missed=missed,
        exact_match=exact,
    )


def _metrics(tp: int, fp: int, fn: int) -> Metrics:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Metrics(precision=precision, recall=recall, f1=f1)


def grade_corpus(pairs: list[tuple[CaseSpec, IOCExtraction, IOCExtraction]]) -> Report:
    """Grade (spec, expected, actual) triples and aggregate.

    Per-type metrics are micro-averaged (TP/FP/FN pooled across cases):
    with 0-3 indicators per case, per-case ratios are noise, and pooled
    counts are what threshold-setting from baseline data operates on.
    """
    if not pairs:
        raise ValueError("cannot grade an empty corpus")
    grades = [grade_case(spec, expected, actual) for spec, expected, actual in pairs]

    per_type: dict[str, Metrics] = {}
    for t in INDICATOR_TYPES:
        per_type[t] = _metrics(
            tp=sum(g.counts[t].tp for g in grades),
            fp=sum(g.counts[t].fp for g in grades),
            fn=sum(g.counts[t].fn for g in grades),
        )

    by_category: dict[str, dict[str, TypeCounts]] = {}
    for category in sorted({g.category for g in grades}):
        in_cat = [g for g in grades if g.category == category]
        by_category[category] = {
            t: TypeCounts(
                tp=sum(g.counts[t].tp for g in in_cat),
                fp=sum(g.counts[t].fp for g in in_cat),
                fn=sum(g.counts[t].fn for g in in_cat),
            )
            for t in INDICATOR_TYPES
        }

    traps = [g for g in grades if g.role == "trap"]
    trap_fp_cases = sorted(
        g.case_id for g in traps if any(c.fp > 0 for c in g.counts.values())
    )
    trap_fp_case_rate = len(trap_fp_cases) / len(traps) if traps else None

    return Report(
        per_type=per_type,
        exact_match_rate=sum(g.exact_match for g in grades) / len(grades),
        by_category=by_category,
        trap_fp_case_rate=trap_fp_case_rate,
        trap_fp_cases=trap_fp_cases,
        cases=grades,
    )
