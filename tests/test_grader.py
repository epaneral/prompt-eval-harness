"""Tests for grade_case / grade_corpus in harness/grader.py.

Fixtures are built INLINE in each test so the hand-computed expected
counts sit next to the data that produces them. Normalization is
exercised implicitly: several actuals are written live (e.g. 45.129.14.203,
https://Evil.COM/Path) while the expected is written defanged, and the
tests assert they still match -- so a regression in canonicalization
would show up as a spurious false positive / false negative here.
"""

from __future__ import annotations

import pytest

from harness.grader import grade_case, grade_corpus
from harness.schema import CaseSpec, IOCExtraction


def make_spec(
    case_id: str,
    category: str = "phishing",
    role: str = "positive",
    twin_id: str = "twin",
) -> CaseSpec:
    """Build a CaseSpec with dummy (unused-by-grader) input/expected paths."""
    return CaseSpec(
        id=case_id,
        category=category,
        role=role,
        twin_id=twin_id,
        input=f"inputs/{case_id}.txt",
        expected=f"expected/{case_id}.json",
    )


def empty_extraction() -> IOCExtraction:
    return IOCExtraction(ipv4=[], domains=[], urls=[], hashes=[])


# --------------------------------------------------------------------------
# grade_case
# --------------------------------------------------------------------------


def test_perfect_match_multitype_live_vs_defanged() -> None:
    """Every type populated; actual written live, expected defanged; they match.

    Expected has 1 ipv4, 1 domain, 1 url, 1 hash. Actual expresses the
    same four indicators in live / mixed-case form. After normalization
    both sides are identical -> tp = 1 per type, fp == fn == 0, exact.
    """
    spec = make_spec("case-perfect")
    expected = IOCExtraction(
        ipv4=["45[.]129[.]14[.]203"],
        domains=["evil[.]com"],
        urls=["hxxps://evil[.]com/Path"],
        hashes=["44d88612fea8a8f36de82e1278abb02f"],
    )
    actual = IOCExtraction(
        ipv4=["45.129.14.203"],  # live dots
        domains=["Evil.COM."],  # live, mixed case, trailing dot
        urls=["https://Evil.COM/Path"],  # live scheme+host, path verbatim
        hashes=["44D88612FEA8A8F36DE82E1278ABB02F"],  # uppercase
    )

    grade = grade_case(spec, expected, actual)

    assert grade.case_id == "case-perfect"
    assert grade.category == "phishing"
    assert grade.role == "positive"
    for t in ("ipv4", "domains", "urls", "hashes"):
        assert grade.counts[t].tp == 1
        assert grade.counts[t].fp == 0
        assert grade.counts[t].fn == 0
        assert grade.false_extractions[t] == []
        assert grade.missed[t] == []
    assert grade.exact_match is True


def test_one_false_positive_domain() -> None:
    """Actual has one extra domain not in expected.

    Expected domains: {evil[.]com}. Actual domains: {evil[.]com, extra[.]net}.
    -> tp = 1, fp = 1 for domains; extra[.]net listed in canonical form.
    """
    spec = make_spec("case-fp")
    expected = IOCExtraction(
        ipv4=[], domains=["evil[.]com"], urls=[], hashes=[]
    )
    actual = IOCExtraction(
        ipv4=[], domains=["evil.com", "Extra.NET"], urls=[], hashes=[]
    )

    grade = grade_case(spec, expected, actual)

    assert grade.counts["domains"].tp == 1
    assert grade.counts["domains"].fp == 1
    assert grade.counts["domains"].fn == 0
    # Canonical form of "Extra.NET" is "extra[.]net".
    assert grade.false_extractions["domains"] == ["extra[.]net"]
    assert grade.missed["domains"] == []
    assert grade.exact_match is False


def test_one_false_negative_hash() -> None:
    """Actual misses one of two expected hashes.

    Expected hashes: {aaa..., bbb...}. Actual hashes: {aaa...}.
    -> tp = 1, fn = 1 for hashes; the missed hash is listed.
    """
    spec = make_spec("case-fn")
    hash_a = "a" * 32
    hash_b = "b" * 40
    expected = IOCExtraction(
        ipv4=[], domains=[], urls=[], hashes=[hash_a, hash_b]
    )
    actual = IOCExtraction(
        ipv4=[], domains=[], urls=[], hashes=[hash_a.upper()]  # uppercase still matches a
    )

    grade = grade_case(spec, expected, actual)

    assert grade.counts["hashes"].tp == 1
    assert grade.counts["hashes"].fp == 0
    assert grade.counts["hashes"].fn == 1
    assert grade.missed["hashes"] == [hash_b]
    assert grade.false_extractions["hashes"] == []
    assert grade.exact_match is False


def test_empty_vs_empty_exact_match() -> None:
    """All four lists empty on both sides -> exact match, zero counts."""
    spec = make_spec("case-empty")
    grade = grade_case(spec, empty_extraction(), empty_extraction())

    for t in ("ipv4", "domains", "urls", "hashes"):
        assert grade.counts[t].tp == 0
        assert grade.counts[t].fp == 0
        assert grade.counts[t].fn == 0
    assert grade.exact_match is True


def test_false_and_missed_lists_are_sorted() -> None:
    """Multiple fp / fn values must come back sorted (deterministic output)."""
    spec = make_spec("case-sorted")
    expected = IOCExtraction(
        ipv4=[],
        domains=["keep[.]com", "gone-a[.]com", "gone-b[.]com"],
        urls=[],
        hashes=[],
    )
    actual = IOCExtraction(
        ipv4=[],
        domains=["keep[.]com", "zzz[.]com", "aaa[.]com", "mmm[.]com"],
        urls=[],
        hashes=[],
    )

    grade = grade_case(spec, expected, actual)

    # False positives: the three actual-only domains, canonical + sorted.
    assert grade.false_extractions["domains"] == ["aaa[.]com", "mmm[.]com", "zzz[.]com"]
    # Missed: the two expected-only domains, sorted.
    assert grade.missed["domains"] == ["gone-a[.]com", "gone-b[.]com"]


# --------------------------------------------------------------------------
# grade_corpus: metrics, aggregation, traps, categories
# --------------------------------------------------------------------------


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError):
        grade_corpus([])


def test_empty_case_metrics_are_none_not_coerced() -> None:
    """A single empty-vs-empty case yields no tp/fp/fn anywhere.

    With a zero denominator, precision/recall/f1 must be None for every
    type -- never silently coerced to 0.0 or 1.0.
    """
    spec = make_spec("case-empty")
    report = grade_corpus([(spec, empty_extraction(), empty_extraction())])

    for t in ("ipv4", "domains", "urls", "hashes"):
        m = report.per_type[t]
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None
    assert report.exact_match_rate == 1.0


def test_micro_aggregation_pooled_precision_recall_f1() -> None:
    """Two hand-built cases; per-type metrics are micro-averaged (pooled).

    Focus on the domains type, hand-computing the pooled counts.

    Case 1 domains:
        expected {a[.]com, b[.]com}, actual {a[.]com, x[.]com}
        -> tp=1 (a), fp=1 (x), fn=1 (b)
    Case 2 domains:
        expected {c[.]com}, actual {c[.]com, y[.]com, z[.]com}
        -> tp=1 (c), fp=2 (y,z), fn=0

    Pooled domains: tp=2, fp=3, fn=1
        precision = 2 / (2+3) = 0.4
        recall    = 2 / (2+1) = 0.666...
        f1        = 2*p*r/(p+r) = 2*0.4*(2/3) / (0.4 + 2/3) = 0.5
    """
    spec1 = make_spec("case-1")
    spec2 = make_spec("case-2")

    case1_expected = IOCExtraction(
        ipv4=[], domains=["a[.]com", "b[.]com"], urls=[], hashes=[]
    )
    case1_actual = IOCExtraction(
        ipv4=[], domains=["a[.]com", "x[.]com"], urls=[], hashes=[]
    )
    case2_expected = IOCExtraction(
        ipv4=[], domains=["c[.]com"], urls=[], hashes=[]
    )
    case2_actual = IOCExtraction(
        ipv4=[], domains=["c[.]com", "y[.]com", "z[.]com"], urls=[], hashes=[]
    )

    report = grade_corpus(
        [
            (spec1, case1_expected, case1_actual),
            (spec2, case2_expected, case2_actual),
        ]
    )

    tp, fp, fn = 2, 3, 1
    expected_precision = tp / (tp + fp)  # 0.4
    expected_recall = tp / (tp + fn)  # 0.666...
    expected_f1 = 2 * expected_precision * expected_recall / (
        expected_precision + expected_recall
    )  # 0.5

    m = report.per_type["domains"]
    assert m.precision == pytest.approx(expected_precision)
    assert m.recall == pytest.approx(expected_recall)
    assert m.f1 == pytest.approx(expected_f1)
    assert m.f1 == pytest.approx(0.5)

    # Types with no data on either side stay None (not coerced).
    for t in ("ipv4", "urls", "hashes"):
        assert report.per_type[t].precision is None
        assert report.per_type[t].recall is None
        assert report.per_type[t].f1 is None


def test_trap_fp_case_rate_and_offending_ids() -> None:
    """2 positives + 2 traps; exactly one trap has a false positive.

    Trap rate is over TRAP cases only (2 traps), so one offending trap
    -> rate == 0.5, and trap_fp_cases lists exactly that trap id. A
    false positive on a positive case must NOT count toward the rate.
    """
    pos_clean = make_spec("pos-clean", role="positive")
    pos_dirty = make_spec("pos-dirty", role="positive")
    trap_clean = make_spec("trap-clean", role="trap")
    trap_dirty = make_spec("trap-dirty", role="trap")

    clean_exp = IOCExtraction(ipv4=[], domains=["good[.]com"], urls=[], hashes=[])
    clean_act = IOCExtraction(ipv4=[], domains=["good[.]com"], urls=[], hashes=[])
    # A false positive: one extra domain beyond expected.
    dirty_act = IOCExtraction(
        ipv4=[], domains=["good[.]com", "bad[.]com"], urls=[], hashes=[]
    )

    report = grade_corpus(
        [
            (pos_clean, clean_exp, clean_act),
            (pos_dirty, clean_exp, dirty_act),  # fp on a POSITIVE: ignored by trap rate
            (trap_clean, clean_exp, clean_act),
            (trap_dirty, clean_exp, dirty_act),  # fp on a TRAP: counts
        ]
    )

    assert report.trap_fp_case_rate == pytest.approx(0.5)
    assert report.trap_fp_cases == ["trap-dirty"]


def test_trap_fp_case_rate_none_when_no_traps() -> None:
    """A corpus with only positive cases has an undefined trap rate (None)."""
    spec = make_spec("pos-only", role="positive")
    exp = IOCExtraction(ipv4=[], domains=["good[.]com"], urls=[], hashes=[])
    report = grade_corpus([(spec, exp, exp)])

    assert report.trap_fp_case_rate is None
    assert report.trap_fp_cases == []


def test_by_category_counts_keyed_and_summed() -> None:
    """Two categories; counts keyed by category and equal to hand-sums.

    Category "phishing" has two cases; category "malware" has one.

    phishing case A domains: exp {a[.]com}, act {a[.]com}          -> tp=1
    phishing case B domains: exp {b[.]com}, act {b[.]com, x[.]com}  -> tp=1 fp=1
        pooled phishing domains: tp=2, fp=1, fn=0
    malware case C hashes: exp {h1, h2}, act {h1}                   -> tp=1 fn=1
        pooled malware hashes: tp=1, fp=0, fn=1
    """
    phish_a = make_spec("phish-a", category="phishing")
    phish_b = make_spec("phish-b", category="phishing")
    malw_c = make_spec("malw-c", category="malware")

    phish_a_exp = IOCExtraction(ipv4=[], domains=["a[.]com"], urls=[], hashes=[])
    phish_a_act = IOCExtraction(ipv4=[], domains=["a[.]com"], urls=[], hashes=[])
    phish_b_exp = IOCExtraction(ipv4=[], domains=["b[.]com"], urls=[], hashes=[])
    phish_b_act = IOCExtraction(
        ipv4=[], domains=["b[.]com", "x[.]com"], urls=[], hashes=[]
    )
    h1, h2 = "1" * 32, "2" * 32
    malw_c_exp = IOCExtraction(ipv4=[], domains=[], urls=[], hashes=[h1, h2])
    malw_c_act = IOCExtraction(ipv4=[], domains=[], urls=[], hashes=[h1])

    report = grade_corpus(
        [
            (phish_a, phish_a_exp, phish_a_act),
            (phish_b, phish_b_exp, phish_b_act),
            (malw_c, malw_c_exp, malw_c_act),
        ]
    )

    assert set(report.by_category) == {"phishing", "malware"}

    phishing_domains = report.by_category["phishing"]["domains"]
    assert (phishing_domains.tp, phishing_domains.fp, phishing_domains.fn) == (2, 1, 0)

    malware_hashes = report.by_category["malware"]["hashes"]
    assert (malware_hashes.tp, malware_hashes.fp, malware_hashes.fn) == (1, 0, 1)

    # Cross-category leakage check: malware category has no domain activity.
    malware_domains = report.by_category["malware"]["domains"]
    assert (malware_domains.tp, malware_domains.fp, malware_domains.fn) == (0, 0, 0)
