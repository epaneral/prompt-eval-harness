"""Tests for the normalization layer in ``harness.grader``.

The module docstring of ``harness.grader`` is the canonicalization policy;
these tests encode it. Canonical form is DEFANGED AT REST:

    domains   evil[.]com          (every dot bracketed, lowercased)
    ipv4      45[.]129[.]14[.]203 (every dot bracketed)
    hashes    lowercase, whitespace stripped
    urls      hxxp/hxxps scheme, host dots bracketed, path/query/fragment
              verbatim; bare-host trailing slash stripped

Every normalizer is expected to be idempotent: f(f(x)) == f(x).
"""

from __future__ import annotations

import pytest

from harness.grader import (
    normalize_domain,
    normalize_extraction,
    normalize_hash,
    normalize_ipv4,
    normalize_url,
    refang,
)
from harness.schema import IOCExtraction


# --------------------------------------------------------------------------
# refang: closed variant table + case-insensitivity
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hxxp://a", "http://a"),
        ("hxxps://a", "https://a"),
        ("a[.]b", "a.b"),
        ("a(.)b", "a.b"),
        ("a[://]b", "a://b"),
    ],
)
def test_refang_closed_variant_table(raw: str, expected: str) -> None:
    assert refang(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hXXp://a", "http://a"),
        ("HXXP://a", "http://a"),
    ],
)
def test_refang_case_insensitive(raw: str, expected: str) -> None:
    assert refang(raw) == expected


def test_refang_fully_uppercase_hxxps() -> None:
    """hxxps must refang as a unit: an hxxp-only pass would leave 'httpS'."""
    assert refang("HXXPS://a") == "https://a"


# --------------------------------------------------------------------------
# normalize_ipv4
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # live -> canonical defanged
        ("45.129.14.203", "45[.]129[.]14[.]203"),
        # already-defanged canonical input unchanged
        ("45[.]129[.]14[.]203", "45[.]129[.]14[.]203"),
        # mixed "(.)" style collapses to canonical "[.]"
        ("45(.)129(.)14(.)203", "45[.]129[.]14[.]203"),
        # surrounding whitespace stripped
        ("  45.129.14.203  ", "45[.]129[.]14[.]203"),
    ],
)
def test_normalize_ipv4(raw: str, expected: str) -> None:
    assert normalize_ipv4(raw) == expected


# --------------------------------------------------------------------------
# normalize_domain
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # lowercasing
        ("EVIL.COM", "evil[.]com"),
        # one trailing dot stripped
        ("evil.com.", "evil[.]com"),
        # defanged "(.)" input canonicalized
        ("evil(.)com", "evil[.]com"),
        # already-canonical input unchanged
        ("evil[.]com", "evil[.]com"),
        # uppercase + trailing dot together
        ("EVIL.COM.", "evil[.]com"),
    ],
)
def test_normalize_domain(raw: str, expected: str) -> None:
    assert normalize_domain(raw) == expected


# --------------------------------------------------------------------------
# normalize_hash
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ABCDEF0123  ", "abcdef0123"),
        ("AbCdEf", "abcdef"),
        ("\tDEADBEEF\n", "deadbeef"),
    ],
)
def test_normalize_hash(raw: str, expected: str) -> None:
    assert normalize_hash(raw) == expected


# --------------------------------------------------------------------------
# normalize_url
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # scheme+host lowercased, path case preserved, host dots bracketed,
        # path dots NOT bracketed
        (
            "HTTPS://EVIL.COM/Path/File.BIN",
            "hxxps://evil[.]com/Path/File.BIN",
        ),
        # http -> hxxp, https -> hxxps
        ("http://x.com/a", "hxxp://x[.]com/a"),
        ("https://x.com/a", "hxxps://x[.]com/a"),
        # bare-host trailing slash stripped
        ("http://x.com/", "hxxp://x[.]com"),
        # no path at all: nothing added
        ("http://x.com", "hxxp://x[.]com"),
        # non-bare path keeps its trailing slash
        ("http://x.com/a/", "hxxp://x[.]com/a/"),
        # trailing slash before a query is preserved (not a bare host)
        ("http://x.com/?q=1", "hxxp://x[.]com/?q=1"),
        # query preserved verbatim
        ("http://x.com/p?Q=Value&B=2", "hxxp://x[.]com/p?Q=Value&B=2"),
        # fragment preserved verbatim (case included)
        ("https://x.com/p#Frag", "hxxps://x[.]com/p#Frag"),
        # query + fragment together, verbatim
        ("http://x.com/?q=1#Frag", "hxxp://x[.]com/?q=1#Frag"),
        # already-defanged canonical URL unchanged
        (
            "hxxps://evil[.]com/Path/File.BIN",
            "hxxps://evil[.]com/Path/File.BIN",
        ),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


# --------------------------------------------------------------------------
# dedupe: normalize_extraction collapses variants and returns per-type sets
# --------------------------------------------------------------------------

def test_normalize_extraction_dedupes_variants() -> None:
    extraction = IOCExtraction(
        ipv4=["45.129.14.203", "45(.)129(.)14(.)203"],
        domains=["EVIL.COM", "evil[.]com", "evil.com."],
        urls=["HTTP://X.COM/", "hxxp://x[.]com"],
        hashes=["ABCDEF", "  abcdef  "],
    )
    result = normalize_extraction(extraction)

    # Each type returns a set with the variants collapsed to one entry.
    assert result == {
        "ipv4": {"45[.]129[.]14[.]203"},
        "domains": {"evil[.]com"},
        "urls": {"hxxp://x[.]com"},
        "hashes": {"abcdef"},
    }
    for value in result.values():
        assert isinstance(value, set)


def test_normalize_extraction_empty() -> None:
    result = normalize_extraction(
        IOCExtraction(ipv4=[], domains=[], urls=[], hashes=[])
    )
    assert result == {"ipv4": set(), "domains": set(), "urls": set(), "hashes": set()}


# --------------------------------------------------------------------------
# idempotency: f(f(x)) == f(x) for every normalizer across a spread of inputs
# --------------------------------------------------------------------------

_IDEMPOTENCY_CASES: list[tuple[str, str]] = [
    # (normalizer name, raw input) -- spread of live / defanged / mixed / upper
    ("normalize_ipv4", "45.129.14.203"),
    ("normalize_ipv4", "45[.]129[.]14[.]203"),
    ("normalize_ipv4", "45(.)129(.)14(.)203"),
    ("normalize_ipv4", "  45.129.14.203  "),
    ("normalize_domain", "EVIL.COM"),
    ("normalize_domain", "evil.com."),
    ("normalize_domain", "evil(.)com"),
    ("normalize_domain", "evil[.]com"),
    ("normalize_hash", "  ABCDEF  "),
    ("normalize_hash", "deadbeef"),
    ("normalize_url", "HTTPS://EVIL.COM/Path/File.BIN"),
    ("normalize_url", "http://x.com/"),
    ("normalize_url", "http://x.com/a/"),
    ("normalize_url", "http://x.com/?q=1#Frag"),
    ("normalize_url", "hxxps://evil[.]com/Path"),
    ("refang", "hXXp://a[.]b"),
    ("refang", "HXXP://a(.)b"),
    ("refang", "already.live/path"),
]

_NORMALIZERS = {
    "refang": refang,
    "normalize_ipv4": normalize_ipv4,
    "normalize_domain": normalize_domain,
    "normalize_hash": normalize_hash,
    "normalize_url": normalize_url,
}


@pytest.mark.parametrize(("func_name", "raw"), _IDEMPOTENCY_CASES)
def test_idempotency(func_name: str, raw: str) -> None:
    func = _NORMALIZERS[func_name]
    once = func(raw)
    twice = func(once)
    assert once == twice
