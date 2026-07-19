"""Corpus consistency suite -- the owner's live-editing safety net.

Every check here is mechanical and reads only the ratified policy from
harness/grader.py (canonicalization + the judgment policies documented
in its module docstring) and harness/schema.py (the data contracts). No
check ever mutates the corpus; a corpus that violates the policy is a
test failure to be reported and fixed by hand, never papered over.

After the owner edits any corpus case, ``pytest`` must instantly
confirm the corpus is still internally consistent and policy-conformant.
"""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

from harness.grader import (
    normalize_domain,
    normalize_hash,
    normalize_ipv4,
    normalize_url,
    refang,
)
from harness.schema import CaseSpec, IOCExtraction

CATEGORIES: frozenset[str] = frozenset(
    {
        "defanged",
        "version_strings",
        "hash_lookalikes",
        "benign_context_domains",
        "reserved_ranges",
    }
)

# Per-indicator-type normalizers from the grader, keyed by IOCExtraction field.
NORMALIZERS = {
    "ipv4": normalize_ipv4,
    "domains": normalize_domain,
    "urls": normalize_url,
    "hashes": normalize_hash,
}

# RFC 2606 / 6761 reserved names the judgment policy excludes.
_RESERVED_DOMAINS: frozenset[str] = frozenset(
    {"example.com", "example.net", "example.org", "localhost"}
)
_RESERVED_TLDS: tuple[str, ...] = (".test", ".invalid")


def _load_expected(corpus_root: Path, spec: CaseSpec) -> IOCExtraction:
    """Parse one case's expected file into an IOCExtraction."""
    payload = json.loads((corpus_root / spec.expected).read_text(encoding="utf-8"))
    return IOCExtraction.model_validate(payload)


def _url_host(url: str) -> str:
    """Refang a stored URL and return its lowercased hostname."""
    return urlsplit(refang(url)).hostname or ""


# --- 1. ids unique ----------------------------------------------------------


def test_ids_unique(manifest: list[CaseSpec]) -> None:
    ids = [spec.id for spec in manifest]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate case ids: {duplicates}"


# --- 2. categories in the closed set ---------------------------------------


def test_categories_in_closed_set(manifest: list[CaseSpec]) -> None:
    unknown = sorted(
        {spec.id: spec.category for spec in manifest if spec.category not in CATEGORIES}.items()
    )
    assert not unknown, f"cases with categories outside the closed set: {unknown}"


# --- 3. twin links ----------------------------------------------------------


def test_no_case_is_its_own_twin(manifest: list[CaseSpec]) -> None:
    self_twins = sorted(spec.id for spec in manifest if spec.twin_id == spec.id)
    assert not self_twins, f"cases that are their own twin: {self_twins}"


def test_twin_ids_resolve(manifest: list[CaseSpec]) -> None:
    ids = {spec.id for spec in manifest}
    dangling = sorted(
        (spec.id, spec.twin_id) for spec in manifest if spec.twin_id not in ids
    )
    assert not dangling, f"twin_ids pointing at nonexistent cases: {dangling}"


def test_twin_links_symmetric(manifest: list[CaseSpec]) -> None:
    by_id = {spec.id: spec for spec in manifest}
    asymmetric = sorted(
        spec.id
        for spec in manifest
        if spec.twin_id in by_id and by_id[spec.twin_id].twin_id != spec.id
    )
    assert not asymmetric, f"cases whose twin does not point back at them: {asymmetric}"


def test_twin_pairs_are_one_positive_one_trap(manifest: list[CaseSpec]) -> None:
    by_id = {spec.id: spec for spec in manifest}
    bad_pairs = sorted(
        tuple(sorted((spec.id, spec.twin_id)))
        for spec in manifest
        if spec.twin_id in by_id
        and {spec.role, by_id[spec.twin_id].role} != {"positive", "trap"}
    )
    # sorted(set(...)) so each pair is reported once, not once per member.
    unique_bad = sorted(set(bad_pairs))
    assert not unique_bad, f"twin pairs that are not exactly one positive + one trap: {unique_bad}"


# --- 4. pair count between 20 and 25 ----------------------------------------


def test_pair_count_in_range(manifest: list[CaseSpec]) -> None:
    case_count = len(manifest)
    assert case_count % 2 == 0, f"odd case count {case_count}; twin pairs cannot be complete"
    pair_count = case_count // 2
    assert 20 <= pair_count <= 25, f"pair count {pair_count} outside [20, 25]"


# --- 5. referenced files exist ----------------------------------------------


def test_referenced_files_exist(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    missing: list[str] = []
    for spec in manifest:
        for rel in (spec.input, spec.expected):
            if not (corpus_root / rel).is_file():
                missing.append(f"{spec.id}: {rel}")
    assert not missing, f"manifest references files that do not exist: {sorted(missing)}"


# --- 6. no orphan files -----------------------------------------------------


def test_no_orphan_files(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    referenced_inputs = [spec.input for spec in manifest]
    referenced_expected = [spec.expected for spec in manifest]

    input_files = {
        f"inputs/{p.name}" for p in (corpus_root / "inputs").glob("*.txt")
    }
    expected_files = {
        f"expected/{p.name}" for p in (corpus_root / "expected").glob("*.json")
    }

    orphan_inputs = sorted(input_files - set(referenced_inputs))
    orphan_expected = sorted(expected_files - set(referenced_expected))
    assert not orphan_inputs, f"unreferenced input files: {orphan_inputs}"
    assert not orphan_expected, f"unreferenced expected files: {orphan_expected}"

    # "referenced by exactly one entry" -- catch a file cited by two cases.
    dup_inputs = sorted({r for r in referenced_inputs if referenced_inputs.count(r) > 1})
    dup_expected = sorted(
        {r for r in referenced_expected if referenced_expected.count(r) > 1}
    )
    assert not dup_inputs, f"input files referenced by more than one entry: {dup_inputs}"
    assert not dup_expected, f"expected files referenced by more than one entry: {dup_expected}"


# --- 7. expected files parse as IOCExtraction -------------------------------


def test_expected_files_parse(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    for spec in manifest:
        # model_validate raises on any contract violation; the case id in
        # the assertion frame makes a failure pinpoint the offending file.
        _load_expected(corpus_root, spec)


# --- 8. expected files are already in canonical form ------------------------


def test_expected_files_canonical(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    violations: list[str] = []
    for spec in manifest:
        expected = _load_expected(corpus_root, spec)
        for field, normalize in NORMALIZERS.items():
            stored: list[str] = getattr(expected, field)
            canonical = sorted({normalize(v) for v in stored})
            if stored != canonical:
                violations.append(
                    f"{spec.id}.{field}: stored={stored!r} canonical={canonical!r}"
                )
    assert not violations, "expected files not in canonical form:\n" + "\n".join(violations)


# --- 9. policy consistency (judgment side) ----------------------------------


def test_expected_ipv4_valid_and_global(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    violations: list[str] = []
    for spec in manifest:
        expected = _load_expected(corpus_root, spec)
        for value in expected.ipv4:
            live = refang(value)
            try:
                addr = ipaddress.IPv4Address(live)
            except ValueError as exc:
                violations.append(f"{spec.id}: {value!r} is not a valid IPv4 ({exc})")
                continue
            if not addr.is_global:
                violations.append(f"{spec.id}: {value!r} ({addr}) is non-global")
    assert not violations, "expected ipv4 values violate the global-only policy:\n" + "\n".join(
        violations
    )


def test_expected_domains_not_reserved(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    violations: list[str] = []
    for spec in manifest:
        expected = _load_expected(corpus_root, spec)
        for value in expected.domains:
            host = refang(value).strip().lower().removesuffix(".")
            is_reserved = (
                host in _RESERVED_DOMAINS
                or any(host.endswith("." + name) for name in _RESERVED_DOMAINS)
                or host.endswith(_RESERVED_TLDS)
            )
            if is_reserved:
                violations.append(f"{spec.id}: {value!r} is a reserved name")
    assert not violations, "expected domains violate the reserved-name policy:\n" + "\n".join(
        violations
    )


def test_expected_url_hosts_duplicated(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    violations: list[str] = []
    for spec in manifest:
        expected = _load_expected(corpus_root, spec)
        domain_hosts = {normalize_domain(d) for d in expected.domains}
        ipv4_hosts = {normalize_ipv4(i) for i in expected.ipv4}
        for url in expected.urls:
            host = _url_host(url)
            if not host:
                violations.append(f"{spec.id}: URL {url!r} has no parseable host")
                continue
            try:
                ipaddress.IPv4Address(host)
                is_ip = True
            except ValueError:
                is_ip = False
            if is_ip:
                if normalize_ipv4(host) not in ipv4_hosts:
                    violations.append(
                        f"{spec.id}: URL host {host!r} (IP) not in ipv4 list"
                    )
            else:
                if normalize_domain(host) not in domain_hosts:
                    violations.append(
                        f"{spec.id}: URL host {host!r} not in domains list"
                    )
    assert not violations, "expected URLs violate the host-duplication policy:\n" + "\n".join(
        violations
    )


# --- 10. input files are non-empty text -------------------------------------


def test_input_files_non_empty(corpus_root: Path, manifest: list[CaseSpec]) -> None:
    empty: list[str] = []
    for spec in manifest:
        text = (corpus_root / spec.input).read_text(encoding="utf-8")
        if not text.strip():
            empty.append(f"{spec.id}: {spec.input}")
    assert not empty, f"input files that are empty or whitespace-only: {sorted(empty)}"


# --- 11. every expected file has at least one indicator ----------------------


def test_expected_files_have_at_least_one_indicator(
    corpus_root: Path, manifest: list[CaseSpec]
) -> None:
    """Grading-policy invariant, not just corpus hygiene: a schema-invalid
    response grades as an EMPTY extraction, so a case whose expected
    output were entirely empty would score an unparseable response as an
    exact match. The corpus was drafted so every case expects at least
    one indicator; this test converts that from accident to invariant.
    """
    empty: list[str] = []
    for spec in manifest:
        expected = _load_expected(corpus_root, spec)
        if not (expected.ipv4 or expected.domains or expected.urls or expected.hashes):
            empty.append(spec.id)
    assert not empty, f"cases whose expected output is entirely empty: {empty}"
