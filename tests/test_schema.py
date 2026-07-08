"""Tests for the data contracts in harness/schema.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.schema import CaseSpec, IOCExtraction


def _valid_ioc_payload() -> dict[str, list[str]]:
    return {
        "ipv4": ["91[.]212[.]166[.]44"],
        "domains": ["update-checker[.]net"],
        "urls": ["hxxps://update-checker[.]net/a"],
        "hashes": ["44d88612fea8a8f36de82e1278abb02f"],
    }


def _valid_case_entry() -> dict[str, str]:
    return {
        "id": "case-001",
        "category": "phishing",
        "role": "positive",
        "twin_id": "case-002",
        "input": "inputs/case-001.txt",
        "expected": "expected/case-001.json",
    }


def test_valid_payload_parses() -> None:
    model = IOCExtraction.model_validate(_valid_ioc_payload())
    assert model.ipv4 == ["91[.]212[.]166[.]44"]
    assert model.domains == ["update-checker[.]net"]
    assert model.urls == ["hxxps://update-checker[.]net/a"]
    assert model.hashes == ["44d88612fea8a8f36de82e1278abb02f"]


def test_extra_key_rejected() -> None:
    payload = _valid_ioc_payload()
    payload["emails"] = ["a@b.com"]
    with pytest.raises(ValidationError):
        IOCExtraction.model_validate(payload)


def test_missing_key_rejected() -> None:
    payload = _valid_ioc_payload()
    del payload["hashes"]
    with pytest.raises(ValidationError):
        IOCExtraction.model_validate(payload)


def test_non_list_value_rejected() -> None:
    payload = _valid_ioc_payload()
    payload["domains"] = "update-checker[.]net"
    with pytest.raises(ValidationError):
        IOCExtraction.model_validate(payload)


def test_json_round_trip_equal() -> None:
    """model_dump_json -> model_validate_json reconstructs an equal model."""
    original = IOCExtraction.model_validate(_valid_ioc_payload())
    restored = IOCExtraction.model_validate_json(original.model_dump_json())
    assert restored == original


def test_casespec_valid_entry_parses() -> None:
    spec = CaseSpec.model_validate(_valid_case_entry())
    assert spec.id == "case-001"
    assert spec.role == "positive"


def test_casespec_unknown_field_rejected() -> None:
    entry = _valid_case_entry()
    entry["severity"] = "high"
    with pytest.raises(ValidationError):
        CaseSpec.model_validate(entry)


def test_casespec_role_outside_literal_rejected() -> None:
    entry = _valid_case_entry()
    entry["role"] = "negative"
    with pytest.raises(ValidationError):
        CaseSpec.model_validate(entry)
