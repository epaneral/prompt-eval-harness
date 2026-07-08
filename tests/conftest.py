"""Shared fixtures for the corpus consistency suite.

The manifest and corpus root are loaded once per session so that the
whole suite operates on a single, consistent view of the corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.schema import CaseSpec, load_manifest

CORPUS_ROOT: Path = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="session")
def corpus_root() -> Path:
    """Absolute path to the corpus/ directory, resolved from this test file."""
    return CORPUS_ROOT


@pytest.fixture(scope="session")
def manifest() -> list[CaseSpec]:
    """Parsed corpus/manifest.yaml, loaded and validated once per session."""
    return load_manifest(CORPUS_ROOT / "manifest.yaml")
