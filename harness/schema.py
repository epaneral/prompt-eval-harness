"""Data contracts: model output and corpus manifest.

The output schema is deliberately structural-only (no per-item format
regexes): a malformed indicator value should surface as a grading false
positive, not a schema failure, so that schema validity stays a pure
"output contract broken" signal for the CI gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class IOCExtraction(BaseModel):
    """Output contract for the extraction prompt; also the format of corpus/expected/*.json.

    All four keys are required with no defaults so that "nothing found"
    must be an explicit empty list -- a missing key is a contract
    violation, not an implicit empty result.
    """

    model_config = ConfigDict(extra="forbid")

    ipv4: list[str]
    domains: list[str]
    urls: list[str]
    hashes: list[str]


class CaseSpec(BaseModel):
    """One corpus manifest entry.

    ``role`` exists so the near-miss twin set is mechanically
    identifiable (the false-extraction gate needs it); an id naming
    convention would be an implicit contract, which is easier to break
    and harder to review.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    role: Literal["positive", "trap"]
    twin_id: str
    input: str  # path relative to corpus/
    expected: str  # path relative to corpus/


def load_manifest(path: Path) -> list[CaseSpec]:
    """Load and validate corpus/manifest.yaml."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [CaseSpec(**entry) for entry in data["cases"]]
