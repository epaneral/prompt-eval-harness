"""Mechanically links the prompt's documented output example to the schema.

The prompt in prompts/v1.md advertises an example output shape. If that
example ever drifts from harness/schema.IOCExtraction, this test fails,
so the docs and the contract cannot silently diverge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harness.schema import IOCExtraction

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "v1.md"

_JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def test_prompt_json_example_matches_schema() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    blocks = _JSON_BLOCK.findall(text)
    assert len(blocks) == 1, f"expected exactly one ```json block, found {len(blocks)}"
    payload = json.loads(blocks[0])
    IOCExtraction.model_validate(payload)
