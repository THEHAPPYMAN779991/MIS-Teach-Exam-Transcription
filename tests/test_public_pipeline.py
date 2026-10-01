"""Static public-package checks; no cloud request or source document is required."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "pipeline.py"


class PublicPipelineTests(unittest.TestCase):
    def test_pipeline_is_syntax_valid_and_has_public_cli(self) -> None:
        source = PIPELINE.read_text(encoding="utf-8")
        compile(source, str(PIPELINE), "exec")
        for option in ("--pdf-folder", "--output-root", "--preflight"):
            self.assertIn(option, source)

    def test_pipeline_contains_no_google_api_key_literal(self) -> None:
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"AIza[0-9A-Za-z_-]{20,}", source))


if __name__ == "__main__":
    unittest.main()
