"""Run the public exam-transcription pipeline from its normalized entry point."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    pipeline = Path(__file__).resolve().parents[1] / "src" / "pipeline.py"
    runpy.run_path(str(pipeline), run_name="__main__")
