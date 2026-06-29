#!/usr/bin/env python3
"""CLI entry point for Piper dataset generation via Qwen3-TTS."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voice_cloning.cli_args import build_arg_parser, run_from_args  # noqa: E402


def main() -> None:
    run_from_args(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
