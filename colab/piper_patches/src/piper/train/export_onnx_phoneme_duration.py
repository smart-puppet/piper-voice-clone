#!/usr/bin/env python3
"""Export ONNX with audio + phoneme duration (w_ceil) outputs.

piper1-gpl's VITS infer() does not return w_ceil; patch the standard ONNX export
with piper.patch_voice_with_alignment instead (same as PiperVoice include_alignments).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import onnx

from piper.patch_voice_with_alignment import add_alignment_output

_LOGGER = logging.getLogger(__name__)


def export_phoneme_duration_onnx(*, checkpoint: Path, output_file: Path) -> Path:
    """Export standard Piper ONNX, then expose w_ceil as a second graph output."""
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint.resolve()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.train.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--output-file",
            str(output_file),
        ],
        check=True,
    )

    model = onnx.load(str(output_file))
    tensor_name = add_alignment_output(model)
    onnx.save(model, str(output_file))
    _LOGGER.info(
        "Exported model with alignments to %s (extra output: %s)",
        output_file,
        tensor_name,
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to output file (.onnx)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    export_phoneme_duration_onnx(
        checkpoint=Path(args.checkpoint),
        output_file=Path(args.output_file),
    )


if __name__ == "__main__":
    main()
