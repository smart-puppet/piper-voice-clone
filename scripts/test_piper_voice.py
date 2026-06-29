#!/usr/bin/env python3
"""Quick test for a fine-tuned Piper ONNX voice — text → WAV."""

from __future__ import annotations

import argparse
import logging
import shutil
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "myvoice-de-medium.onnx"
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "datasets" / "myvoice-de_v03" / "config.json"
DEFAULT_TEXT = (
    "Hallo! Das ist ein kurzer Test, um die Stimme zu überprüfen."
)


def resolve_config_path(model_path: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path.resolve()

    sidecar = model_path.with_suffix(model_path.suffix + ".json")
    if sidecar.is_file():
        return sidecar

    if DEFAULT_DATASET_CONFIG.is_file():
        return DEFAULT_DATASET_CONFIG

    raise SystemExit(
        "No voice config found. Pass --config or copy the training config next to the model:\n"
        f"  cp datasets/myvoice-de_v03/config.json {sidecar}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test a Piper ONNX voice (after export).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Defaults: models/myvoice-de-medium.onnx + dataset config.json
  python scripts/test_piper_voice.py

  python scripts/test_piper_voice.py \\
    --model ../../models/myvoice-de-medium.onnx \\
    --config ../../datasets/myvoice-de_v03/config.json \\
    --text "Guten Morgen! Wie geht es dir heute?"
        """,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to exported .onnx model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Voice JSON config (default: <model>.onnx.json or dataset config.json)",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="German text to synthesize")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("test_piper_output.wav"),
        help="Output WAV path (22.05 kHz)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use ONNX Runtime CUDA provider")
    parser.add_argument(
        "--length-scale",
        type=float,
        help="Speaking rate (higher = slower; default from config)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        help="Generator noise (default from config)",
    )
    parser.add_argument(
        "--install-config",
        action="store_true",
        help="Copy dataset config to <model>.onnx.json if missing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    sidecar = model_path.with_suffix(model_path.suffix + ".json")
    if args.install_config and not sidecar.is_file() and DEFAULT_DATASET_CONFIG.is_file():
        shutil.copy2(DEFAULT_DATASET_CONFIG, sidecar)
        logging.info("Installed voice config at %s", sidecar)

    config_path = resolve_config_path(model_path, args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {model_path}")
    print(f"Config : {config_path}")
    print(f"Text   : {args.text}")
    print(f"Output : {output}")

    voice = PiperVoice.load(
        model_path,
        config_path=config_path,
        use_cuda=args.cuda,
    )

    syn_config = SynthesisConfig()
    if args.length_scale is not None:
        syn_config.length_scale = args.length_scale
    if args.noise_scale is not None:
        syn_config.noise_scale = args.noise_scale

    logging.info("Synthesizing...")
    with wave.open(str(output), "wb") as wav_file:
        voice.synthesize_wav(args.text, wav_file, syn_config=syn_config)

    with wave.open(str(output), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    duration_s = frames / sample_rate
    print(f"Done — wrote {duration_s:.1f}s to {output}")
    print("Play with: ffplay -nodisp -autoexit", output)


if __name__ == "__main__":
    main()
