"""Shared CLI argument parsing."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Piper-compatible TTS dataset using Qwen3-TTS voice cloning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        metavar="N",
        help="Number of utterances to generate (overrides total_entries in config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="TTS sampling temperature (overrides tts.temperature in config)",
    )
    parser.add_argument(
        "--force-rewrite",
        action="store_true",
        help="Regenerate all utterances instead of resuming skipped WAVs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run_from_args(args: argparse.Namespace) -> Path:
    import logging
    from dataclasses import replace

    from voice_cloning.dataset_generator import generate_dataset, load_config

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if args.samples is not None:
        if args.samples < 1:
            raise ValueError("--samples must be at least 1")
        config = replace(config, total_entries=args.samples)
    if args.temperature is not None:
        config = replace(config, temperature=args.temperature)
    if args.force_rewrite:
        config = replace(config, resume=False)

    dataset_dir = generate_dataset(config)
    print(f"Dataset ready: {dataset_dir}")
    return dataset_dir
