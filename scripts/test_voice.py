#!/usr/bin/env python3
"""Quick Qwen3-TTS voice-cloning test — reference WAV + text → output WAV."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import soundfile as sf

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voice_cloning.dataset_generator import (  # noqa: E402
    PROJECT_ROOT as DS_PROJECT_ROOT,
    load_config,
)
from voice_cloning.config_paths import read_text_file, require_file  # noqa: E402
from voice_cloning.faster_qwen3tts_engine import (
    FasterQwen3TTSEngine,
    resolve_hf_model,
)
from voice_cloning.qwen3tts_engine import (  # noqa: E402
    QWEN3_SAMPLE_RATE,
    Qwen3TTSEngine,
    ensure_qwen3_models,
    resolve_model_dir,
    resolve_qwen_tts_bin,
)

DEFAULT_TEXT = (
    "Hallo! Das ist ein kurzer Test, um die Stimme zu überprüfen."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Qwen3-TTS voice cloning (qwentts.cpp + GGUF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_voice.py \\
    --ref-wav references/my-voice.wav \\
    --ref-text-file references/my-voice.txt \\
    --text "Guten Morgen, wie geht es dir?"

  python scripts/test_voice.py --config config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Reuse tts/model settings from YAML config",
    )
    parser.add_argument("--ref-wav", type=Path, help="Reference audio WAV")
    parser.add_argument("--ref-text", help="Exact transcript of the reference audio")
    parser.add_argument("--ref-text-file", type=Path, help="File with reference transcript")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="German text to synthesize")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("test_output.wav"),
        help="Output WAV path (24 kHz)",
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

    if args.config:
        try:
            cfg = load_config(args.config.resolve())
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        ref_wav = args.ref_wav or cfg.reference.wav_path
        ref_text = args.ref_text
        if not ref_text and args.ref_text_file:
            ref_text = read_text_file(
                args.ref_text_file.resolve(),
                label="reference transcript",
            )
        if not ref_text:
            ref_text = cfg.reference.ref_text
        active_wav = ref_wav

        if cfg.backend == "faster":
            engine = FasterQwen3TTSEngine(
                model_name=resolve_hf_model(
                    size=cfg.model_size,
                    mode=cfg.mode,
                    configured=cfg.hf_model,
                ),
                mode=cfg.mode,
                language=cfg.language,
                device=cfg.device,
                speaker=cfg.speaker,
                temperature=cfg.temperature,
                dtype=cfg.dtype,
            )
        else:
            bin_path = resolve_qwen_tts_bin(cfg.qwen_bin, DS_PROJECT_ROOT)
            model_dir = resolve_model_dir(cfg.model_dir, DS_PROJECT_ROOT)
            talker, codec = ensure_qwen3_models(
                DS_PROJECT_ROOT,
                model_dir,
                size=cfg.model_size,
                mode=cfg.mode,
                quant=cfg.quant,
                repo=cfg.model_repo,
            )
            engine = Qwen3TTSEngine(
                bin_path=bin_path,
                talker_model=talker,
                codec_model=codec,
                mode=cfg.mode,
                language=cfg.language,
                device=cfg.device,
                speaker=cfg.speaker,
                seed=cfg.tts_seed,
                temperature=cfg.temperature,
            )
        engine.set_reference(active_wav, ref_text)
        if cfg.temperature is not None and cfg.backend == "qwentts":
            engine.set_temperature(cfg.temperature)
    else:
        if not args.ref_wav:
            raise SystemExit("Provide --ref-wav or --config")
        active_wav = args.ref_wav.resolve()
        require_file(active_wav, label="reference WAV")
        ref_text = args.ref_text
        if not ref_text and args.ref_text_file:
            ref_text = read_text_file(
                args.ref_text_file.resolve(),
                label="reference transcript",
            )
        if not ref_text:
            raise SystemExit("Provide --ref-text or --ref-text-file")

        bin_path = resolve_qwen_tts_bin(None, PROJECT_ROOT)
        model_dir = resolve_model_dir(None, PROJECT_ROOT)
        talker, codec = ensure_qwen3_models(
            PROJECT_ROOT,
            model_dir,
            size="0.6b",
            mode="base",
            quant="Q8_0",
            repo="Serveurperso/Qwen3-TTS-GGUF",
        )
        engine = Qwen3TTSEngine(
            bin_path=bin_path,
            talker_model=talker,
            codec_model=codec,
            mode="base",
            language="German",
            device="CUDA0",
        )
        engine.set_reference(active_wav, ref_text)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reference : {active_wav}")
    print(f"Ref text  : {ref_text}")
    print(f"Input text: {args.text}")
    print(f"Output    : {output}")

    logging.info("Synthesizing...")
    audio = engine.synthesize(args.text)
    sf.write(str(output), audio, QWEN3_SAMPLE_RATE, subtype="PCM_16")

    duration_s = len(audio) / QWEN3_SAMPLE_RATE
    print(f"Done — wrote {duration_s:.1f}s to {output}")
    print("Play with: ffplay -nodisp -autoexit", output)


if __name__ == "__main__":
    main()
