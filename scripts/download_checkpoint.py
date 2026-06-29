#!/usr/bin/env python3
"""Download a Piper pretrained checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voice_cloning.checkpoint_download import (  # noqa: E402
    DEFAULT_REPO_ID,
    PRESETS,
    default_checkpoint_path,
    download_piper_checkpoint,
    list_repo_checkpoints,
)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Download a Piper checkpoint from "
            f"huggingface.co/datasets/{DEFAULT_REPO_ID} "
            "for fine-tuning with --model.pretrained_ckpt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --preset de-thorsten-medium\n"
            "  %(prog)s --list --lang de\n"
            "  %(prog)s --path de/de_DE/thorsten/medium/epoch=3135-step=2702056.ckpt\n"
            "  %(prog)s --preset en-lessac-medium --output checkpoints/en-lessac.ckpt\n"
        ),
    )
    parser.add_argument(
        "--preset",
        metavar="NAME",
        help="Shortcut voice preset (e.g. de-thorsten-medium, en-lessac-medium)",
    )
    parser.add_argument(
        "--path",
        metavar="HF_PATH",
        help="Exact file path inside the Hugging Face dataset repo",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available .ckpt files in the repo and exit",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show built-in presets and exit",
    )
    parser.add_argument(
        "--lang",
        metavar="CODE",
        help="Filter --list output by language prefix (e.g. de, en)",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face branch, tag, or commit",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token (default: HF_TOKEN / huggingface-cli login)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Destination .ckpt path "
            "(default: checkpoints/pretrained-finetune.ckpt, "
            "or /content/piper-work/checkpoints/ on Colab)"
        ),
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Keep full Lightning checkpoint (default: save weights-only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_presets:
        for key in sorted(PRESETS):
            preset = PRESETS[key]
            print(f"{key:22}  {preset.prefix}  ({preset.description})")
        return

    if args.list:
        paths = list_repo_checkpoints(
            repo_id=args.repo,
            lang=args.lang,
            token=args.token,
        )
        if not paths:
            scope = f" under {args.lang}/" if args.lang else ""
            print(f"No checkpoints found{scope}in {args.repo}")
            return
        for path in paths:
            print(path)
        print(f"\n{len(paths)} checkpoint(s)")
        return

    if not args.preset and not args.path:
        raise SystemExit("Either --preset or --path is required (or use --list)")

    output = args.output or default_checkpoint_path(args.project_root)
    download_piper_checkpoint(
        dest=output,
        repo_id=args.repo,
        preset=args.preset,
        hf_path=args.path,
        revision=args.revision,
        token=args.token,
        strip_optimizer=not args.keep_optimizer,
        force=args.force,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
