#!/usr/bin/env python3
"""Download a dataset zip and extract it into the project datasets/ layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voice_cloning.dataset_generator import dataset_dir_name, load_config  # noqa: E402
from voice_cloning.dataset_import import (  # noqa: E402
    default_output_root,
    import_dataset_archive,
    import_dataset_from_hf,
    import_dataset_from_url,
    inspect_zip_archive,
)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Download a Piper dataset zip (Hugging Face, URL, or local file) "
            "and extract or merge it into datasets/{voice}-{lang}_{version}/. "
            "Full archives include metadata; WAV-only archives merge missing files "
            "into an existing dataset."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--zip",
        type=Path,
        metavar="PATH",
        help="Local .zip file to extract",
    )
    source.add_argument(
        "--hf-repo",
        metavar="REPO_ID",
        help="Hugging Face repo id (e.g. user/my-datasets). Works with private repos via token.",
    )
    source.add_argument(
        "--url",
        metavar="URL",
        help="Direct download URL (GitHub release asset, etc.)",
    )

    parser.add_argument(
        "--file",
        metavar="NAME",
        help="Archive filename inside --hf-repo (e.g. archives/myvoice-de_v01.zip)",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=("dataset", "model"),
        help="Hugging Face repo type (default: dataset)",
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face branch, tag, or commit",
    )
    parser.add_argument(
        "--token",
        help="Access token (default: HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or huggingface-cli login)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Parent directory for datasets (default: ./datasets locally, "
            "/content/piper-work/datasets on Colab)"
        ),
    )
    parser.add_argument(
        "--name",
        help="Target dataset folder name (e.g. myvoice-de_v01). Inferred from zip name when possible.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config used to infer --output-root and/or --name when omitted",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="List zip contents and detected layout, then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Full dataset zip: replace an existing dataset directory. "
            "WAV-only zip: overwrite existing WAV files when merging."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help=argparse.SUPPRESS,
    )
    return parser


def resolve_targets(args: argparse.Namespace) -> tuple[Path, str | None]:
    output_root = args.output_root
    dataset_name = args.name

    if args.config is not None:
        config = load_config(args.config.resolve())
        if output_root is None:
            output_root = config.output_root
        if dataset_name is None:
            dataset_name = dataset_dir_name(
                config.voice_name,
                config.language_code,
                config.dataset_version,
            )

    if output_root is None:
        output_root = default_output_root(args.project_root)

    return output_root.resolve(), dataset_name


def main() -> None:
    args = build_parser().parse_args()
    output_root, dataset_name = resolve_targets(args)

    if args.hf_repo:
        if not args.file:
            raise SystemExit("--file is required with --hf-repo")
        import_dataset_from_hf(
            repo_id=args.hf_repo,
            filename=args.file,
            output_root=output_root,
            dataset_name=dataset_name,
            repo_type=args.repo_type,
            revision=args.revision,
            token=args.token,
            force=args.force,
        )
        return

    if args.url:
        import_dataset_from_url(
            url=args.url,
            output_root=output_root,
            dataset_name=dataset_name,
            token=args.token,
            force=args.force,
        )
        return

    zip_path = args.zip.resolve()
    if not zip_path.is_file():
        raise SystemExit(f"Zip file not found: {zip_path}")

    if args.inspect:
        info = inspect_zip_archive(zip_path)
        print(f"Archive: {info['path']}")
        print(f"Layout:  {info['layout']}")
        print(
            f"Files:   {info['meaningful_members']} meaningful "
            f"({info['wav_files']} wav, {info['cache_files']} cache, "
            f"{info['metadata_files']} metadata) "
            f"of {info['raw_members']} total"
        )
        for path in info["sample_paths"]:
            print(f"  {path}")
        return

    import_dataset_archive(
        zip_path,
        output_root=output_root,
        dataset_name=dataset_name,
        force=args.force,
    )


if __name__ == "__main__":
    main()
