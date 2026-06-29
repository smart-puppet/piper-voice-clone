"""CLI module for setuptools entry point."""

from __future__ import annotations

from voice_cloning.cli_args import build_arg_parser, run_from_args


def main() -> None:
    run_from_args(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
