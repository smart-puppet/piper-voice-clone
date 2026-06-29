"""Validate paths referenced from YAML config."""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def require_file(path: Path, *, label: str) -> None:
    if path.is_file():
        return
    raise FileNotFoundError(
        f"Missing {label}:\n  {path}\n"
        "Add the file or update the path in config.yaml."
    )


def read_text_file(path: Path, *, label: str) -> str:
    require_file(path, label=label)
    return path.read_text(encoding="utf-8").strip()


def format_missing_files(missing: list[tuple[str, Path]]) -> str:
    lines = "\n".join(f"  - {label}: {path}" for label, path in missing)
    return (
        "Missing files referenced in config:\n"
        f"{lines}\n"
        "Add the files under references/ or update config.yaml."
    )
