"""Download and import Piper dataset archives into the project layout."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

LayoutType = Literal["nested", "flat", "wavs_only", "cache_only", "partial"]

import requests

from voice_cloning.hf_utils import resolve_hf_token
from voice_cloning.piper_training import resolve_metadata_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+-[a-z]{2}_v[0-9]+$", re.IGNORECASE)
DATASET_NAME_PREFIX_RE = re.compile(
    r"^([a-zA-Z0-9_-]+-[a-z]{2}_v[0-9]+)",
    re.IGNORECASE,
)
UTTERANCE_ID_RE = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)
_IGNORED_ZIP_PREFIXES = ("__MACOSX/", ".git/")
_IGNORED_ZIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
_IGNORED_EXTRA_NAMES = {
    "config.json",
    ".generation_progress.json",
    ".generation_manifest.json",
}


def default_output_root(project_root: Path | None = None) -> Path:
    """Pick datasets/ under the project, or Colab work dir when present."""
    if os.environ.get("PIPER_COLAB") == "1":
        return Path("/content/piper-work/datasets")

    colab_datasets = Path("/content/piper-work/datasets")
    if colab_datasets.parent.is_dir():
        return colab_datasets

    root = project_root or PROJECT_ROOT
    return (root / "datasets").resolve()


def infer_dataset_name(
    zip_path: Path,
    explicit: str | None = None,
    members: list[str] | None = None,
) -> str:
    if explicit:
        return explicit

    stem = zip_path.stem
    if stem.endswith("-wavs"):
        stem = stem[: -len("-wavs")]

    match = DATASET_NAME_PREFIX_RE.match(stem)
    if match:
        return match.group(1)

    if DATASET_NAME_RE.fullmatch(stem):
        return stem

    if members:
        inferred = _infer_dataset_name_from_members(members)
        if inferred:
            return inferred

    raise ValueError(
        f"Could not infer dataset directory name from {zip_path.name}. "
        "Pass --name myvoice-de_v03"
    )


def _infer_dataset_name_from_members(members: list[str]) -> str | None:
    for member in members:
        for part in Path(member).parts:
            if DATASET_NAME_RE.fullmatch(part):
                return part
    return None


def _normalize_zip_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_wav_member(path: str) -> bool:
    path = _normalize_zip_path(path)
    name = Path(path).name
    if Path(path).suffix.lower() == ".wav":
        return True
    # Piper utterance files are md5 hashes; some archives omit the .wav suffix.
    if UTTERANCE_ID_RE.fullmatch(name):
        return True
    if UTTERANCE_ID_RE.fullmatch(Path(path).stem) and Path(path).suffix.lower() == ".wav":
        return True
    parts = [part.lower() for part in Path(path).parts]
    if "wavs" in parts and UTTERANCE_ID_RE.fullmatch(name):
        return True
    return False


def _wav_dest_name(member: str) -> str | None:
    path = _normalize_zip_path(member)
    name = Path(path).name
    if name.lower().endswith(".wav"):
        return name
    if UTTERANCE_ID_RE.fullmatch(name):
        return f"{name}.wav"
    stem = Path(path).stem
    if UTTERANCE_ID_RE.fullmatch(stem):
        return f"{stem}.wav"
    return None


def _is_cache_member(path: str) -> bool:
    parts = [part.lower() for part in Path(_normalize_zip_path(path)).parts]
    return "cache" in parts


def _cache_rel_path(member: str) -> Path:
    """Path under cache/ inside the dataset (e.g. foo.spec.pt)."""
    path = _normalize_zip_path(member)
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part.lower() == "cache":
            return Path(*parts[index + 1 :])
    return Path(path).name


def _is_metadata_member(path: str) -> bool:
    path = _normalize_zip_path(path)
    return Path(path).name.startswith("metadata") and path.endswith(".csv")


def _is_extra_member(path: str) -> bool:
    name = Path(_normalize_zip_path(path)).name
    return name in _IGNORED_EXTRA_NAMES or name.startswith(".generation_")


def _should_skip_zip_member(path: str) -> bool:
    path = _normalize_zip_path(path)
    if path.endswith("/"):
        return True
    if any(path.startswith(prefix) for prefix in _IGNORED_ZIP_PREFIXES):
        return True
    name = Path(path).name
    if name in _IGNORED_ZIP_NAMES:
        return True
    if _is_extra_member(path) and not (
        _is_wav_member(path) or _is_metadata_member(path) or _is_cache_member(path)
    ):
        return True
    return False


def _meaningful_zip_members(members: list[str]) -> list[str]:
    return [name for name in members if not _should_skip_zip_member(name)]


def _zip_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return [
            _normalize_zip_path(name)
            for name in archive.namelist()
            if not name.endswith("/")
        ]


def _unwrap_nested_zip(zip_path: Path) -> Path:
    """If the archive contains a single inner .zip, use that instead."""
    members = _meaningful_zip_members(_zip_members(zip_path))
    if len(members) != 1 or not members[0].lower().endswith(".zip"):
        return zip_path

    inner_name = members[0]
    with zipfile.ZipFile(zip_path) as outer:
        inner_bytes = outer.read(inner_name)
    inner_path = zip_path.with_name(Path(inner_name).name)
    inner_path.write_bytes(inner_bytes)
    print(f"Unpacking nested archive: {inner_name}")
    return inner_path


def _has_metadata(paths: list[str]) -> bool:
    return any(_is_metadata_member(path) for path in paths)


def _analyze_zip_members(
    members: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    meaningful = _meaningful_zip_members(members)
    wav_members = [path for path in meaningful if _is_wav_member(path)]
    meta_members = [path for path in meaningful if _is_metadata_member(path)]
    cache_members = [path for path in meaningful if _is_cache_member(path)]
    return meaningful, wav_members, meta_members, cache_members


def detect_zip_layout(zip_path: Path) -> LayoutType:
    members = _zip_members(zip_path)
    meaningful, wav_members, meta_members, cache_members = _analyze_zip_members(members)

    if not meaningful:
        raise ValueError(f"Zip archive is empty: {zip_path}")

    if meta_members:
        top_levels = {path.split("/")[0] for path in meaningful}
        if len(top_levels) == 1:
            top = next(iter(top_levels))
            under_top = [path for path in meaningful if path.startswith(f"{top}/")]
            if under_top and (
                _has_metadata(under_top)
                or any("/wavs/" in path for path in under_top)
                or any("/cache/" in path.lower() for path in under_top)
            ):
                return "nested"
        return "flat"

    if wav_members and cache_members:
        return "partial"
    if wav_members:
        return "wavs_only"
    if cache_members:
        return "cache_only"

    sample = ", ".join(meaningful[:8])
    if len(meaningful) > 8:
        sample += ", ..."
    raise ValueError(
        f"Unrecognized zip layout in {zip_path.name}. "
        f"Found {len(meaningful)} file(s) but no metadata, WAV audio, or Piper cache. "
        f"Examples: {sample}"
    )


def _dataset_has_metadata(dataset_dir: Path) -> bool:
    try:
        resolve_metadata_csv(dataset_dir)
    except FileNotFoundError:
        return False
    return True


def validate_partial_target(dataset_dir: Path) -> None:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    csv_path = resolve_metadata_csv(dataset_dir)
    print(f"Partial import into {dataset_dir} ({csv_path.name})")


def validate_dataset_dir(dataset_dir: Path) -> None:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    wav_dir = dataset_dir / "wavs"
    if not wav_dir.is_dir() or not any(wav_dir.glob("*.wav")):
        raise FileNotFoundError(f"No WAV files found under {wav_dir}")

    csv_path = resolve_metadata_csv(dataset_dir)
    print(f"Validated dataset: {dataset_dir} ({csv_path.name})")


def _resolve_dataset_name(
    zip_path: Path,
    members: list[str],
    layout: str,
    dataset_name: str | None,
) -> str:
    inferred = infer_dataset_name(zip_path, dataset_name, members=members)
    if dataset_name:
        return inferred

    top_levels = {
        path.split("/")[0]
        for path in _meaningful_zip_members(members)
        if DATASET_NAME_RE.fullmatch(path.split("/")[0])
    }
    if layout == "nested" and len(top_levels) == 1:
        return next(iter(top_levels))
    if layout in {"cache_only", "partial", "wavs_only"} and len(top_levels) == 1:
        top = next(iter(top_levels))
        if DATASET_NAME_RE.fullmatch(top):
            return top

    return inferred


def _safe_extract_members(
    archive: zipfile.ZipFile,
    members: list[str],
    dest: Path,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for member in members:
        member = _normalize_zip_path(member)
        if _should_skip_zip_member(member):
            continue
        archive.extract(member, dest)


def _merge_wav_archive(
    archive: zipfile.ZipFile,
    wav_dir: Path,
    *,
    overwrite: bool,
    wav_members: list[str] | None = None,
) -> tuple[int, int]:
    """Copy WAV members into wav_dir. Returns (written, skipped)."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    members = wav_members or [
        name
        for name in archive.namelist()
        if not _should_skip_zip_member(name) and _is_wav_member(name)
    ]

    for member in members:
        member = _normalize_zip_path(member)
        dest_name = _wav_dest_name(member)
        if dest_name is None:
            continue

        dest = wav_dir / dest_name
        if dest.exists() and not overwrite:
            skipped += 1
            continue

        with archive.open(member) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        written += 1

    return written, skipped


def _resolve_archive_member(archive: zipfile.ZipFile, member: str) -> str | None:
    normalized = _normalize_zip_path(member)
    for name in archive.namelist():
        if _normalize_zip_path(name) == normalized:
            return name
    return None


def _merge_cache_archive(
    archive: zipfile.ZipFile,
    cache_dir: Path,
    *,
    overwrite: bool,
    cache_members: list[str] | None = None,
) -> tuple[int, int]:
    """Copy Piper cache members into cache_dir. Returns (written, skipped)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    members = cache_members or [
        _normalize_zip_path(name)
        for name in archive.namelist()
        if not _should_skip_zip_member(name) and _is_cache_member(name)
    ]

    for member in members:
        member = _normalize_zip_path(member)
        rel = _cache_rel_path(member)
        if not rel.parts:
            continue

        dest = cache_dir / rel
        if dest.exists() and not overwrite:
            skipped += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        arc_name = _resolve_archive_member(archive, member)
        if arc_name is None:
            continue

        with archive.open(arc_name) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        written += 1

    return written, skipped


def inspect_zip_archive(zip_path: Path) -> dict[str, object]:
    zip_path = _unwrap_nested_zip(zip_path.resolve())
    members = _zip_members(zip_path)
    meaningful, wav_members, meta_members, cache_members = _analyze_zip_members(members)
    layout = "unknown"
    try:
        layout = detect_zip_layout(zip_path)
    except ValueError:
        pass
    return {
        "path": zip_path,
        "raw_members": len(members),
        "meaningful_members": len(meaningful),
        "wav_files": len(wav_members),
        "metadata_files": len(meta_members),
        "cache_files": len(cache_members),
        "layout": layout,
        "sample_paths": meaningful[:15],
    }


def extract_dataset_zip(
    zip_path: Path,
    *,
    output_root: Path,
    dataset_name: str | None = None,
    force: bool = False,
) -> Path:
    zip_path = _unwrap_nested_zip(zip_path.resolve())
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    members = _zip_members(zip_path)
    meaningful, wav_members, meta_members, cache_members = _analyze_zip_members(members)
    layout = detect_zip_layout(zip_path)
    name = _resolve_dataset_name(zip_path, members, layout, dataset_name)
    target = output_root / name
    zip_has_metadata = bool(meta_members)
    zip_has_partial = bool(wav_members or cache_members)

    if not zip_has_metadata and zip_has_partial:
        if not _dataset_has_metadata(target):
            raise FileNotFoundError(
                f"Partial archive requires an existing dataset with metadata at:\n"
                f"  {target}\n"
                "Import a full dataset zip first, or generate the dataset locally, "
                "then use this archive to add WAV and/or Piper cache files.\n"
                f"Inferred dataset name: {name}. Pass --name if that is wrong."
            )

        total_written = 0
        with zipfile.ZipFile(zip_path) as archive:
            if wav_members:
                written, skipped = _merge_wav_archive(
                    archive,
                    target / "wavs",
                    overwrite=force,
                    wav_members=wav_members,
                )
                total_written += written
                print(
                    f"Merged {written} WAV file(s) into {target / 'wavs'}"
                    + (f" ({skipped} already present)" if skipped else "")
                )

            if cache_members:
                written, skipped = _merge_cache_archive(
                    archive,
                    target / "cache",
                    overwrite=force,
                    cache_members=cache_members,
                )
                total_written += written
                print(
                    f"Merged {written} cache file(s) into {target / 'cache'}"
                    + (f" ({skipped} already present)" if skipped else "")
                )

        if total_written == 0:
            raise ValueError(
                f"No new files imported from {zip_path.name} "
                "(all files already present; use --force to overwrite)"
            )

        validate_partial_target(target)
        return target

    if not zip_has_metadata:
        raise ValueError(f"No importable content found in {zip_path.name}")

    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(
            f"Dataset already exists: {target}\n"
            "Use --force to replace it."
        )

    if target.exists() and force:
        shutil.rmtree(target)

    with zipfile.ZipFile(zip_path) as archive:
        if layout == "nested":
            _safe_extract_members(archive, meaningful, output_root)
            if not target.is_dir():
                raise ValueError(
                    f"Expected nested dataset folder {target.name} inside {zip_path.name}"
                )
            validate_dataset_dir(target)
            return target

        target.mkdir(parents=True, exist_ok=True)
        _safe_extract_members(archive, meaningful, target)
        validate_dataset_dir(target)
        return target


def download_http_file(
    url: str,
    dest: Path,
    *,
    token: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    headers = {"User-Agent": "piper-voice-clone/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return dest


def download_hf_file(
    *,
    repo_id: str,
    filename: str,
    dest: Path | None = None,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    resolved_token = resolve_hf_token(token)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
        token=resolved_token,
        local_dir=None,
    )
    src = Path(downloaded)
    if dest is None:
        return src

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def import_dataset_archive(
    zip_path: Path,
    *,
    output_root: Path,
    dataset_name: str | None = None,
    force: bool = False,
) -> Path:
    try:
        dataset_dir = extract_dataset_zip(
            zip_path,
            output_root=output_root,
            dataset_name=dataset_name,
            force=force,
        )
    except (ValueError, zipfile.BadZipFile) as exc:
        try:
            info = inspect_zip_archive(zip_path)
            sample = ", ".join(str(p) for p in info["sample_paths"][:6])
            hint = (
                f"\nArchive inspection: {info['wav_files']} wav, "
                f"{info['cache_files']} cache, {info['metadata_files']} metadata "
                f"({info['meaningful_members']} meaningful / {info['raw_members']} raw). "
                f"Sample paths: {sample or '(none)'}"
            )
            if info["wav_files"] or info["cache_files"]:
                if not info["metadata_files"]:
                    hint += (
                        "\nThis looks like a partial archive (WAV and/or Piper cache). "
                        "Ensure metadata already exists under the target dataset folder."
                    )
        except Exception:
            hint = ""
        raise type(exc)(f"{exc}{hint}") from exc

    print(f"Dataset ready: {dataset_dir}")
    return dataset_dir


def import_dataset_from_hf(
    *,
    repo_id: str,
    filename: str,
    output_root: Path,
    dataset_name: str | None = None,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    force: bool = False,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="piper-dataset-") as tmp:
        zip_path = Path(tmp) / Path(filename).name
        print(f"Downloading {filename} from huggingface.co/{repo_id} ...")
        download_hf_file(
            repo_id=repo_id,
            filename=filename,
            dest=zip_path,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
        return import_dataset_archive(
            zip_path,
            output_root=output_root,
            dataset_name=dataset_name,
            force=force,
        )


def import_dataset_from_url(
    *,
    url: str,
    output_root: Path,
    dataset_name: str | None = None,
    token: str | None = None,
    force: bool = False,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="piper-dataset-") as tmp:
        zip_name = Path(urlparse(url).path).name or "dataset.zip"
        zip_path = Path(tmp) / zip_name
        print(f"Downloading {url} ...")
        download_http_file(url, zip_path, token=token)
        return import_dataset_archive(
            zip_path,
            output_root=output_root,
            dataset_name=dataset_name,
            force=force,
        )
