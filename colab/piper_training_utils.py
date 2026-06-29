"""Helpers for Piper Colab training — local disk during runs, Drive publish at the end."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

import torch

T = TypeVar("T")

# Piper Lightning checkpoints pickle pathlib objects; required on PyTorch 2.6+.
torch.serialization.add_safe_globals([pathlib.PosixPath])


def _progress(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str = "file",
    total: int | None = None,
) -> Iterator[T]:
    """tqdm in notebooks; falls back gracefully if tqdm is missing."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iter(iterable)

    return tqdm(iterable, desc=desc, unit=unit, total=total, dynamic_ncols=True)


def copy_file_with_progress(src: Path, dst: Path, *, desc: str) -> None:
    """Copy a file with a byte-level tqdm bar (useful for large checkpoints)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    try:
        from tqdm.auto import tqdm
    except ImportError:
        shutil.copy2(src, dst)
        return

    with (
        open(src, "rb") as fsrc,
        open(dst, "wb") as fdst,
        tqdm(total=total, desc=desc, unit="B", unit_scale=True, dynamic_ncols=True) as bar,
    ):
        while chunk := fsrc.read(1024 * 1024):
            fdst.write(chunk)
            bar.update(len(chunk))
    shutil.copystat(src, dst)


def _version_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("version_"):
            try:
                return int(part.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
    return -1


def colab_training_profile(
    *,
    batch_size: int = 0,
    num_workers: int = -1,
    precision: str = "auto",
    accumulate_grad_batches: int = 1,
    log_every_n_steps: int = 50,
    val_check_every_n_epochs: int = 1,
    limit_val_batches: int = 15,
) -> dict[str, Any]:
    """Pick Colab-friendly training settings from the active GPU/CPU."""
    cpu_count = os.cpu_count() or 2
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda else 0.0
    )

    if num_workers < 0:
        resolved_workers = 0 if not cuda else min(4, max(1, cpu_count - 1))
    else:
        resolved_workers = num_workers

    if batch_size <= 0:
        if vram_gb >= 38:
            resolved_batch = 24
        elif vram_gb >= 14:
            resolved_batch = 16
        elif vram_gb >= 8:
            resolved_batch = 8
        else:
            resolved_batch = 4
    else:
        resolved_batch = batch_size

    if precision == "auto":
        # Piper mel/STFT uses torch.stft; cuFFT does not support bfloat16 tensors.
        resolved_precision = "16-mixed" if cuda else "32-true"
    else:
        resolved_precision = precision

    return {
        "batch_size": resolved_batch,
        "num_workers": resolved_workers,
        "precision": resolved_precision,
        "pin_memory": cuda,
        "benchmark": cuda,
        "accumulate_grad_batches": max(1, accumulate_grad_batches),
        "log_every_n_steps": max(1, log_every_n_steps),
        "val_check_every_n_epochs": max(1, val_check_every_n_epochs),
        "limit_val_batches": max(0, limit_val_batches),
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "cpu_count": cpu_count,
    }


def training_cli_args(profile: dict[str, Any]) -> list[str]:
    """Convert a training profile into Piper/Lightning CLI flags."""
    args = [
        "--data.batch_size",
        str(profile["batch_size"]),
        "--data.num_workers",
        str(profile["num_workers"]),
        "--trainer.precision",
        profile["precision"],
        "--trainer.accumulate_grad_batches",
        str(profile["accumulate_grad_batches"]),
        "--trainer.log_every_n_steps",
        str(profile["log_every_n_steps"]),
        "--trainer.num_sanity_val_steps",
        "0",
        "--trainer.check_val_every_n_epoch",
        str(profile["val_check_every_n_epochs"]),
    ]
    if profile.get("limit_val_batches", 0) > 0:
        args.extend(
            ["--trainer.limit_val_batches", str(profile["limit_val_batches"])]
        )
    if profile.get("pin_memory"):
        args.extend(["--data.pin_memory", "true"])
    if profile.get("benchmark"):
        args.extend(["--trainer.benchmark", "true"])
    return args


def sync_local_dataset(
    *,
    dataset_dir: Path,
    local_dataset_dir: Path,
) -> dict[str, Path]:
    """Copy metadata + wavs to local disk so Piper startup avoids Drive I/O."""
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_wavs = local_dataset_dir / "wavs"
    local_wavs.mkdir(parents=True, exist_ok=True)

    drive_wavs = dataset_dir / "wavs"
    for name in ("metadata.csv", "metadata_train.csv", "config.json"):
        src = dataset_dir / name
        dst = local_dataset_dir / name
        if src.exists() and (
            (not dst.exists()) or (dst.stat().st_size != src.stat().st_size)
        ):
            copy_file_with_progress(src, dst, desc=f"Copy {name}")

    wav_files = sorted(drive_wavs.glob("*.wav"))
    local_wav_count = len(list(local_wavs.glob("*.wav")))
    if local_wav_count < len(wav_files):
        copied = 0
        for src in _progress(wav_files, desc="WAVs → local", unit="file"):
            dst = local_wavs / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                copied += 1
        print(f"Wav sync complete: {copied} copied, {len(wav_files)} total")
    else:
        print(f"Local wavs ready: {local_wav_count}")

    csv_path = local_dataset_dir / "metadata.csv"
    if not csv_path.exists():
        csv_path = local_dataset_dir / "metadata_train.csv"

    return {
        "csv_path": csv_path,
        "audio_dir": local_wavs,
        "config_path": local_dataset_dir / "config.json",
    }


def sync_local_cache_from_drive(
    *,
    drive_cache_dir: Path,
    local_cache_dir: Path,
) -> tuple[int, int]:
    """Copy missing cache files from Drive to local disk."""
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    drive_files = [p for p in drive_cache_dir.iterdir() if p.is_file()]
    copied = 0
    for src in _progress(drive_files, desc="Cache → local", unit="file"):
        dst = local_cache_dir / src.name
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
            copied += 1
    return copied, len(drive_files)


def sync_local_cache_to_drive(
    *,
    local_cache_dir: Path,
    drive_cache_dir: Path,
) -> int:
    """Copy local cache files back to Drive."""
    drive_cache_dir.mkdir(parents=True, exist_ok=True)
    local_files = [p for p in local_cache_dir.iterdir() if p.is_file()]
    synced = 0
    for src in _progress(local_files, desc="Cache → Drive", unit="file"):
        dst = drive_cache_dir / src.name
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
            synced += 1
    return synced


def run_subprocess_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command with live notebook output (supports tqdm in subprocess)."""
    merged_env = {
        **os.environ,
        **(env or {}),
        "PYTHONUNBUFFERED": "1",
        "PIP_PROGRESS_BAR": "on",
    }
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        stdout=None,
        stderr=subprocess.STDOUT,
    )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _newest_checkpoint_in_root(lightning_root: Path) -> Path | None:
    if not lightning_root.is_dir():
        return None

    last_ckpts = list(lightning_root.rglob("last.ckpt"))
    if last_ckpts:
        return max(
            last_ckpts,
            key=lambda path: (_version_from_path(path), path.stat().st_mtime),
        )

    epoch_ckpts = [
        path
        for path in lightning_root.rglob("*.ckpt")
        if path.name != "last.ckpt"
    ]
    if not epoch_ckpts:
        return None

    return max(
        epoch_ckpts,
        key=lambda path: (_version_from_path(path), path.stat().st_mtime),
    )


def restore_checkpoint_from_drive(
    *,
    lightning_dir: Path,
    drive_checkpoints: Path,
) -> Path | None:
    """Copy latest.ckpt from Drive when local Colab disk has no checkpoint."""
    if find_resume_checkpoint(lightning_dir) is not None:
        return find_resume_checkpoint(lightning_dir)

    if not _drive_is_mounted():
        return None

    latest = drive_checkpoints / "latest.ckpt"
    if not latest.is_file():
        return None

    dest_dir = lightning_dir / "lightning_logs" / "version_0" / "checkpoints"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "last.ckpt"
    copy_file_with_progress(latest, dest, desc="Drive latest.ckpt → local")
    print(f"Restored training checkpoint from Drive: {dest}")
    return dest


def find_resume_checkpoint(*lightning_roots: Path) -> Path | None:
    """Return the newest checkpoint across one or more Lightning log roots."""
    candidates = [
        ckpt
        for root in lightning_roots
        if (ckpt := _newest_checkpoint_in_root(root)) is not None
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (_version_from_path(path), path.stat().st_mtime),
    )


def _drive_is_mounted() -> bool:
    """Return True when Colab Drive is available at /content/drive/MyDrive."""
    return Path("/content/drive/MyDrive").is_dir()


def _is_colab_runtime() -> bool:
    return os.environ.get("PIPER_COLAB") == "1" or Path("/content").is_dir()


def disconnect_colab_runtime() -> bool:
    """Release the Colab GPU runtime after training (no-op on local machines)."""
    if not _is_colab_runtime():
        return False

    if not _can_mount_drive_from_ipython():
        print(
            "Training finished. GPU disconnect skipped (not running in the notebook kernel).\n"
            "Runtime → Disconnect and delete runtime to release the GPU,\n"
            "or enable KEEP_RUNTIME_CONNECTED in the Train cell."
        )
        return False

    try:
        from google.colab import runtime
    except ImportError:
        print(
            "Colab runtime disconnect skipped (google.colab not available).\n"
            "To release the GPU manually: Runtime → Disconnect and delete runtime."
        )
        return False

    print("Training finished — releasing Colab runtime (runtime.unassign())...")
    try:
        runtime.unassign()
        return True
    except Exception as exc:
        print(
            "Could not auto-disconnect Colab runtime "
            f"({type(exc).__name__}: {exc}).\n"
            "Use Runtime → Disconnect and delete runtime to release the GPU."
        )
        return False


def _can_mount_drive_from_ipython() -> bool:
    """drive.mount() only works inside a Colab notebook kernel, not in Terminal."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def drive_mount_instructions() -> str:
    return (
        "Google Drive is not mounted. Colab Terminal cannot run drive.mount() — "
        "use a notebook cell first:\n\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n\n"
        "Then re-run this script, or pass --skip-publish to keep results on local disk only."
    )


def ensure_drive_mounted(drive_root: Path) -> Path:
    """Use an existing Drive mount, or mount from a notebook kernel when possible."""
    drive_root = drive_root.resolve()

    if _drive_is_mounted():
        drive_root.mkdir(parents=True, exist_ok=True)
        return drive_root

    if drive_root.is_dir():
        return drive_root

    try:
        from google.colab import drive
    except ImportError as exc:
        raise FileNotFoundError(
            f"Google Drive is not mounted and this is not a Colab runtime: {drive_root}"
        ) from exc

    if not _can_mount_drive_from_ipython():
        raise RuntimeError(drive_mount_instructions())

    print("Mounting Google Drive to publish results...")
    drive.mount("/content/drive", force_remount=False)
    if not _drive_is_mounted():
        raise RuntimeError(
            "Drive mount did not complete. " + drive_mount_instructions()
        )

    drive_root.mkdir(parents=True, exist_ok=True)
    return drive_root


def sync_directory_to_drive(
    *,
    local_dir: Path,
    drive_dir: Path,
    desc: str = "Files → Drive",
) -> int:
    """Recursively copy local_dir to drive_dir."""
    local_dir = local_dir.resolve()
    drive_dir = drive_dir.resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    copied = 0
    for src in _progress(local_dir.rglob("*"), desc=desc, unit="file"):
        if not src.is_file():
            continue
        rel = src.relative_to(local_dir)
        dst = drive_dir / rel
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def zip_dataset_directory(dataset_dir: Path, zip_path: Path) -> int:
    """Create a zip archive with metadata, config, and wavs/ for distribution."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    wav_dir = dataset_dir / "wavs"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {wav_dir}")

    meta_files = sorted(dataset_dir.glob("metadata*.csv"))
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata CSV files found in {dataset_dir} "
            "(expected metadata.csv or metadata_train.csv)"
        )

    extra_files = [p for p in (dataset_dir / "config.json",) if p.is_file()]
    prefix = dataset_dir.name
    to_zip: list[tuple[Path, str]] = []

    for meta in meta_files:
        to_zip.append((meta, f"{prefix}/{meta.name}"))
    for extra in extra_files:
        to_zip.append((extra, f"{prefix}/{extra.name}"))
    for wav in wav_files:
        to_zip.append((wav, f"{prefix}/wavs/{wav.name}"))

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src, arcname in _progress(to_zip, desc="Zipping dataset", unit="file"):
            archive.write(src, arcname=arcname)
    return len(to_zip)


def archive_dataset_to_drive(
    *,
    dataset_dir: Path,
    drive_root: Path,
) -> Path:
    """Zip the full dataset locally, then copy the archive to Drive."""
    dataset_dir = dataset_dir.resolve()
    drive_root = ensure_drive_mounted(drive_root)
    archive_dir = drive_root / "archives"
    zip_name = f"{dataset_dir.name}.zip"
    final_zip = archive_dir / zip_name
    work_zip = Path("/content") / zip_name

    count = zip_dataset_directory(dataset_dir, work_zip)
    copy_file_with_progress(work_zip, final_zip, desc="Dataset archive → Drive")
    work_zip.unlink(missing_ok=True)

    print(f"Archived {count} file(s) (metadata + wavs) to {final_zip}")
    return final_zip


def publish_dataset_to_drive(
    *,
    dataset_dir: Path,
    drive_root: Path,
) -> dict[str, Path]:
    """Zip the dataset and copy only the archive to Google Drive."""
    drive_root = ensure_drive_mounted(drive_root)

    print("Publishing dataset archive to Google Drive...")
    archive = archive_dataset_to_drive(
        dataset_dir=dataset_dir,
        drive_root=drive_root,
    )
    return {"archive": archive}


def publish_checkpoints_to_drive(
    *,
    local_root: Path,
    drive_root: Path,
) -> dict[str, Any]:
    """Copy training checkpoints to Drive (not TensorBoard logs or event files)."""
    drive_root = ensure_drive_mounted(drive_root)
    drive_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in _progress(
        sorted(local_root.rglob("*.ckpt")) if local_root.is_dir() else [],
        desc="Checkpoints → Drive",
        unit="file",
    ):
        rel = src.relative_to(local_root)
        dst = drive_root / rel
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    ckpt = find_resume_checkpoint(local_root)
    result: dict[str, Any] = {
        "synced_files": copied,
        "drive_root": drive_root,
        "latest_checkpoint": ckpt,
        "latest_on_drive": None,
    }
    if ckpt is None:
        print("No checkpoints found to publish to Drive.")
        return result

    latest = drive_root / "latest.ckpt"
    if (not latest.exists()) or (
        latest.stat().st_size != ckpt.stat().st_size
        or latest.stat().st_mtime < ckpt.stat().st_mtime
    ):
        copy_file_with_progress(ckpt, latest, desc="latest.ckpt → Drive")
    result["latest_on_drive"] = latest
    print(f"Latest checkpoint on Drive: {latest}")
    print(f"Source: {ckpt}")
    print(f"Copied {copied} checkpoint file(s) under {drive_root}")
    return result


def publish_training_to_drive(
    *,
    drive_root: Path,
    local_lightning: Path | None = None,
    drive_checkpoints: Path | None = None,
    local_onnx: Path | None = None,
    drive_onnx: Path | None = None,
    local_onnx_phoneme: Path | None = None,
    drive_onnx_phoneme: Path | None = None,
    local_onnx_config: Path | None = None,
    drive_onnx_config: Path | None = None,
) -> dict[str, Any]:
    """Publish checkpoints and ONNX to Drive. TensorBoard logs stay on local Colab disk."""
    ensure_drive_mounted(drive_root)
    result: dict[str, Any] = {}

    if local_lightning is not None and drive_checkpoints is not None:
        result.update(
            publish_checkpoints_to_drive(
                local_root=local_lightning,
                drive_root=drive_checkpoints,
            )
        )

    if local_onnx and local_onnx.is_file() and drive_onnx is not None:
        drive_onnx.parent.mkdir(parents=True, exist_ok=True)
        copy_file_with_progress(local_onnx, drive_onnx, desc="ONNX → Drive")
        result["onnx"] = drive_onnx
        print(f"ONNX on Drive: {drive_onnx}")

    if (
        local_onnx_phoneme
        and local_onnx_phoneme.is_file()
        and drive_onnx_phoneme is not None
    ):
        drive_onnx_phoneme.parent.mkdir(parents=True, exist_ok=True)
        copy_file_with_progress(
            local_onnx_phoneme, drive_onnx_phoneme, desc="ONNX (phoneme duration) → Drive"
        )
        result["onnx_phoneme_duration"] = drive_onnx_phoneme
        print(f"ONNX (phoneme duration) on Drive: {drive_onnx_phoneme}")

    if (
        local_onnx_config
        and local_onnx_config.is_file()
        and drive_onnx_config is not None
    ):
        drive_onnx_config.parent.mkdir(parents=True, exist_ok=True)
        copy_file_with_progress(
            local_onnx_config, drive_onnx_config, desc="ONNX config → Drive"
        )
        result["onnx_config"] = drive_onnx_config

    return result


def sync_lightning_to_drive(
    *,
    local_root: Path,
    drive_root: Path,
) -> int:
    """Copy Lightning checkpoints and TensorBoard events from local disk to Drive."""
    drive_root.mkdir(parents=True, exist_ok=True)
    if not local_root.is_dir():
        return 0

    to_sync = []
    patterns = ("*.ckpt", "events.out.tfevents.*", "hparams.yaml")
    for pattern in patterns:
        to_sync.extend(sorted(local_root.rglob(pattern)))

    copied = 0
    for src in _progress(to_sync, desc="Checkpoints → Drive", unit="file"):
        rel = src.relative_to(local_root)
        dst = drive_root / rel
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def resolve_tensorboard_logdir(*log_roots: Path) -> Path:
    """Return the TensorBoard parent dir Lightning uses (`.../lightning_logs`)."""
    for root in log_roots:
        if not root.is_dir():
            continue

        candidates = (root / "lightning_logs", root)
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            if list(candidate.glob("version_*")) or list(
                candidate.rglob("events.out.tfevents.*")
            ):
                return candidate

        expected = root / "lightning_logs"
        expected.mkdir(parents=True, exist_ok=True)
        return expected

    if not log_roots:
        raise ValueError("At least one log root is required")
    expected = log_roots[0] / "lightning_logs"
    expected.mkdir(parents=True, exist_ok=True)
    return expected


def load_training_scalars(*log_roots: Path) -> dict[str, list[tuple[int, float]]]:
    """Read scalar metrics from Lightning TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, dict[int, float]] = {}
    seen_dirs: set[Path] = set()

    search_roots = []
    for root in log_roots:
        if not root.is_dir():
            continue
        search_roots.extend([root / "lightning_logs", root])

    for root in search_roots:
        if not root.is_dir():
            continue

        candidate_dirs = sorted(root.rglob("version_*"))
        if not candidate_dirs:
            candidate_dirs = [root]

        for log_dir in candidate_dirs:
            if not log_dir.is_dir() or log_dir in seen_dirs:
                continue
            if not list(log_dir.glob("events.out.tfevents.*")):
                continue
            seen_dirs.add(log_dir)

            accumulator = EventAccumulator(str(log_dir))
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                bucket = merged.setdefault(tag, {})
                for event in accumulator.Scalars(tag):
                    bucket[event.step] = event.value

    return {
        tag: sorted(step_values.items())
        for tag, step_values in merged.items()
    }


def plot_training_curves(
    *log_roots: Path,
    metrics: tuple[str, ...] = ("loss_g", "loss_d", "val_loss"),
):
    """Plot training/validation loss curves from TensorBoard logs."""
    import matplotlib.pyplot as plt

    scalars = load_training_scalars(*log_roots)
    if not scalars:
        searched = ", ".join(str(root) for root in log_roots)
        raise FileNotFoundError(
            f"No TensorBoard metrics found under: {searched}\n"
            "Run training first, or wait until a few steps have been logged."
        )

    def _resolve_tag(name: str) -> str | None:
        if name in scalars:
            return name
        for tag in scalars:
            if tag.endswith(f"/{name}") or tag == name:
                return tag
        return None

    train_tags = [tag for name in ("loss_g", "loss_d") if (tag := _resolve_tag(name))]
    val_tag = _resolve_tag("val_loss")

    fig, axes = plt.subplots(1, 2 if val_tag else 1, figsize=(14, 5), squeeze=False)
    train_ax = axes[0, 0]

    for tag in train_tags:
        steps, values = zip(*scalars[tag])
        train_ax.plot(steps, values, label=tag.split("/")[-1], linewidth=1.5)
    train_ax.set_title("Training losses")
    train_ax.set_xlabel("Step")
    train_ax.set_ylabel("Loss")
    train_ax.grid(True, alpha=0.3)
    if train_tags:
        train_ax.legend()

    if val_tag:
        val_ax = axes[0, 1]
        steps, values = zip(*scalars[val_tag])
        val_ax.plot(steps, values, color="tab:orange", label="val_loss", linewidth=1.8)
        val_ax.set_title("Validation loss")
        val_ax.set_xlabel("Step")
        val_ax.set_ylabel("Loss")
        val_ax.grid(True, alpha=0.3)
        val_ax.legend()

    fig.suptitle("Piper fine-tuning progress", fontsize=13)
    fig.tight_layout()
    plt.show()

    print("Logged metrics:", ", ".join(sorted(scalars)))
    for name in metrics:
        tag = _resolve_tag(name)
        if tag and scalars[tag]:
            latest_step, latest_value = scalars[tag][-1]
            print(f"  {name}: {latest_value:.4f} @ step {latest_step}")


def training_resume_status(
    *,
    lightning_root: Path,
    max_epochs: int,
    onnx_out: Path | None = None,
) -> dict[str, Any]:
    """Summarize training progress for Colab status cells."""
    resume_ckpt = find_resume_checkpoint(lightning_root)
    status: dict[str, Any] = {
        "lightning_root": lightning_root,
        "resume_checkpoint": resume_ckpt,
        "can_resume": resume_ckpt is not None,
        "max_epochs": max_epochs,
        "onnx_exists": bool(onnx_out and onnx_out.exists()),
    }

    if resume_ckpt is not None:
        meta = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        completed_epoch = int(meta.get("epoch", -1))
        status["completed_epoch"] = completed_epoch
        status["next_epoch"] = completed_epoch + 1
        status["global_step"] = int(meta.get("global_step", 0))
        status["epochs_remaining"] = max(0, max_epochs - completed_epoch - 1)

    return status
