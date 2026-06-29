"""Helpers for Piper fine-tuning CLI arguments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def phoneme_duration_onnx_path(output_onnx: Path) -> Path:
    """Companion ONNX path with audio + phoneme duration outputs."""
    return output_onnx.with_name(f"{output_onnx.stem}.phoneme_duration.onnx")


def export_onnx_models(
    *,
    checkpoint: Path,
    output_onnx: Path,
    cwd: Path | None = None,
) -> tuple[Path, Path]:
    """Export standard Piper ONNX and phoneme-duration ONNX from a checkpoint."""
    import shutil

    import onnx
    from piper.patch_voice_with_alignment import add_alignment_output

    output_onnx = output_onnx.resolve()
    phoneme_onnx = phoneme_duration_onnx_path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    run_cwd = str(cwd.resolve()) if cwd is not None else None
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.train.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--output-file",
            str(output_onnx),
        ],
        check=True,
        cwd=run_cwd,
    )

    shutil.copy2(output_onnx, phoneme_onnx)
    model = onnx.load(str(phoneme_onnx))
    add_alignment_output(model)
    onnx.save(model, str(phoneme_onnx))

    return output_onnx, phoneme_onnx


def resolve_metadata_csv(dataset_dir: Path) -> Path:
    """Prefer full metadata.csv so Piper can build a validation split."""
    for name in ("metadata.csv", "metadata_train.csv"):
        path = dataset_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No metadata.csv or metadata_train.csv under {dataset_dir}"
    )


def count_metadata_rows(csv_path: Path) -> int:
    count = 0
    with csv_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def dataset_training_args(dataset_dir: Path) -> dict[str, Path | float | int]:
    """Pick csv_path and validation settings that always leave val samples."""
    csv_path = resolve_metadata_csv(dataset_dir)
    num_utterances = count_metadata_rows(csv_path)

    if num_utterances < 3:
        raise ValueError(
            f"Dataset too small ({num_utterances} utterances in {csv_path.name}). "
            "Generate at least 15–20 samples before fine-tuning."
        )

    num_test_examples = 1 if num_utterances < 50 else 5
    validation_split = 0.1 if num_utterances >= 50 else max(0.12, 1.0 / num_utterances)

    while int(num_utterances * validation_split) < 1:
        validation_split = min(0.5, validation_split + 0.05)

    train_size = num_utterances - int(num_utterances * validation_split) - num_test_examples
    if train_size < 1:
        num_test_examples = 1
        validation_split = max(0.12, 1.0 / num_utterances)
        train_size = num_utterances - int(num_utterances * validation_split) - num_test_examples

    if train_size < 1:
        raise ValueError(
            f"Dataset too small ({num_utterances} utterances) to split into "
            "train/validation/test. Generate more audio samples."
        )

    return {
        "csv_path": csv_path,
        "validation_split": validation_split,
        "num_test_examples": num_test_examples,
        "num_utterances": num_utterances,
    }
