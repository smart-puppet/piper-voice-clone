"""Piper-compatible metadata helpers (thorsten-neutral_v03 format)."""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetadataRow:
    utterance_id: str
    text: str
    speaker: str | None = None

    @staticmethod
    def make_id(text: str, speaker: str | None = None) -> str:
        payload = f"{speaker}|{text}" if speaker else text
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_text(cls, text: str, speaker: str | None = None) -> MetadataRow:
        return cls(
            utterance_id=cls.make_id(text, speaker),
            text=text,
            speaker=speaker,
        )


def write_metadata_csv(rows: list[MetadataRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="|", lineterminator="\n")
        for row in rows:
            writer.writerow([row.utterance_id, row.text])


def split_train_val(
    rows: list[MetadataRow],
    val_ratio: float = 0.12,
    seed: int = 42,
) -> tuple[list[MetadataRow], list[MetadataRow]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    return train_rows, val_rows


def write_piper_dataset(
    rows: list[MetadataRow],
    output_dir: Path,
    val_ratio: float = 0.12,
    seed: int = 42,
) -> None:
    """Write metadata.csv, metadata_shuf.csv, metadata_train.csv, metadata_val.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    write_metadata_csv(rows, output_dir / "metadata.csv")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    write_metadata_csv(shuffled, output_dir / "metadata_shuf.csv")

    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)
    write_metadata_csv(train_rows, output_dir / "metadata_train.csv")
    write_metadata_csv(val_rows, output_dir / "metadata_val.csv")


@dataclass(frozen=True)
class DatasetPreviewSample:
    utterance_id: str
    text: str
    wav_path: Path


def read_metadata_csv(path: Path) -> list[MetadataRow]:
    rows: list[MetadataRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for parts in reader:
            if len(parts) < 2:
                continue
            utterance_id = parts[0].strip()
            text = parts[1].strip()
            if utterance_id and text:
                rows.append(MetadataRow(utterance_id=utterance_id, text=text))
    return rows


def resolve_dataset_metadata_csv(dataset_dir: Path) -> Path:
    for name in ("metadata.csv", "metadata_train.csv"):
        path = dataset_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No metadata.csv or metadata_train.csv under {dataset_dir}"
    )


def list_dataset_wav_samples(dataset_dir: Path) -> list[DatasetPreviewSample]:
    wav_dir = dataset_dir / "wavs"
    samples: list[DatasetPreviewSample] = []
    for row in read_metadata_csv(resolve_dataset_metadata_csv(dataset_dir)):
        wav_path = wav_dir / f"{row.utterance_id}.wav"
        if wav_path.is_file():
            samples.append(
                DatasetPreviewSample(
                    utterance_id=row.utterance_id,
                    text=row.text,
                    wav_path=wav_path,
                )
            )
    return samples


def sample_dataset_previews(
    dataset_dir: Path,
    *,
    count: int = 10,
    seed: int = 42,
) -> tuple[list[DatasetPreviewSample], int]:
    """Return random WAV samples and total available count."""
    available = list_dataset_wav_samples(dataset_dir)
    total = len(available)
    if total == 0:
        raise FileNotFoundError(
            f"No WAV files with matching metadata under {dataset_dir / 'wavs'}"
        )

    requested = count if count > 0 else 10
    take = min(requested, total)
    picked = random.Random(seed).sample(available, k=take)
    return picked, total
