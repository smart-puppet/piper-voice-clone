"""Audio utilities for Piper-compatible output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


PIPER_SAMPLE_RATE = 22050
QWEN3_SAMPLE_RATE = 24000
# Backward-compatible aliases
F5_SAMPLE_RATE = QWEN3_SAMPLE_RATE
NEUTTS_SAMPLE_RATE = QWEN3_SAMPLE_RATE


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def resample_for_piper(
    audio: np.ndarray,
    source_rate: int = QWEN3_SAMPLE_RATE,
    target_rate: int = PIPER_SAMPLE_RATE,
) -> np.ndarray:
    """Resample 24 kHz TTS output to Piper's 22050 Hz."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if source_rate == target_rate:
        return audio
    return librosa.resample(
        audio,
        orig_sr=source_rate,
        target_sr=target_rate,
        res_type="kaiser_best",
    )
