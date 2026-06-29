"""Qwen3-TTS voice cloning via faster-qwen3-tts (PyTorch + CUDA graphs)."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

_LOGGER = logging.getLogger(__name__)

QWEN3_SAMPLE_RATE = 24_000
DEFAULT_BACKEND = "faster"
DEFAULT_LANGUAGE = "German"
DEFAULT_MODEL_SIZE = "0.6b"
DEFAULT_MODE = "base"

VALID_BACKENDS = {"faster", "qwentts"}
VALID_MODES = {"base", "customvoice", "voicedesign"}
VALID_SIZES = {"0.6b", "1.7b"}


def hf_model_id(size: str, mode: str) -> str:
    size_key = size.lower()
    mode_key = mode.lower()
    if size_key not in VALID_SIZES:
        raise ValueError(f"Unsupported model_size {size!r} for faster backend")
    if mode_key not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r} for faster backend")

    label = {"0.6b": "0.6B", "1.7b": "1.7B"}[size_key]
    suffix = {
        "base": "Base",
        "customvoice": "CustomVoice",
        "voicedesign": "VoiceDesign",
    }[mode_key]
    return f"Qwen/Qwen3-TTS-12Hz-{label}-{suffix}"


def resolve_hf_model(
    *,
    size: str,
    mode: str,
    configured: str | None = None,
) -> str:
    if configured:
        return configured
    return hf_model_id(size, mode)


def _concat_audio(audio_list: list, sample_rate: int) -> np.ndarray:
    if not audio_list:
        raise RuntimeError("faster-qwen3-tts returned no audio")
    chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in audio_list]
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    if sample_rate != QWEN3_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=QWEN3_SAMPLE_RATE)
    return audio.astype(np.float32, copy=False)


class FasterQwen3TTSEngine:
    """Voice cloning through the faster-qwen3-tts Python package."""

    def __init__(
        self,
        *,
        model_name: str,
        mode: str = DEFAULT_MODE,
        language: str = DEFAULT_LANGUAGE,
        device: str = "cuda",
        speaker: str | None = None,
        temperature: float | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        from faster_qwen3_tts import FasterQwen3TTS

        self.mode = mode.lower()
        self.language = language
        self.device = device
        self.speaker = speaker
        self.temperature = temperature
        self.dtype = dtype

        self._ref_wav: Path | None = None
        self._ref_text: str = ""
        self._instruct: str | None = None

        _LOGGER.info("Loading faster-qwen3-tts model %s on %s", model_name, device)
        self._model = FasterQwen3TTS.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )

    def set_reference(self, wav_path: Path, ref_text: str) -> None:
        if not wav_path.exists():
            raise FileNotFoundError(f"Reference WAV not found: {wav_path}")
        self._ref_wav = wav_path.resolve()
        self._ref_text = ref_text.strip()
        duration = librosa.get_duration(path=self._ref_wav)
        _LOGGER.info(
            "faster-qwen3-tts reference: %s (%.1fs, %d words)",
            wav_path.name,
            duration,
            len(self._ref_text.split()),
        )

    def set_instruct(self, instruct: str | None) -> None:
        self._instruct = instruct.strip() if instruct else None

    def set_temperature(self, temperature: float | None) -> None:
        self.temperature = temperature

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("Cannot synthesize empty text")

        temperature = 0.9 if self.temperature is None else float(self.temperature)
        gen_kwargs = {
            "text": text,
            "language": self.language,
            "temperature": temperature,
        }

        if self.mode == "base":
            if self._ref_wav is None or not self._ref_text:
                raise RuntimeError("base mode requires reference wav + transcript")
            audio_list, sr = self._model.generate_voice_clone(
                ref_audio=str(self._ref_wav),
                ref_text=self._ref_text,
                **gen_kwargs,
            )
        elif self.mode == "customvoice":
            if not self.speaker:
                raise RuntimeError("customvoice mode requires tts.speaker")
            audio_list, sr = self._model.generate_custom_voice(
                speaker=self.speaker,
                instruct=self._instruct,
                **gen_kwargs,
            )
        elif self.mode == "voicedesign":
            if not self._instruct:
                raise RuntimeError("voicedesign mode requires instruct text")
            audio_list, sr = self._model.generate_voice_design(
                instruct=self._instruct,
                **gen_kwargs,
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return _concat_audio(audio_list, sr)
