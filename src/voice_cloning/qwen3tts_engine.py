"""Qwen3-TTS voice cloning via qwentts.cpp + GGUF (Serveurperso/Qwen3-TTS-GGUF)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

_LOGGER = logging.getLogger(__name__)

QWEN3_SAMPLE_RATE = 24_000
DEFAULT_MODEL_REPO = "Serveurperso/Qwen3-TTS-GGUF"
DEFAULT_MODEL_SIZE = "0.6b"
DEFAULT_MODE = "base"
DEFAULT_QUANT = "Q8_0"
DEFAULT_LANGUAGE = "German"
DEFAULT_DEVICE = "CUDA0"

VALID_MODES = {"base", "customvoice", "voicedesign"}
VALID_SIZES = {"0.6b", "1.7b"}
VALID_QUANTS = {"Q8_0", "Q4_K_M", "BF16", "F32"}

# CustomVoice preset speakers (1.7B/0.6B customvoice checkpoints).
CUSTOMVOICE_SPEAKERS = {
    "serena",
    "vivian",
    "uncle_fu",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
    "eric",
    "dylan",
}


def talker_filename(size: str, mode: str, quant: str) -> str:
    return f"qwen-talker-{size}-{mode}-{quant}.gguf"


def tokenizer_filename(quant: str) -> str:
    return f"qwen-tokenizer-12hz-{quant}.gguf"


def resolve_qwen_tts_bin(configured: Path | None, project_root: Path) -> Path:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    candidates.extend(
        [
            project_root / "qwentts.cpp/build/qwen-tts",
            project_root / "qwen3-tts.cpp/build/qwen3-tts-cli",
            project_root / "qwen3-tts.cpp/build/qwen-tts",
        ]
    )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(
        "qwen-tts binary not found. Build with:\n"
        "  ./install/install_old_gpu.sh  (or install_new_gpu.sh)"
    )


def resolve_model_dir(configured: Path | None, project_root: Path) -> Path:
    if configured is not None:
        path = configured if configured.is_absolute() else (project_root / configured)
        if path.is_dir():
            return path.resolve()
    default = project_root / "models" / "Qwen3-TTS-GGUF"
    if default.is_dir():
        return default.resolve()
    return (project_root / "models").resolve()


def download_qwen3_models(
    project_root: Path,
    *,
    size: str,
    mode: str,
    quant: str,
    repo: str = DEFAULT_MODEL_REPO,
) -> Path:
    from huggingface_hub import hf_hub_download

    model_dir = project_root / "models" / "Qwen3-TTS-GGUF"
    model_dir.mkdir(parents=True, exist_ok=True)

    talker = talker_filename(size, mode, quant)
    codec = tokenizer_filename(quant)

    for filename in (talker, codec):
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=model_dir,
        )
        _LOGGER.info("Downloaded %s", filename)

    return model_dir


def ensure_qwen3_models(
    project_root: Path,
    model_dir: Path,
    *,
    size: str,
    mode: str,
    quant: str,
    repo: str,
) -> tuple[Path, Path]:
    talker = model_dir / talker_filename(size, mode, quant)
    codec = model_dir / tokenizer_filename(quant)

    if talker.is_file() and codec.is_file():
        return talker.resolve(), codec.resolve()

    _LOGGER.info("Downloading Qwen3-TTS GGUF models (%s %s %s)...", size, mode, quant)
    model_dir = download_qwen3_models(
        project_root,
        size=size,
        mode=mode,
        quant=quant,
        repo=repo,
    )
    talker = model_dir / talker_filename(size, mode, quant)
    codec = model_dir / tokenizer_filename(quant)
    if not talker.is_file() or not codec.is_file():
        raise FileNotFoundError(f"Missing model files in {model_dir}")
    return talker.resolve(), codec.resolve()


class Qwen3TTSEngine:
    """Zero-shot / instructable TTS through qwentts.cpp."""

    def __init__(
        self,
        *,
        bin_path: Path,
        talker_model: Path,
        codec_model: Path,
        mode: str = DEFAULT_MODE,
        language: str = DEFAULT_LANGUAGE,
        device: str = DEFAULT_DEVICE,
        speaker: str | None = None,
        seed: int = 42,
        temperature: float | None = None,
    ) -> None:
        self.bin_path = bin_path
        self.talker_model = talker_model
        self.codec_model = codec_model
        self.mode = mode
        self.language = language
        self.device = device
        self.speaker = speaker
        self.seed = seed
        self.temperature = temperature

        self._ref_wav: Path | None = None
        self._ref_text: str = ""
        self._instruct: str | None = None

    def set_reference(self, wav_path: Path, ref_text: str) -> None:
        if not wav_path.exists():
            raise FileNotFoundError(f"Reference WAV not found: {wav_path}")
        self._ref_wav = wav_path.resolve()
        self._ref_text = ref_text.strip()
        duration = librosa.get_duration(path=self._ref_wav)
        _LOGGER.info(
            "Qwen3-TTS reference: %s (%.1fs, %d words)",
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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)

        cmd = [
            str(self.bin_path),
            "--model",
            str(self.talker_model),
            "--codec",
            str(self.codec_model),
            "--lang",
            self.language,
            "-o",
            str(out_path),
        ]

        env = os.environ.copy()
        env["GGML_BACKEND"] = self.device

        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])
        if self.temperature is not None:
            cmd.extend(["--temp", str(self.temperature)])

        ref_text_path: Path | None = None
        try:
            if self.mode == "base":
                if self._ref_wav is None or not self._ref_text:
                    raise RuntimeError("base mode requires reference wav + transcript")
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    delete=False,
                    encoding="utf-8",
                ) as ref_tmp:
                    ref_tmp.write(self._ref_text)
                    ref_text_path = Path(ref_tmp.name)
                cmd.extend(
                    [
                        "--ref-wav",
                        str(self._ref_wav),
                        "--ref-text",
                        str(ref_text_path),
                    ]
                )
            elif self.mode == "customvoice":
                if not self.speaker:
                    raise RuntimeError("customvoice mode requires tts.speaker")
                cmd.extend(["--speaker", self.speaker])
                if self._instruct:
                    cmd.extend(["--instruct", self._instruct])
                if self._ref_wav is not None:
                    cmd.extend(["--ref-wav", str(self._ref_wav)])
                    if self._ref_text:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            suffix=".txt",
                            delete=False,
                            encoding="utf-8",
                        ) as ref_tmp:
                            ref_tmp.write(self._ref_text)
                            ref_text_path = Path(ref_tmp.name)
                        cmd.extend(["--ref-text", str(ref_text_path)])
            elif self.mode == "voicedesign":
                if not self._instruct:
                    raise RuntimeError("voicedesign mode requires instruct text")
                cmd.extend(["--instruct", self._instruct])
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            result = subprocess.run(
                cmd,
                input=text + "\n",
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"qwen-tts failed (exit {result.returncode}):\n"
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            if not out_path.exists() or out_path.stat().st_size < 1000:
                raise RuntimeError(f"qwen-tts produced no audio for: {text!r}")

            wav, _ = librosa.load(out_path, sr=None, mono=True)
            return np.asarray(wav, dtype=np.float32)
        finally:
            out_path.unlink(missing_ok=True)
            if ref_text_path is not None:
                ref_text_path.unlink(missing_ok=True)
