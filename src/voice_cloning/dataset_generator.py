"""Generate Piper training datasets using Qwen3-TTS voice cloning."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml
from tqdm import tqdm

from voice_cloning.audio_utils import (
    PIPER_SAMPLE_RATE,
    QWEN3_SAMPLE_RATE,
    resample_for_piper,
    save_wav,
)
from voice_cloning.book_loader import (
    DEFAULT_BOOK_SOURCE,
    MAX_BOOK_SENTENCE_WORDS,
    filter_sentences_by_max_words,
    load_sentences,
)
from voice_cloning.faster_qwen3tts_engine import (
    FasterQwen3TTSEngine,
    resolve_hf_model,
)
from voice_cloning.config_paths import (
    read_text_file,
    require_file,
    resolve_config_path,
)
from voice_cloning.metadata import MetadataRow, write_piper_dataset
from voice_cloning.qwen3tts_engine import (
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODE,
    DEFAULT_MODEL_REPO,
    DEFAULT_MODEL_SIZE,
    DEFAULT_QUANT,
    Qwen3TTSEngine,
    ensure_qwen3_models,
    resolve_model_dir,
    resolve_qwen_tts_bin,
)

_LOGGER = logging.getLogger(__name__)


def _use_line_progress() -> bool:
    """Colab / notebook subprocess: tqdm \\r bars do not render; use line prints."""
    if os.environ.get("PIPER_COLAB"):
        return True
    return not sys.stdout.isatty()


def _progress_bar(step: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = max(0, min(width, int(width * step / total)))
    return "#" * filled + "-" * (width - filled)


def _synthesis_progress(
    tasks: list[str],
    *,
    desc: str,
    already_done: int,
    total_all: int,
) -> Iterator[str]:
    """Yield utterances with tqdm (terminal) or line-based progress (Colab)."""
    pending = len(tasks)
    if pending == 0:
        return

    if not _use_line_progress():
        yield from tqdm(
            tasks,
            desc=desc,
            unit="utt",
            initial=already_done,
            total=total_all,
        )
        return

    batch_times: list[float] = []
    for index, sentence in enumerate(tasks):
        started = time.perf_counter()
        yield sentence
        batch_times.append(time.perf_counter() - started)
        step = already_done + index + 1
        pct = 100.0 * step / total_all
        avg_sec = sum(batch_times) / len(batch_times)
        last_sec = batch_times[-1]
        print(
            f"{desc}: {pct:3.0f}%|{_progress_bar(index + 1, pending)}| "
            f"{step}/{total_all} utt "
            f"avg={avg_sec:.2f}s/utt last={last_sec:.2f}s",
            flush=True,
        )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_VERSION = "v03"
DEFAULT_LANGUAGE_CODE = "de"


def dataset_dir_name(
    voice_name: str,
    language_code: str,
    dataset_version: str,
) -> str:
    """Build Piper dataset directory name: ``{voice}-{lang}_{version}``."""
    lang = language_code.strip().lower()
    version = dataset_version.strip().lstrip("_")
    if not re.fullmatch(r"[a-z0-9_-]+", lang):
        raise ValueError(f"Invalid language_code {language_code!r}")
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError(
            f"Invalid dataset_version {dataset_version!r} (expected e.g. v03)"
        )
    return f"{voice_name}-{lang}_{version}"


@dataclass
class StyleReference:
    name: str
    wav_path: Path
    ref_text: str


@dataclass
class GeneratorConfig:
    voice_name: str
    output_root: Path
    total_entries: int
    reference: StyleReference
    backend: str
    qwen_bin: Path | None
    model_dir: Path | None
    hf_model: str | None
    model_repo: str
    model_size: str
    mode: str
    quant: str
    language: str
    language_code: str
    dataset_version: str
    espeak_voice: str
    device: str
    dtype: str
    speaker: str | None
    tts_seed: int
    temperature: float | None
    book_source: str
    book_path: Path | None
    val_ratio: float
    seed: int
    resume: bool
    max_sentence_words: int

    @property
    def dataset_dir(self) -> Path:
        name = dataset_dir_name(
            self.voice_name,
            self.language_code,
            self.dataset_version,
        )
        return self.output_root / name

    @property
    def wav_dir(self) -> Path:
        return self.dataset_dir / "wavs"


def _load_reference(raw: dict[str, Any], config_dir: Path) -> StyleReference:
    if "reference" not in raw:
        raise ValueError(
            "Config must define a single 'reference' block with 'wav' and "
            "'ref_text' or 'ref_text_file'."
        )

    ref = raw["reference"]
    if "wav" not in ref:
        raise ValueError("reference block must include 'wav'")

    wav_path = resolve_config_path(ref["wav"], config_dir)
    require_file(wav_path, label="reference WAV")

    ref_text = ref.get("ref_text")
    if not ref_text:
        ref_text_path = ref.get("ref_text_file")
        if not ref_text_path:
            raise ValueError(
                "reference needs 'ref_text' or 'ref_text_file' in config.yaml"
            )
        ref_text_file = resolve_config_path(ref_text_path, config_dir)
        ref_text = read_text_file(ref_text_file, label="reference transcript")

    if not ref_text:
        raise ValueError("reference transcript is empty")

    return StyleReference(
        name=ref.get("name", "default"),
        wav_path=wav_path,
        ref_text=ref_text,
    )


def _resolve_optional_path(value: str | None, config_dir: Path) -> Path | None:
    if not value:
        return None
    return resolve_config_path(value, config_dir)


def load_config(path: Path) -> GeneratorConfig:
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found:\n  {path.resolve()}\n"
            "Pass --config with a valid config.yaml path."
        )

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    config_dir = path.parent.resolve()
    reference = _load_reference(raw, config_dir)
    tts = raw.get("tts", {})

    book_path = raw.get("book_path")
    resolved_book_path = None
    if book_path:
        resolved_book_path = resolve_config_path(book_path, config_dir)
        require_file(resolved_book_path, label="book_path")

    output_root = Path(raw.get("output_root", "datasets"))
    if not output_root.is_absolute():
        output_root = (config_dir / output_root).resolve()

    emotion = raw.get("emotion") or tts.get("emotion")
    if emotion or raw.get("emotion_profiles"):
        _LOGGER.warning(
            "emotion and emotion_profiles are deprecated — use the single 'reference' block only"
        )

    backend = str(tts.get("backend", "qwentts")).lower()
    device = str(tts.get("device", DEFAULT_DEVICE))
    if backend == "faster" and device.upper().startswith("CUDA"):
        device = "cuda"

    language_code = str(
        raw.get("language_code", tts.get("language_code", DEFAULT_LANGUAGE_CODE))
    ).lower()
    dataset_version = str(
        raw.get("dataset_version", tts.get("dataset_version", DEFAULT_DATASET_VERSION))
    )
    espeak_voice = str(
        raw.get("espeak_voice", tts.get("espeak_voice", language_code))
    )

    return GeneratorConfig(
        voice_name=raw["voice_name"],
        output_root=output_root,
        total_entries=int(raw.get("total_entries", 5000)),
        reference=reference,
        backend=backend,
        qwen_bin=_resolve_optional_path(tts.get("bin"), config_dir),
        model_dir=_resolve_optional_path(tts.get("model_dir"), config_dir),
        hf_model=tts.get("hf_model"),
        model_repo=str(tts.get("model_repo", DEFAULT_MODEL_REPO)),
        model_size=str(tts.get("model_size", DEFAULT_MODEL_SIZE)).lower(),
        mode=str(tts.get("mode", DEFAULT_MODE)).lower(),
        quant=str(tts.get("quant", DEFAULT_QUANT)),
        language=str(tts.get("language", DEFAULT_LANGUAGE)),
        language_code=language_code,
        dataset_version=dataset_version,
        espeak_voice=espeak_voice,
        device=device,
        dtype=str(tts.get("dtype", "bfloat16")),
        speaker=tts.get("speaker"),
        tts_seed=int(tts.get("seed", raw.get("seed", 42))),
        temperature=tts.get("temperature"),
        book_source=raw.get("book_source") or raw.get("book_url", DEFAULT_BOOK_SOURCE),
        book_path=resolved_book_path,
        val_ratio=float(raw.get("val_ratio", 0.12)),
        seed=int(raw.get("seed", 42)),
        resume=bool(raw.get("resume", True)),
        max_sentence_words=int(raw.get("max_sentence_words", MAX_BOOK_SENTENCE_WORDS)),
    )


class Qwen3TTSSynthesizer:
    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config

        if config.backend == "faster":
            model_name = resolve_hf_model(
                size=config.model_size,
                mode=config.mode,
                configured=config.hf_model,
            )
            self._engine = FasterQwen3TTSEngine(
                model_name=model_name,
                mode=config.mode,
                language=config.language,
                device=config.device,
                speaker=config.speaker,
                temperature=config.temperature,
                dtype=config.dtype,
            )
        elif config.backend == "qwentts":
            bin_path = resolve_qwen_tts_bin(config.qwen_bin, PROJECT_ROOT)
            model_dir = resolve_model_dir(config.model_dir, PROJECT_ROOT)
            talker, codec = ensure_qwen3_models(
                PROJECT_ROOT,
                model_dir,
                size=config.model_size,
                mode=config.mode,
                quant=config.quant,
                repo=config.model_repo,
            )
            self._engine = Qwen3TTSEngine(
                bin_path=bin_path,
                talker_model=talker,
                codec_model=codec,
                mode=config.mode,
                language=config.language,
                device=config.device,
                speaker=config.speaker,
                seed=config.tts_seed,
                temperature=config.temperature,
            )
        else:
            raise ValueError(
                f"Unsupported tts.backend {config.backend!r} (use 'faster' or 'qwentts')"
            )

    def encode_style(self, style: StyleReference) -> None:
        self._engine.set_reference(style.wav_path, style.ref_text)
        if self._config.temperature is not None:
            self._engine.set_temperature(self._config.temperature)

    def synthesize(self, text: str, style: StyleReference) -> list[float]:
        del style
        return list(self._engine.synthesize(text))


def _progress_path(dataset_dir: Path) -> Path:
    return dataset_dir / ".generation_progress.json"


def _manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / ".generation_manifest.json"


def _load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("completed_ids", []))


def _save_progress(path: Path, completed_ids: set[str]) -> None:
    path.write_text(
        json.dumps({"completed_ids": sorted(completed_ids)}, indent=2),
        encoding="utf-8",
    )


def _scan_completed_wavs(wav_dir: Path) -> set[str]:
    if not wav_dir.is_dir():
        return set()
    return {path.stem for path in wav_dir.glob("*.wav")}


def _load_manifest(path: Path, config: GeneratorConfig) -> list[str] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("seed") != config.seed
        or data.get("book_source") != config.book_source
        or data.get("voice_name") != config.voice_name
        or data.get("language_code") != config.language_code
        or data.get("dataset_version") != config.dataset_version
    ):
        _LOGGER.warning(
            "Generation manifest does not match config — rebuilding sentence list"
        )
        return None
    return list(data.get("sentences", []))


def _save_manifest(path: Path, config: GeneratorConfig, sentences: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "voice_name": config.voice_name,
                "language_code": config.language_code,
                "dataset_version": config.dataset_version,
                "seed": config.seed,
                "book_source": config.book_source,
                "total_entries": config.total_entries,
                "sentences": sentences,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _resolve_sentences(config: GeneratorConfig) -> list[str]:
    manifest_path = _manifest_path(config.dataset_dir)
    manifest = _load_manifest(manifest_path, config)

    if manifest is not None:
        manifest = filter_sentences_by_max_words(
            manifest,
            config.max_sentence_words,
            log_label="cached manifest",
        )
        if len(manifest) >= config.total_entries:
            return manifest[: config.total_entries]
        _LOGGER.info(
            "Manifest has %d sentences, need %d — extending",
            len(manifest),
            config.total_entries,
        )

    sentences = load_sentences(
        book_path=config.book_path,
        book_source=config.book_source,
        cache_dir=config.dataset_dir / "source_book",
        limit=config.total_entries,
        seed=config.seed,
        max_words=config.max_sentence_words,
    )
    _save_manifest(manifest_path, config, sentences)
    return sentences


def dataset_resume_status(config: GeneratorConfig) -> dict[str, int | Path | bool | str]:
    """Return progress counts for UI / Colab status cells."""
    wav_dir = config.wav_dir
    progress_path = _progress_path(config.dataset_dir)
    completed_ids = set()
    if config.resume:
        completed_ids |= _load_progress(progress_path)
        completed_ids |= _scan_completed_wavs(wav_dir)

    wav_on_disk = len(_scan_completed_wavs(wav_dir))

    try:
        sentences = _resolve_sentences(config)
    except Exception:
        sentences = []

    if sentences:
        done = sum(
            1
            for row in (MetadataRow.from_text(s) for s in sentences)
            if row.utterance_id in completed_ids
            and (wav_dir / f"{row.utterance_id}.wav").exists()
        )
        total = len(sentences)
    else:
        done = wav_on_disk
        total = config.total_entries

    return {
        "dataset_dir": config.dataset_dir,
        "dataset_name": config.dataset_dir.name,
        "total": total,
        "done": done,
        "remaining": max(0, total - done),
        "wav_on_disk": wav_on_disk,
        "resume": config.resume,
    }


def generate_dataset(config: GeneratorConfig) -> Path:
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    config.wav_dir.mkdir(parents=True, exist_ok=True)

    sentences = _resolve_sentences(config)

    style = config.reference
    synthesizer = Qwen3TTSSynthesizer(config)
    if config.backend == "faster":
        _LOGGER.info(
            "Qwen3-TTS (faster): %s %s on %s → %s",
            config.model_size,
            config.mode,
            config.device,
            config.dataset_dir.name,
        )
    else:
        _LOGGER.info(
            "Qwen3-TTS (qwentts): %s %s %s on %s → %s",
            config.model_size,
            config.mode,
            config.quant,
            config.device,
            config.dataset_dir.name,
        )
    synthesizer.encode_style(style)

    progress_path = _progress_path(config.dataset_dir)
    completed_ids: set[str] = set()
    if config.resume:
        completed_ids |= _load_progress(progress_path)
        completed_ids |= _scan_completed_wavs(config.wav_dir)

    metadata_rows: list[MetadataRow] = []
    tasks: list[str] = []
    for sentence in sentences:
        row = MetadataRow.from_text(sentence)
        metadata_rows.append(row)
        wav_path = config.wav_dir / f"{row.utterance_id}.wav"
        if config.resume and row.utterance_id in completed_ids and wav_path.exists():
            continue
        if config.resume and row.utterance_id in completed_ids and not wav_path.exists():
            _LOGGER.warning("Missing WAV for %s — regenerating", row.utterance_id)
            completed_ids.discard(row.utterance_id)
        tasks.append(sentence)

    _LOGGER.info(
        "Synthesizing %d utterances (%d already done)",
        len(tasks),
        len(completed_ids),
    )

    for sentence in _synthesis_progress(
        tasks,
        desc="Synthesizing",
        already_done=len(completed_ids),
        total_all=len(metadata_rows),
    ):
        row = MetadataRow.from_text(sentence)
        wav_path = config.wav_dir / f"{row.utterance_id}.wav"

        audio = synthesizer.synthesize(sentence, style)
        audio = resample_for_piper(audio, QWEN3_SAMPLE_RATE, PIPER_SAMPLE_RATE)
        save_wav(wav_path, audio, PIPER_SAMPLE_RATE)

        completed_ids.add(row.utterance_id)
        _save_progress(progress_path, completed_ids)

    write_piper_dataset(
        metadata_rows,
        config.dataset_dir,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    _LOGGER.info("Dataset written to %s", config.dataset_dir)
    return config.dataset_dir
