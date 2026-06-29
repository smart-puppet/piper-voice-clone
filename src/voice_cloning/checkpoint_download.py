"""Download Piper pretrained checkpoints from Hugging Face."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voice_cloning.hf_utils import resolve_hf_token

DEFAULT_REPO_ID = "rhasspy/piper-checkpoints"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPOCH_CKPT_RE = re.compile(r"epoch=\d+-step=\d+\.ckpt$")


@dataclass(frozen=True)
class CheckpointPreset:
    """Shortcut for a voice under rhasspy/piper-checkpoints."""

    key: str
    lang: str
    locale: str
    voice: str
    quality: str = "medium"
    description: str = ""

    @property
    def prefix(self) -> str:
        return f"{self.lang}/{self.locale}/{self.voice}/{self.quality}/"


CUSTOM_CHECKPOINT_KEY = "custom"

PRESETS: dict[str, CheckpointPreset] = {
    "de-thorsten-medium": CheckpointPreset(
        key="de-thorsten-medium",
        lang="de",
        locale="de_DE",
        voice="thorsten",
        quality="medium",
        description="German Thorsten (recommended for de datasets)",
    ),
    "de-thorsten-high": CheckpointPreset(
        key="de-thorsten-high",
        lang="de",
        locale="de_DE",
        voice="thorsten",
        quality="high",
        description="German Thorsten high quality",
    ),
    "de-thorsten-emotional-medium": CheckpointPreset(
        key="de-thorsten-emotional-medium",
        lang="de",
        locale="de_DE",
        voice="thorsten_emotional",
        quality="medium",
        description="German Thorsten emotional medium",
    ),
    "en-lessac-medium": CheckpointPreset(
        key="en-lessac-medium",
        lang="en",
        locale="en_US",
        voice="lessac",
        quality="medium",
        description="US English Lessac",
    ),
    "en-amy-medium": CheckpointPreset(
        key="en-amy-medium",
        lang="en",
        locale="en_US",
        voice="amy",
        quality="medium",
        description="US English Amy",
    ),
    "fr-siwis-medium": CheckpointPreset(
        key="fr-siwis-medium",
        lang="fr",
        locale="fr_FR",
        voice="siwis",
        quality="medium",
        description="French Siwis (neutral)",
    ),
    "fr-tom-medium": CheckpointPreset(
        key="fr-tom-medium",
        lang="fr",
        locale="fr_FR",
        voice="tom",
        quality="medium",
        description="French Tom (warmer tone; closest to expressive for FR)",
    ),
}


MEDIUM_CHECKPOINT_PRESETS: dict[str, CheckpointPreset] = {
    key: preset for key, preset in PRESETS.items() if preset.quality == "medium"
}


def resolve_checkpoint_download(
    *,
    preset: str | None,
    hf_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (preset_key, hf_path) for download_piper_checkpoint."""
    if preset and preset.strip().lower() == CUSTOM_CHECKPOINT_KEY:
        path = (hf_path or "").strip().lstrip("/")
        if not path:
            raise ValueError(
                "CHECKPOINT_PRESET is 'custom' — set CUSTOM_CHECKPOINT_HF_PATH "
                "(path inside rhasspy/piper-checkpoints, e.g. "
                "de/de_DE/thorsten/medium/epoch=3135-step=2702056.ckpt)"
            )
        return None, path
    if not preset:
        raise ValueError("CHECKPOINT_PRESET is required unless using custom HF path")
    return preset.strip().lower(), None


def default_checkpoint_path(project_root: Path | None = None) -> Path:
    if os.environ.get("PIPER_COLAB") == "1":
        return Path("/content/piper-work/checkpoints/pretrained-finetune.ckpt")

    colab_ckpt = Path("/content/piper-work/checkpoints/pretrained-finetune.ckpt")
    if colab_ckpt.parent.parent.is_dir():
        return colab_ckpt

    root = project_root or PROJECT_ROOT
    return (root / "checkpoints" / "pretrained-finetune.ckpt").resolve()


def list_repo_checkpoints(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    lang: str | None = None,
    token: str | None = None,
) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(
        repo_id,
        repo_type="dataset",
        token=resolve_hf_token(token),
    )
    ckpts = sorted(path for path in files if path.endswith(".ckpt"))
    if lang:
        prefix = f"{lang.strip().lower()}/"
        ckpts = [path for path in ckpts if path.startswith(prefix)]
    return ckpts


def resolve_checkpoint_path(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    preset: str | None = None,
    hf_path: str | None = None,
    token: str | None = None,
) -> str:
    if hf_path:
        return hf_path.lstrip("/")

    if not preset:
        raise ValueError("Either --preset or --path is required")

    key = preset.strip().lower()
    if key not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {preset!r}. Known presets: {known}")

    spec = PRESETS[key]
    matches = [
        path
        for path in list_repo_checkpoints(repo_id=repo_id, token=token)
        if path.startswith(spec.prefix) and EPOCH_CKPT_RE.search(path)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint files found under {spec.prefix} in {repo_id}"
        )

    return sorted(matches, key=_checkpoint_sort_key)[-1]


def _checkpoint_sort_key(path: str) -> tuple[int, int]:
    match = re.search(r"epoch=(\d+)-step=(\d+)", path)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def download_hf_checkpoint(
    *,
    repo_id: str,
    hf_path: str,
    dest: Path,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {hf_path} from huggingface.co/datasets/{repo_id} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=hf_path,
        revision=revision,
        token=resolve_hf_token(token),
    )
    src = Path(downloaded)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def strip_checkpoint_weights(src: Path, dest: Path) -> Path:
    """Keep only model weights for --model.pretrained_ckpt fine-tuning."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to strip optimizer state. "
            "Install project dependencies first, or pass --keep-optimizer."
        ) from exc

    payload = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        shutil.copy2(src, dest)
        return dest

    torch.save({"state_dict": payload["state_dict"]}, dest)
    return dest


def prepare_pretrained_checkpoint(
    src: Path,
    dest: Path,
    *,
    strip_optimizer: bool = True,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if strip_optimizer:
        strip_checkpoint_weights(src, dest)
        print(f"Weights-only checkpoint: {dest}")
    else:
        shutil.copy2(src, dest)
        print(f"Checkpoint saved: {dest}")
    return dest


def download_piper_checkpoint(
    *,
    dest: Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    preset: str | None = None,
    hf_path: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    strip_optimizer: bool = True,
    force: bool = False,
    project_root: Path | None = None,
) -> Path:
    output = (dest or default_checkpoint_path(project_root)).resolve()
    if output.exists() and not force:
        raise FileExistsError(
            f"Checkpoint already exists: {output}\nUse --force to replace it."
        )

    resolved_path = resolve_checkpoint_path(
        repo_id=repo_id,
        preset=preset,
        hf_path=hf_path,
        token=token,
    )
    print(f"Selected: {resolved_path}")

    with tempfile.TemporaryDirectory(prefix="piper-ckpt-") as tmp:
        raw = Path(tmp) / Path(resolved_path).name
        download_hf_checkpoint(
            repo_id=repo_id,
            hf_path=resolved_path,
            dest=raw,
            token=token,
            revision=revision,
        )
        return prepare_pretrained_checkpoint(
            raw,
            output,
            strip_optimizer=strip_optimizer,
        )
