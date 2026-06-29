"""Shared Hugging Face Hub helpers."""

from __future__ import annotations

import os


def resolve_hf_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
