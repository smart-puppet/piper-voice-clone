#!/usr/bin/env python3
"""Copy Colab Piper training patches into a piper1-gpl checkout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_piper_patches.py /path/to/piper1-gpl")

    piper_dir = Path(sys.argv[1]).resolve()
    project_root = Path(__file__).resolve().parents[1]
    patch_src = project_root / "colab" / "piper_patches" / "src"
    if not patch_src.is_dir():
        raise FileNotFoundError(patch_src)

    for patch_file in sorted(patch_src.rglob("*")):
        if not patch_file.is_file():
            continue
        rel = patch_file.relative_to(patch_src)
        dest = piper_dir / "src" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(patch_file, dest)
        print(f"Patched {rel}")

    print("Done")


if __name__ == "__main__":
    main()
