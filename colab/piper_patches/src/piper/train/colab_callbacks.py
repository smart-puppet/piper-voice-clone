"""Notebook-friendly training progress with tqdm."""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
from pathlib import Path

import lightning as L

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is installed in Colab notebook
    tqdm = None  # type: ignore[assignment,misc]


def _format_metric(trainer: L.Trainer, name: str) -> str:
    value = trainer.callback_metrics.get(name)
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _use_line_progress() -> bool:
    """Notebook / !bash subprocess output: tqdm \\r bars do not render; use prints."""
    if os.environ.get("PIPER_COLAB"):
        return True
    return not sys.stdout.isatty()


def _safe_batch_count(trainer: L.Trainer) -> int:
    """Lightning may report inf/0 batches when len(dataset) < batch_size."""
    raw = trainer.num_training_batches
    if raw is None:
        return 0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value) or value < 1:
        return 0
    return int(value)


def _progress_bar(step: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = max(0, min(width, int(width * step / total)))
    return "#" * filled + "-" * (width - filled)


class ColabProgressCallback(L.Callback):
    """Training progress for Colab: line-based in notebooks, tqdm in terminals."""

    def __init__(self) -> None:
        self._bar = None
        self._batch_start: float | None = None
        self._last_batch_sec: float | None = None
        self._batch_times: list[float] = []
        self._line_progress = _use_line_progress()
        self._epoch_total = 0

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _open_bar(self, trainer: L.Trainer) -> None:
        self._close_bar()
        self._batch_times.clear()
        self._epoch_total = _safe_batch_count(trainer)

        if self._epoch_total <= 0:
            print(
                "Warning: no training batches this epoch — dataset may be smaller than "
                "batch size. Generate at least 50–100 utterances (recommend ≥ 2,000) "
                "before fine-tuning.",
                flush=True,
            )

        if self._line_progress or tqdm is None or self._epoch_total <= 0:
            return

        self._bar = tqdm(
            total=self._epoch_total,
            desc=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            unit="it",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    def _print_batch_progress(
        self, trainer: L.Trainer, batch_idx: int
    ) -> None:
        total = self._epoch_total or _safe_batch_count(trainer) or 1
        step = batch_idx + 1
        pct = 100.0 * step / total
        timing = ""
        if self._last_batch_sec is not None:
            timing = f" s/it={self._last_batch_sec:.2f}"
        if self._batch_times:
            avg_sec = sum(self._batch_times) / len(self._batch_times)
            timing += f" avg={avg_sec:.2f}s"
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}: "
            f"{pct:3.0f}%|{_progress_bar(step, total)}| {step}/{total} "
            f"loss_g={_format_metric(trainer, 'loss_g')} "
            f"loss_d={_format_metric(trainer, 'loss_d')}{timing}",
            flush=True,
        )

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._open_bar(trainer)

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.current_epoch > 0:
            self._open_bar(trainer)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        self._batch_start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self._batch_start is not None:
            self._last_batch_sec = time.perf_counter() - self._batch_start
            self._batch_times.append(self._last_batch_sec)
            self._batch_start = None

        if self._line_progress:
            self._print_batch_progress(trainer, batch_idx)
            return

        if self._bar is not None:
            self._bar.update(1)
            postfix: dict[str, str] = {
                "loss_g": _format_metric(trainer, "loss_g"),
                "loss_d": _format_metric(trainer, "loss_d"),
            }
            if self._last_batch_sec is not None:
                postfix["s/it"] = f"{self._last_batch_sec:.2f}"
            if self._batch_times:
                avg_sec = sum(self._batch_times) / len(self._batch_times)
                postfix["avg_s/it"] = f"{avg_sec:.2f}"
            self._bar.set_postfix(postfix, refresh=False)
            return

        interval = max(1, int(trainer.log_every_n_steps))
        if trainer.global_step % interval != 0:
            return

        total_steps = trainer.estimated_stepping_batches or "?"
        timing = ""
        if self._last_batch_sec is not None:
            timing = f" | s/it={self._last_batch_sec:.2f}"
        print(
            f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"| step {trainer.global_step}/{total_steps} "
            f"| loss_g={_format_metric(trainer, 'loss_g')} "
            f"loss_d={_format_metric(trainer, 'loss_d')}{timing}",
            flush=True,
        )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._line_progress:
            print(
                f"--- val_loss={_format_metric(trainer, 'val_loss')} "
                f"@ epoch {trainer.current_epoch + 1} ---",
                flush=True,
            )
            return

        if self._bar is not None:
            self._bar.set_postfix(
                val_loss=_format_metric(trainer, "val_loss"),
                refresh=True,
            )
            return

        print(
            f"--- val_loss={_format_metric(trainer, 'val_loss')} "
            f"@ epoch {trainer.current_epoch + 1} ---",
            flush=True,
        )

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._line_progress and self._batch_times:
            avg_sec = sum(self._batch_times) / len(self._batch_times)
            print(
                f"Epoch {trainer.current_epoch + 1} done — avg {avg_sec:.2f} s/it",
                flush=True,
            )
        elif self._batch_times:
            avg_sec = sum(self._batch_times) / len(self._batch_times)
            print(
                f"Epoch {trainer.current_epoch + 1} avg: {avg_sec:.2f} s/it",
                flush=True,
            )
        self._close_bar()

    def on_fit_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._close_bar()


def _newest_last_ckpt(lightning_root: Path) -> Path | None:
    if not lightning_root.is_dir():
        return None
    last_ckpts = list(lightning_root.rglob("last.ckpt"))
    if not last_ckpts:
        return None
    return max(last_ckpts, key=lambda path: path.stat().st_mtime)


class DriveCheckpointCallback(L.Callback):
    """Copy last.ckpt to Google Drive periodically (survives Colab disconnect)."""

    def __init__(
        self,
        local_lightning: Path,
        drive_checkpoints: Path,
        *,
        sync_every_n_epochs: int = 10,
    ) -> None:
        self.local_lightning = local_lightning.resolve()
        self.drive_checkpoints = drive_checkpoints.resolve()
        self.sync_every_n_epochs = max(1, sync_every_n_epochs)

    def _sync_to_drive(self, trainer: L.Trainer) -> None:
        if not Path("/content/drive/MyDrive").is_dir():
            return

        ckpt = _newest_last_ckpt(self.local_lightning)
        if ckpt is None or not ckpt.is_file():
            return

        try:
            rel = ckpt.relative_to(self.local_lightning)
            dst = self.drive_checkpoints / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if (not dst.exists()) or (dst.stat().st_size != ckpt.stat().st_size):
                shutil.copy2(ckpt, dst)

            latest = self.drive_checkpoints / "latest.ckpt"
            if (not latest.exists()) or (latest.stat().st_mtime < ckpt.stat().st_mtime):
                shutil.copy2(ckpt, latest)

            print(
                f"Drive checkpoint sync @ epoch {trainer.current_epoch + 1}: {latest}",
                flush=True,
            )
        except OSError as exc:
            print(f"Drive checkpoint sync failed: {exc}", flush=True)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.sync_every_n_epochs != 0:
            return
        self._sync_to_drive(trainer)

    def on_fit_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._sync_to_drive(trainer)


def colab_extra_callbacks() -> list[L.Callback]:
    """Optional callbacks configured via environment (Colab training)."""
    callbacks: list[L.Callback] = []
    drive_dir = os.environ.get("PIPER_DRIVE_CHECKPOINTS", "").strip()
    local_dir = os.environ.get("PIPER_LIGHTNING_DIR", "").strip()
    sync_every = int(os.environ.get("PIPER_DRIVE_SYNC_EVERY_N_EPOCHS", "10"))
    if drive_dir and local_dir:
        callbacks.append(
            DriveCheckpointCallback(
                local_lightning=Path(local_dir),
                drive_checkpoints=Path(drive_dir),
                sync_every_n_epochs=sync_every,
            )
        )
    return callbacks
