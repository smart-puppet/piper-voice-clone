import logging
import os
import pathlib

import torch
from lightning.pytorch.cli import LightningCLI

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

# Piper checkpoints predate PyTorch 2.6 safe unpickling defaults.
torch.serialization.add_safe_globals([pathlib.PosixPath])


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")

    def _parse_ckpt_path(self) -> None:
        """Skip merging legacy checkpoint hyperparameters; Trainer loads weights."""
        return


def _trainer_defaults() -> dict:
    defaults: dict = {"max_epochs": -1}
    if os.environ.get("PIPER_COLAB"):
        from .colab_callbacks import ColabProgressCallback, colab_extra_callbacks

        defaults["enable_progress_bar"] = False
        defaults["callbacks"] = [ColabProgressCallback(), *colab_extra_callbacks()]
    return defaults


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel, VitsDataModule, trainer_defaults=_trainer_defaults()
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
