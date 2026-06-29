# Google Colab

**Best way to use Colab:** open **[`colab/piper_voice_clone.ipynb`](piper_voice_clone.ipynb)** and run all cells in order. Output streams inline in each cell.

[Open in Colab](https://colab.research.google.com/github/smart-puppet/piper-voice-clone/blob/main/colab/piper_voice_clone.ipynb?v=9)

### Still seeing the old notebook?

Colab often caches notebooks opened from GitHub. Re-open from GitHub (**File → Open notebook → GitHub**) or use an incognito window with the link above. Do not reopen an old copy saved to Google Drive.

## Quick start

1. **Runtime → Change runtime type → GPU → L4** (recommended)
2. Run all cells in `colab/piper_voice_clone.ipynb`

Mount Google Drive in the notebook before dataset/train publish to Drive.

## Pretrained Piper checkpoint

Fine-tuning starts from a **pretrained Piper checkpoint** — an existing voice model from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints). Your dataset adapts that base to your voice. **The checkpoint language must match your dataset language.**

**German (`de`):** `de-thorsten-emotional-medium` (expressive, recommended) or `de-thorsten-medium` (neutral)  
**English (`en`):** `en-lessac-medium` or `en-amy-medium`  
**French (`fr`):** `fr-siwis-medium` (neutral) or `fr-tom-medium` (warmer; no dedicated FR emotional checkpoint)

In the notebook, set `LANGUAGE_CODE` and `CHECKPOINT_PRESET` in **Project settings** (medium presets only), then run **Download pretrained Piper checkpoint**. Choose **`custom`** and set `CUSTOM_CHECKPOINT_HF_PATH` to use your own file from [piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints).

Browse presets: `bash colab/download_checkpoint.sh --list-presets`  
Browse German files on HF: `bash colab/download_checkpoint.sh --list --lang de`

## Book text (dataset sentences)

Pick `BOOK_PRESET` in **Project settings** — fairy-tale sentences from Project Gutenberg:

| Preset | Language | Book |
|--------|----------|------|
| `de-andersen-maerchen` | German | Andersen, *Märchen* |
| `en-grimm-fairy-tales` | English | Grimm, *Household Tales* |
| `fr-perrault-contes` | French | Perrault, *Contes de ma mère l'Oye* |

Pick **`custom`** and set `CUSTOM_BOOK_URL` for any Project Gutenberg plain-text URL.

Match checkpoint and book text to the same `LANGUAGE_CODE`.

## Recommendations

- **GPU:** **L4** on Colab Pro — best price/performance (T4 works but slower; A100 is fastest but costly).
- **Dataset:** generate **at least 2,000 samples** for a usable clone (default `total_entries`: 2,500).
- **Training time (≈2,000 samples on L4):** epoch 1 ~**30 min** (cache warmup), then ~**1 min/epoch**.

## Restore or import a dataset

**Cell 9 (Import dataset)** in the notebook can load an existing Piper dataset zip from **Google Drive** — default path `MyDrive/piper-voice-clone/archives/{voice}-{lang}_{version}.zip`

Or from the terminal:

```bash
# From Drive (after mount)
bash colab/download_dataset.sh --config config.colab.yaml --zip /content/drive/MyDrive/piper-voice-clone/archives/myvoice-de_v01.zip
```

Skip **cell 10 (Generate dataset)** if you imported a complete dataset in cell 9.

## Avoid disconnect during long training

Colab disconnects when the **notebook kernel is idle**. Dataset generation and training run in notebook cells so the kernel stays active and output stays visible.

- Run the **anti-disconnect** cell before long runs; keep the browser tab open.
- Checkpoints sync to **Drive every 10 epochs** (and at the end). After a full runtime loss, re-run setup and training — `train_piper.sh` restores `latest.ckpt` from Drive when local checkpoints are missing.

`train_piper.sh` disconnects the runtime when finished to free the GPU. In the notebook **Train** cell, set **KEEP_RUNTIME_CONNECTED** to `True` to stay connected (`--no-disconnect`).

## Terminal scripts (optional)

The same steps are available as shell scripts if you prefer a Colab terminal:

```bash
git clone https://github.com/smart-puppet/piper-voice-clone.git /content/piper-voice-clone
cd /content/piper-voice-clone
bash install/install_colab.sh
bash colab/download_checkpoint.sh --preset de-thorsten-emotional-medium
bash colab/generate_dataset.sh --config config.colab.yaml
bash colab/train_piper.sh --voice myvoice --lang de --version v01
```

Mount Drive in a **notebook cell** before publish (`drive.mount` does not work from Terminal).

## Local layout (during a session)

```
/content/piper-voice-clone/     # cloned repo
/content/piper-work/
├── datasets/myvoice-de_v01/
├── checkpoints/pretrained-finetune.ckpt
├── cache/myvoice/
├── lightning_logs/myvoice/
└── models/myvoice-de-medium.onnx
```

## Drive layout (after publish)

```
MyDrive/piper-voice-clone/
├── archives/myvoice-de_v01.zip      # dataset generation (zip only)
├── checkpoints/myvoice/latest.ckpt  # training (weights; resume source)
└── models/
    ├── myvoice-de-medium.onnx
    └── myvoice-de-medium.phoneme_duration.onnx
```

TensorBoard event files stay on **local Colab disk** only (`/content/piper-work/lightning_logs/`).

See **Restore or import a dataset** above (notebook cell 9 or `colab/download_dataset.sh`).

## Resume after disconnect

Colab local disk is wiped when the runtime ends. Re-open the notebook, re-run setup, restore the **dataset zip** from Drive if needed, then run training again — `train_piper.sh` auto-restores `latest.ckpt` from Drive when local checkpoints are missing.

Manual restore (if needed):

```bash
mkdir -p /content/piper-work/lightning_logs/myvoice/lightning_logs/version_0/checkpoints
cp /content/drive/MyDrive/piper-voice-clone/checkpoints/myvoice/latest.ckpt \
   /content/piper-work/lightning_logs/myvoice/lightning_logs/version_0/checkpoints/last.ckpt
```

Or start fresh from the pretrained checkpoint with `colab/download_checkpoint.sh`.
