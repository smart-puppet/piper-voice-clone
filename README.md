# Voice Cloning + Piper TTS

Clone a voice with [Qwen3-TTS](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF), build a Piper-compatible dataset, and fine-tune a Piper model for fast local inference.

**Pipeline:** reference audio → Qwen3-TTS dataset → Piper fine-tune → ONNX export → `piper` CLI

## Quick start

| Step | What |
|------|------|
| 1 | Pick an install script for your hardware |
| 2 | Add reference WAV + transcript (`references/`, or upload in Colab notebook) |
| 3 | Edit `config.yaml` |
| 4 | Generate dataset → fine-tune → export ONNX |

### Install (choose one)

```bash
cd piper-voice-clone
chmod +x install/*.sh scripts/*.sh colab/*.sh

# Pascal / Quadro P2000 (sm_61) — PyTorch cu126
./install/install_old_gpu.sh

# Ampere, Ada, Blackwell — PyTorch cu130
./install/install_new_gpu.sh

# Google Colab — run in a Colab terminal after GPU runtime (or use colab/piper_voice_clone.ipynb)
bash install/install_colab.sh
```

Activate the environment (local installs):

```bash
source .venv/bin/activate
```

## Dataset naming

Generated WAV datasets use **language + version** in the directory name:

```
datasets/{voice_name}-{language_code}_{dataset_version}/
```

Examples:

| Config | Output directory |
|--------|------------------|
| `voice_name: myvoice`, `language_code: de`, `dataset_version: v03` | `datasets/myvoice-de_v03/` |
| `voice_name: narrator`, `language_code: en`, `dataset_version: v03` | `datasets/narrator-en_v03/` |

Set `language_code` and `dataset_version` in `config.yaml`. `espeak_voice` defaults to `language_code` for Piper training.

Layout inside each dataset:

```
datasets/myvoice-de_v03/
├── metadata_train.csv
├── metadata_val.csv
├── config.json          # written during Piper training
└── wavs/
    └── <md5>.wav
```

## Import a dataset zip

If you already have a dataset archive (from Drive, a teammate, or a private Hugging Face repo), import it into the correct layout:

**Local GPU**

```bash
# Hugging Face private dataset repo (uses HF_TOKEN or huggingface-cli login)
bash scripts/download_dataset.sh \
  --hf-repo your-org/piper-datasets \
  --file archives/myvoice-de_v01.zip

# Local zip file
bash scripts/download_dataset.sh --zip ~/Downloads/myvoice-de_v01.zip

# Direct URL (GitHub private release, etc.)
export HF_TOKEN=hf_...   # or pass --token
bash scripts/download_dataset.sh --url "https://..." --name myvoice-de_v01
```

**Colab** (extracts to `/content/piper-work/datasets/`)

```bash
bash colab/download_dataset.sh \
  --config config.colab.yaml \
  --zip /content/drive/MyDrive/piper-voice-clone/archives/myvoice-de_v01.zip
```

Use `--config config.yaml` (or `config.colab.yaml`) to infer the target folder from `voice_name`, `language_code`, and `dataset_version`.

**Archive types**

- **Full dataset** — metadata CSV files + `wavs/` (+ optional `cache/`)
- **Partial WAV** — only missing `.wav` files; merges into an existing dataset
- **Partial cache** — Piper training cache (`.spec.pt`, `.phonemes.pt`, …) under `cache/`; merges into an existing dataset
- **Partial mixed** — WAV + cache in one zip

Partial archives require metadata already present locally. Skips files that already exist; use `--force` to overwrite.

## Checkpoints

Download a pretrained Piper checkpoint from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints) before fine-tuning. The checkpoint language **must match** your dataset `language_code`.

**German (`de`):** `de-thorsten-emotional-medium` — expressive Thorsten (recommended for narrative / fairy-tale datasets). Neutral alternative: `de-thorsten-medium`.

**French (`fr`):** `fr-siwis-medium` (neutral) or `fr-tom-medium` (warmer tone). There is no French emotional checkpoint like Thorsten emotional; Tom is the closest option.

```bash
# German emotional Thorsten (recommended for de)
bash scripts/download_checkpoint.sh --preset de-thorsten-emotional-medium

# Custom path inside the HF repo
bash scripts/download_checkpoint.sh --path de/de_DE/thorsten/medium/epoch=3135-step=2702056.ckpt

# Browse presets and files
bash scripts/download_checkpoint.sh --list-presets
bash scripts/download_checkpoint.sh --list --lang de
```

Default output: `checkpoints/pretrained-finetune.ckpt` (weights-only, for `--model.pretrained_ckpt`). On Colab the notebook download cell writes `/content/piper-work/checkpoints/pretrained-finetune.ckpt`.

| Preset | Language | Voice |
|--------|----------|-------|
| `de-thorsten-emotional-medium` | German | Thorsten emotional (recommended) |
| `de-thorsten-medium` | German | Thorsten neutral |
| `en-lessac-medium` | English | Lessac |
| `en-amy-medium` | English | Amy |
| `fr-siwis-medium` | French | Siwis (neutral) |
| `fr-tom-medium` | French | Tom (warmer) |

All built-in presets are **medium** quality. On Colab, pick `CHECKPOINT_PRESET` in the notebook, or choose **`custom`** and set `CUSTOM_CHECKPOINT_HF_PATH`.

### Book presets (dataset text)

| Preset | Language | Source |
|--------|----------|--------|
| `de-andersen-maerchen` | German | Andersen, *Märchen* |
| `en-grimm-fairy-tales` | English | Grimm, *Household Tales* |
| `fr-perrault-contes` | French | Perrault, *Contes* |

On Colab, pick a preset or choose **`custom`** and set `CUSTOM_BOOK_URL` to any Project Gutenberg plain-text URL.

## Workflow

### 1. Test voice cloning

```bash
python scripts/test_voice.py --config config.yaml
ffplay -nodisp -autoexit test_output.wav
```

### 2. Generate dataset

```bash
python scripts/generate_dataset.py --config config.yaml --samples 10   # smoke test
python scripts/generate_dataset.py --config config.yaml              # full run
```

Generation resumes automatically if interrupted (`resume: true` in config).

### 3. Fine-tune Piper (local)

Piper loads `metadata.csv` (all utterances) and splits train/validation/test internally. With only `--samples 10`, validation may be empty — generate **at least 50–100 utterances** for meaningful fine-tuning.

```bash
bash scripts/train_piper_local.sh --voice myvoice --lang de --version v03 --smoke
bash scripts/train_piper_local.sh --voice myvoice --lang de --version v03 --epochs 10
```

### 4. Export and test ONNX

```bash
python -c "
from pathlib import Path
from voice_cloning.piper_training import export_onnx_models
export_onnx_models(
    checkpoint=Path('lightning_logs/version_0/checkpoints/last.ckpt'),
    output_onnx=Path('models/myvoice-de-medium.onnx'),
)
"

cp datasets/myvoice-de_v03/config.json models/myvoice-de-medium.onnx.json

python scripts/test_piper_voice.py \
  --model models/myvoice-de-medium.onnx \
  --config models/myvoice-de-medium.onnx.json
```

Exports `models/myvoice-de-medium.onnx` (audio) and `models/myvoice-de-medium.phoneme_duration.onnx` (audio + phoneme durations).

## Google Colab

**Recommended:** open **[`colab/piper_voice_clone.ipynb`](colab/piper_voice_clone.ipynb)** in a GPU runtime and run cells **0–11** in order (all steps in the notebook with live output).

- **GPU:** L4 on Colab Pro (best value); T4 works, A100 is fastest but costly.
- **Dataset:** at least **2,000 samples** recommended (`total_entries` default: 2,500).
- **Training (≈2,000 samples on L4):** epoch 1 ~30 min, then ~1 min/epoch.

[Open in Colab](https://colab.research.google.com/github/smart-puppet/piper-voice-clone/blob/main/colab/piper_voice_clone.ipynb?v=4)

If Colab shows an older notebook, see [colab/README.md](colab/README.md#still-seeing-the-old-notebook).

See [`colab/README.md`](colab/README.md).

## Configuration

Key fields in `config.yaml`:

| Key | Description |
|-----|-------------|
| `voice_name` | Speaker / dataset prefix |
| `language_code` | Short code in dataset dir (`de`, `en`, …) |
| `dataset_version` | Format version suffix (`v03`) |
| `espeak_voice` | espeak-ng voice for Piper (defaults to `language_code`) |
| `tts.backend` | `qwentts` (local GGUF) or `faster` (Colab / PyTorch) |
| `reference` | Reference WAV + transcript for voice cloning |
| `tts.temperature` | Optional sampling temperature |
| `book_source` | Project Gutenberg plain-text URL for fairy-tale / children's book text (see `config.yaml` comments) |
| `resume` | Skip already-generated WAV files |

## Project layout

```
piper-voice-clone/
├── install/
│   ├── install_old_gpu.sh      # Pascal / sm_61
│   ├── install_new_gpu.sh      # Ampere+
│   └── install_colab.sh        # Colab GPU runtime
├── src/voice_cloning/          # Python package
├── scripts/                    # CLI tools
├── colab/                      # Terminal scripts + minimal notebook
├── references/                 # Your reference audio
├── config.yaml
├── checkpoints/                # Upload pretrained .ckpt here
├── datasets/                   # Generated (gitignored)
├── lightning_logs/             # Training runs (gitignored)
└── models/                     # Exported ONNX (gitignored)
```

## Hardware notes

| GPU | Install script | PyTorch |
|-----|----------------|---------|
| Pascal (Quadro P2000, sm_61) | `install_old_gpu.sh` | cu126 |
| Ampere / Ada / newer | `install_new_gpu.sh` | cu130 |
| Google Colab T4/L4/A100 | `install_colab.sh` | Colab runtime |

- **Dataset generation:** GPU recommended; full runs can take hours. Use `--samples 10` first.
- **Piper training:** CUDA required. On 5–8 GB VRAM, use `batch_size` 2–4.

## TTS backends

| Backend | Best for | Install |
|---------|----------|---------|
| `qwentts` | Local machine with `qwentts.cpp` | Included in GPU install scripts |
| `faster` | Colab, PyTorch CUDA | `pip install -e .[faster]` or Colab install |

## License

This project bundles workflows around [piper1-gpl](https://github.com/OHF-voice/piper1-gpl), [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), and Qwen3-TTS models. Check each upstream project for license terms.
