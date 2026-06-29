# Reference audio

Place your voice-cloning reference clips here. Each clip needs:

- A **mono WAV** file, **3–15 seconds** of clean speech
- An **exact transcript** in a `.txt` file with the same base name (or use `ref_text_file` in `config.yaml`)

Example:

```
references/
├── reference.wav      # your voice clip (not in git — upload in Colab or add locally)
└── reference.txt      # exact transcript
```

Point `reference` in `config.yaml` at your files. On Colab, the notebook uploads these for you.
