# Spectrogram View — Setup Guide for MJ

## What This Is

A new retrieval view that represents each track as a **mel spectrogram** image, then extracts a 512-d embedding using a pretrained CNN (ResNet18). This gives us a visual/spectral representation of audio that's different from what OpenL3 and CLAP capture.

```
audio (.mp3) → mel spectrogram (224x224 image) → ResNet18 (no classification head) → 512-d embedding
```

## Files

| File | What it does | Where to put it |
|---|---|---|
| `spectrogram.py` | The embedding generator class | `src/embeddings/spectrogram.py` |
| `generate_spectrogram_embeddings.py` | Script to run it on the 2,000 tracks | `scripts/generate_spectrogram_embeddings.py` |

## Step-by-Step

### 1. Copy the files into the project

```bash
cp forMj/spectrogram.py src/embeddings/spectrogram.py
cp forMj/generate_spectrogram_embeddings.py scripts/generate_spectrogram_embeddings.py
```

### 2. Install dependencies (if not already installed)

Everything should already be in `requirements.txt`, but double check:

```bash
pip install librosa torchvision Pillow
```

### 3. Generate the embeddings

Run from the project root (`Finals/`):

```bash
python scripts/generate_spectrogram_embeddings.py
```

If you run out of memory, lower the batch size:

```bash
python scripts/generate_spectrogram_embeddings.py --batch-size 8
```

This will:
- Load the canonical 2,000 track IDs from `data/processed/openl3_track_ids.npy`
- Convert each track's audio to a mel spectrogram
- Pass it through ResNet18 to get a 512-d embedding
- Save to `data/processed/spectrogram_embeddings.npy` and `spectrogram_track_ids.npy`

Expect it to take ~5-10 minutes on a laptop (no GPU needed, ResNet18 is small).

### 4. Add to the app

In `app.py`, add one line to the `VIEWS` dict (~line 86):

```python
VIEWS = {
    "sbert":       load_view("SBERT (Text)",        PROCESSED_DIR / "sbert_embeddings.npy",       PROCESSED_DIR / "sbert_track_ids.npy"),
    "openl3":      load_view("OpenL3 (Audio)",       PROCESSED_DIR / "openl3_embeddings.npy",      PROCESSED_DIR / "openl3_track_ids.npy"),
    "clap":        load_view("CLAP (Vibe)",          PROCESSED_DIR / "clap_embeddings.npy",        PROCESSED_DIR / "clap_track_ids.npy"),
    "spectrogram": load_view("Spectrogram (Visual)", PROCESSED_DIR / "spectrogram_embeddings.npy", PROCESSED_DIR / "spectrogram_track_ids.npy"),
}
```

That's it — `load_view` handles mean-centering and normalization automatically, and the new view will show up in per-view recommendations and be included in RRF fusion.

### 5. Run the evaluation

Copy your `.npy` files into the `evaluation/` directory and add a loader in `evaluate_genre_retrieval.py`:

```python
# Add to the models list in main()
load_id_embedding_pair(
    root / "Spectrogram" / "spectrogram_embeddings.npy",
    root / "Spectrogram" / "spectrogram_track_ids.npy",
    "Spectrogram",
),
```

Then run:

```bash
cd evaluation
python3 evaluate_genre_retrieval.py --root . --num-samples -1 --seed 42
```

This gives you a top-1 genre accuracy score to compare directly against CLAP (53.6%), OpenL3 (55.3%), and SBERT (57.6%).

## Things You Can Experiment With

### Backbone swap (most impactful)

The current code uses ResNet18 pretrained on **ImageNet** (photos of cats, cars, etc. — not audio). Swapping to a backbone pretrained on audio spectrograms should significantly improve results. Options:

- **PANNs (Pretrained Audio Neural Networks):** CNN14 is the most popular. Trained on AudioSet (2M audio clips). Drop-in replacement — just load the model weights and remove the classification head.
- **AST (Audio Spectrogram Transformer):** A ViT fine-tuned on AudioSet. Higher accuracy than CNNs but slower. Produces 768-d embeddings.
- **VGGish:** Google's audio CNN. Older but simple and well-documented. Produces 128-d embeddings.

To swap, edit `build_backbone()` in `spectrogram.py` and update `self.dim` in `__init__`.

### Spectrogram parameters

The current settings in `spectrogram.py` are reasonable defaults, but you can tune:

| Parameter | Current | What it controls | Try |
|---|---|---|---|
| `N_MELS` | 128 | Frequency resolution (number of mel bands) | 64 or 256 |
| `HOP_LENGTH` | 512 | Time resolution (smaller = more time frames) | 256 for finer time detail |
| `DURATION_S` | 10 | How much of each track to use | 30 for more context |
| `N_FFT` | 2048 | FFT window size (frequency vs time tradeoff) | 1024 for more time resolution |

### Data augmentation (if fine-tuning)

If you decide to fine-tune the backbone on FMA data, consider:
- **SpecAugment:** Randomly mask frequency bands and time steps in the spectrogram. Standard augmentation for audio models.
- **Time shifting:** Randomly offset the start position of the 10s clip.
- **Pitch shifting / time stretching:** Via `librosa.effects`.

## How It Compares to Existing Views

| | Spectrogram (this) | OpenL3 | CLAP |
|---|---|---|---|
| Input | Audio → image | Audio | Audio |
| What it sees | Visual patterns in frequency/time | Learned acoustic features | High-level audio semantics |
| Pretrained on | ImageNet (photos) | AudioSet + ImageNet | Audio-text pairs (LAION) |
| Dimension | 512 | 512 | 512 |
| Strength | Captures visual spectral structure (e.g., harmonic patterns, drum hits) | Timbre, rhythm, texture | Mood, genre feel, instruments |
| Weakness | ImageNet features aren't ideal for audio — swap backbone for best results | No text understanding | Lower on acoustic similarity |

## Questions?

Ping in the group chat or check the `README.md` in the project root — the "Adding a New Model" section has the full 6-step workflow.
