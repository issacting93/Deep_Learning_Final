# Spectrogram View

**Audio-to-spectrogram embeddings via ResNet18 (ImageNet-pretrained) — 512-d per track.**

## Pipeline

```
audio (.mp3) → mel spectrogram (128 bands, 224x224 image) → ResNet18 (no classification head) → 512-d embedding
```

Source: `src/embeddings/spectrogram.py`
Generation script: `scripts/generate_spectrogram_embeddings.py`

## How It Compares to Other Views

| | Spectrogram | OpenL3 | CLAP |
|---|---|---|---|
| Input | Audio → image | Audio | Audio |
| What it sees | Visual patterns in frequency/time | Learned acoustic features | High-level audio semantics |
| Pretrained on | ImageNet (photos) | AudioSet + ImageNet | Audio-text pairs (LAION) |
| Dimension | 512 | 512 | 512 |
| Strength | Captures visual spectral structure (harmonic patterns, drum hits) | Timbre, rhythm, texture | Mood, genre feel, instruments |
| Weakness | ImageNet features aren't ideal for audio — swap backbone for best results | No text understanding | Lower on acoustic similarity |

## Spectrogram Parameters

| Parameter | Current | What it controls | Alternatives |
|---|---|---|---|
| `N_MELS` | 128 | Frequency resolution (number of mel bands) | 64 or 256 |
| `HOP_LENGTH` | 512 | Time resolution (smaller = more time frames) | 256 for finer time detail |
| `DURATION_S` | 10 | How much of each track to use | 30 for more context |
| `N_FFT` | 2048 | FFT window size (frequency vs time tradeoff) | 1024 for more time resolution |

## Potential Backbone Swaps

The current code uses ResNet18 pretrained on ImageNet (photos, not audio). Swapping to an audio-pretrained backbone should significantly improve results:

- **PANNs (Pretrained Audio Neural Networks):** CNN14, trained on AudioSet (2M audio clips). Drop-in replacement.
- **AST (Audio Spectrogram Transformer):** ViT fine-tuned on AudioSet. Higher accuracy but slower. 768-d embeddings.
- **VGGish:** Google's audio CNN. Older but simple. 128-d embeddings.

To swap, edit `build_backbone()` in `src/embeddings/spectrogram.py` and update `self.dim` in `__init__`.

## Data Augmentation (If Fine-Tuning)

- **SpecAugment:** Randomly mask frequency bands and time steps
- **Time shifting:** Randomly offset the start position of the 10s clip
- **Pitch shifting / time stretching:** Via `librosa.effects`

## Visuals

Presentation visuals (per-genre spectrograms, PCA/t-SNE plots, cross-view overlap) are in `reports/spectrogram/`.
