"""Spectrogram-based embedding generator.

Pipeline: audio → mel spectrogram → pretrained CNN → embedding vector.

Uses a ResNet18 backbone (ImageNet-pretrained) with the final classification
head removed, producing a 512-d embedding per track.
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Normalize

from src.audio_utils import get_audio_path, discover_valid_tracks
from src.config import PROCESSED_DIR
from src.embeddings.base import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Audio / spectrogram constants
SAMPLE_RATE = 22050
DURATION_S = 10
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
IMG_SIZE = 224  # ResNet input size


def audio_to_mel_spectrogram(file_path: str) -> np.ndarray:
    """Load audio and convert to a log-mel spectrogram image (1, 224, 224)."""
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION_S, mono=True)

    # Pad short clips to exactly DURATION_S seconds
    target_len = SAMPLE_RATE * DURATION_S
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # shape: (N_MELS, time_frames)

    # Resize to (IMG_SIZE, IMG_SIZE) via simple interpolation
    from PIL import Image
    img = Image.fromarray(log_mel).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Normalize to [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr  # (224, 224)


def build_backbone() -> nn.Module:
    """ResNet18 with the classification head removed → 512-d output."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove final FC layer, keep avgpool → (batch, 512)
    backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    backbone.eval()
    return backbone


class SpectrogramEmbeddingGenerator(EmbeddingGenerator):
    """Generate embeddings from mel spectrograms via a pretrained CNN."""

    def __init__(self):
        logger.info("Loading spectrogram backbone (ResNet18)...")
        self.backbone = build_backbone()
        # ImageNet normalization (applied to each of the 3 repeated channels)
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.dim = 512
        logger.info("Spectrogram backbone ready.")

    def generate(
        self,
        track_ids: list,
        output_dir: Path = PROCESSED_DIR,
        batch_size: int = 32,
        resume: bool = True,
    ) -> tuple:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_ids = discover_valid_tracks(track_ids)
        logger.info(f"{len(valid_ids)}/{len(track_ids)} tracks have audio files.")

        all_embeddings = []
        all_ids = []
        batches = [valid_ids[i:i + batch_size] for i in range(0, len(valid_ids), batch_size)]

        for batch_idx, batch_ids in enumerate(batches):
            specs = []
            good_ids = []
            for tid in batch_ids:
                try:
                    spec = audio_to_mel_spectrogram(str(get_audio_path(tid)))
                    specs.append(spec)
                    good_ids.append(tid)
                except Exception as e:
                    logger.warning(f"Track {tid} failed: {e}")

            if not specs:
                continue

            # Stack into tensor: (batch, 1, 224, 224) → repeat to 3 channels for ResNet
            batch_arr = np.stack(specs)[:, np.newaxis, :, :]  # (B, 1, 224, 224)
            batch_tensor = torch.from_numpy(np.repeat(batch_arr, 3, axis=1))  # (B, 3, 224, 224)

            # Apply ImageNet normalization
            batch_tensor = torch.stack([self.normalize(t) for t in batch_tensor])

            with torch.no_grad():
                emb = self.backbone(batch_tensor).numpy()  # (B, 512)

            all_embeddings.append(emb)
            all_ids.extend(good_ids)
            logger.info(f"Batch {batch_idx + 1}/{len(batches)} done ({len(good_ids)} tracks).")

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms

        np.save(output_dir / "spectrogram_embeddings.npy", embeddings)
        np.save(output_dir / "spectrogram_track_ids.npy", np.array(all_ids))
        logger.info(f"Saved {len(all_ids)} spectrogram embeddings → {output_dir}")
        return embeddings, all_ids

    def load_embeddings(self, output_dir: Path = PROCESSED_DIR) -> tuple:
        output_dir = Path(output_dir)
        embeddings = np.load(output_dir / "spectrogram_embeddings.npy")
        track_ids = np.load(output_dir / "spectrogram_track_ids.npy").tolist()
        return embeddings, track_ids
