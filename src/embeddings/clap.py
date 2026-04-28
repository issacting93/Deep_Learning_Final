import json
import logging
from pathlib import Path

import numpy as np
import laion_clap

from src.config import PROCESSED_DIR, CLAP_BATCH_SIZE
from src.audio_utils import get_audio_path, discover_valid_tracks
from src.embeddings.base import EmbeddingGenerator

logger = logging.getLogger(__name__)


class CLAPEmbeddingGenerator(EmbeddingGenerator):
    """Generate CLAP audio embeddings using LAION-CLAP."""

    def __init__(self, enable_fusion: bool = False):
        logger.info("Loading LAION-CLAP model...")
        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion, amodel="HTSAT-tiny"
        )
        self.model.load_ckpt()
        logger.info("CLAP model ready.")

    def generate(
        self,
        track_ids: list,
        output_dir: Path = PROCESSED_DIR,
        batch_size: int = CLAP_BATCH_SIZE,
        resume: bool = True,
    ) -> tuple:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_dir = output_dir / "clap_batches"
        batch_dir.mkdir(exist_ok=True)
        progress_file = output_dir / "clap_progress.json"

        # Filter to tracks with audio on disk
        valid_ids = discover_valid_tracks(track_ids)
        logger.info(f"{len(valid_ids)}/{len(track_ids)} tracks have audio files.")

        # Resume: load completed batches
        completed_batches = set()
        if resume and progress_file.exists():
            progress = json.loads(progress_file.read_text())
            completed_batches = set(progress.get("completed", []))
            logger.info(f"Resuming: {len(completed_batches)} batches already done.")

        # Process in batches
        batches = [
            valid_ids[i : i + batch_size] for i in range(0, len(valid_ids), batch_size)
        ]

        for batch_idx, batch_ids in enumerate(batches):
            if batch_idx in completed_batches:
                continue

            file_paths = [str(get_audio_path(tid)) for tid in batch_ids]

            try:
                embeddings = self.model.get_audio_embedding_from_filelist(
                    x=file_paths, use_tensor=False
                )
            except Exception:
                # Fallback: process one at a time, skip corrupt files
                embeddings, batch_ids = self._process_single(file_paths, batch_ids)
                if len(batch_ids) == 0:
                    logger.warning(f"Batch {batch_idx}: all files failed, skipping.")
                    completed_batches.add(batch_idx)
                    self._save_progress(progress_file, completed_batches)
                    continue

            np.savez(
                batch_dir / f"batch_{batch_idx:04d}.npz",
                embeddings=embeddings,
                track_ids=np.array(batch_ids),
            )
            completed_batches.add(batch_idx)
            self._save_progress(progress_file, completed_batches)
            logger.info(
                f"Batch {batch_idx + 1}/{len(batches)} done ({len(batch_ids)} tracks)."
            )

        # Consolidate
        return self._consolidate(batch_dir, output_dir)

    def _process_single(self, file_paths, batch_ids):
        """Fallback: process files individually, skip failures."""
        good_embeds, good_ids = [], []
        for path, tid in zip(file_paths, batch_ids):
            try:
                emb = self.model.get_audio_embedding_from_filelist(
                    x=[path], use_tensor=False
                )
                good_embeds.append(emb)
                good_ids.append(tid)
            except Exception as e:
                logger.warning(f"Track {tid} failed: {e}")
        if good_embeds:
            return np.vstack(good_embeds), good_ids
        return np.array([]), []

    def _consolidate(self, batch_dir: Path, output_dir: Path) -> tuple:
        """Merge batch files into final output."""
        all_embeds, all_ids = [], []
        for f in sorted(batch_dir.glob("batch_*.npz")):
            data = np.load(f)
            all_embeds.append(data["embeddings"])
            all_ids.extend(data["track_ids"].tolist())

        embeddings = np.vstack(all_embeds)

        # Validate L2 norms — CLAP should return unit-length vectors
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=0.05):
            logger.warning(
                f"CLAP embeddings not normalized (range {norms.min():.4f}–{norms.max():.4f}), "
                "renormalizing."
            )
            embeddings = embeddings / (norms[:, np.newaxis] + 1e-8)

        np.save(output_dir / "clap_embeddings.npy", embeddings)
        np.save(output_dir / "clap_track_ids.npy", np.array(all_ids))
        logger.info(f"Consolidated {len(all_ids)} embeddings -> {output_dir}")
        return embeddings, all_ids

    def load_embeddings(self, output_dir: Path = PROCESSED_DIR) -> tuple:
        output_dir = Path(output_dir)
        embeddings = np.load(output_dir / "clap_embeddings.npy")
        track_ids = np.load(output_dir / "clap_track_ids.npy").tolist()
        return embeddings, track_ids

    def embed_text(self, queries: list) -> np.ndarray:
        """Embed text queries for cross-modal retrieval."""
        return self.model.get_text_embedding(queries, use_tensor=False)

    @staticmethod
    def _save_progress(path: Path, completed: set):
        path.write_text(json.dumps({"completed": sorted(completed)}))
