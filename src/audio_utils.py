from pathlib import Path
from src.config import FMA_SMALL_DIR


def get_audio_path(track_id: int, data_dir: Path = FMA_SMALL_DIR) -> Path:
    """FMA directory structure: 000/000002.mp3"""
    tid_str = f"{track_id:06d}"
    return data_dir / tid_str[:3] / f"{tid_str}.mp3"


def discover_valid_tracks(track_ids: list, data_dir: Path = FMA_SMALL_DIR) -> list:
    """Filter track IDs to those with audio files on disk."""
    return [tid for tid in track_ids if get_audio_path(tid, data_dir).exists()]
