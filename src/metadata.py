import pandas as pd
from src.config import FMA_METADATA_DIR


def load_tracks(metadata_dir=FMA_METADATA_DIR):
    """Load FMA tracks.csv with multi-index columns."""
    path = metadata_dir / "tracks.csv"
    return pd.read_csv(path, index_col=0, header=[0, 1])


def load_genres(metadata_dir=FMA_METADATA_DIR):
    """Load FMA genres.csv."""
    path = metadata_dir / "genres.csv"
    return pd.read_csv(path, index_col=0)


def get_small_subset_ids(tracks) -> list:
    """Return track IDs belonging to the 'small' FMA subset."""
    mask = tracks[("set", "subset")] == "small"
    return tracks[mask].index.tolist()
