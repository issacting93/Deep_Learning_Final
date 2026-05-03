import logging
import pandas as pd
from lyricsgenius import Genius
import time
import re
import os
from src.config import FMA_METADATA_DIR, PROCESSED_DIR
from src.metadata import load_tracks, get_small_subset_ids

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 0.5  # seconds


def _fetch_with_backoff(genius, title_clean, artist):
    """Fetch lyrics with exponential backoff on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            song = genius.search_song(title_clean, artist)
            if song:
                return song.lyrics
            return ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed for '{title_clean}' by {artist}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"All retries failed for '{title_clean}' by {artist}: {e}")
                return ""


def fetch_lyrics_for_fma(api_token, output_path=PROCESSED_DIR / "lyrics.csv"):
    """Fetches lyrics for FMA small subset tracks using Genius API."""
    logger.info("Loading tracks...")
    tracks = load_tracks(FMA_METADATA_DIR)
    small_ids = get_small_subset_ids(tracks)
    df = tracks.loc[small_ids].copy()

    genius = Genius(api_token)
    genius.verbose = False
    genius.remove_section_headers = True

    lyrics_list = []
    logger.info(f"Fetching lyrics for {len(df)} tracks...")

    for tid, row in df.iterrows():
        artist = row[("artist", "name")]
        title = row[("track", "title")]

        # Simple fuzzy: remove feat., remastered, etc.
        title_clean = re.sub(r"\(.*?\)|\[.*?\]|feat\..*", "", str(title)).strip()

        lyrics = _fetch_with_backoff(genius, title_clean, artist)
        lyrics_list.append({"track_id": tid, "lyrics": lyrics})
        time.sleep(0.2)  # respect Genius rate limit (5 req/s)

    pd.DataFrame(lyrics_list).to_csv(output_path, index=False)
    logger.info(f"Saved lyrics to {output_path}")


if __name__ == "__main__":
    # This requires an API token. In a real scenario, use environment variables.
    api_token = os.getenv("GENIUS_API_TOKEN")
    if api_token:
        fetch_lyrics_for_fma(api_token)
    else:
        print("GENIUS_API_TOKEN not found in environment. Skipping.")
