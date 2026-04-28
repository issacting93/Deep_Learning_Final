import pandas as pd
from lyricsgenius import Genius
import time
import re
import os
from src.config import FMA_METADATA_DIR, PROCESSED_DIR
from src.metadata import load_tracks, get_small_subset_ids


def fetch_lyrics_for_fma(api_token, output_path=PROCESSED_DIR / "lyrics.csv"):
    """Fetches lyrics for FMA small subset tracks using Genius API."""
    print("Loading tracks...")
    tracks = load_tracks(FMA_METADATA_DIR)
    small_ids = get_small_subset_ids(tracks)
    df = tracks.loc[small_ids].copy()

    genius = Genius(api_token)
    genius.verbose = False
    genius.remove_section_headers = True

    lyrics_list = []
    print(f"Fetching lyrics for {len(df)} tracks...")

    for tid, row in df.iterrows():
        artist = row[("artist", "name")]
        title = row[("track", "title")]

        # Simple fuzzy: remove feat., remastered, etc.
        title_clean = re.sub(r"\(.*?\)|\[.*?\]|feat\..*", "", str(title)).strip()

        try:
            song = genius.search_song(title_clean, artist)
            if song:
                lyrics_list.append({"track_id": tid, "lyrics": song.lyrics})
            else:
                lyrics_list.append({"track_id": tid, "lyrics": ""})
            time.sleep(0.1)  # respect rate limit
        except Exception as e:
            print(f"Error {artist} - {title}: {e}")
            lyrics_list.append({"track_id": tid, "lyrics": ""})

    pd.DataFrame(lyrics_list).to_csv(output_path, index=False)
    print(f"Saved lyrics to {output_path}")


if __name__ == "__main__":
    # This requires an API token. In a real scenario, use environment variables.
    api_token = os.getenv("GENIUS_API_TOKEN")
    if api_token:
        fetch_lyrics_for_fma(api_token)
    else:
        print("GENIUS_API_TOKEN not found in environment. Skipping.")
