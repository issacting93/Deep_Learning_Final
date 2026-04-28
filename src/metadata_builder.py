import pandas as pd
import re
from src.config import PROCESSED_DIR, FMA_METADATA_DIR
from src.metadata import load_tracks, get_small_subset_ids

def normalize_text(s: str) -> str:
    """Lowercase, strip extra spaces and punctuation."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)  # remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_metadata_strings():
    """Constructs the text string for each track from tracks.csv."""
    print("Loading tracks metadata...")
    tracks = load_tracks(FMA_METADATA_DIR)
    
    # Get small subset IDs
    small_ids = get_small_subset_ids(tracks)
    df = tracks.loc[small_ids].copy()
    
    print(f"Processing {len(df)} tracks...")
    
    # Extract fields with multi-index awareness
    # ('artist', 'name'), ('track', 'title'), ('track', 'genre_top'), ('track', 'tags')
    
    # Genre labels to exclude from tags to prevent evaluation leakage
    genre_labels = {"electronic", "experimental", "folk", "hip-hop", "instrumental",
                    "international", "pop", "rock"}

    def extract_row_text(row):
        artist = row[('artist', 'name')] if not pd.isna(row[('artist', 'name')]) else "unknown"
        title = row[('track', 'title')] if not pd.isna(row[('track', 'title')]) else "unknown"
        tags = row[('track', 'tags')] if not pd.isna(row[('track', 'tags')]) else ""

        # In FMA, tags is often a string representation of a list like "['tag1', 'tag2']"
        if isinstance(tags, str) and tags.startswith("["):
            tags = tags.strip("[]").replace("'", "").replace('"', "")

        # Filter out genre-like words from tags to avoid evaluation leakage
        if tags:
            tokens = [t.strip() for t in tags.split(",")]
            tags = ", ".join(t for t in tokens if t.lower() not in genre_labels)

        parts = [f"{artist} - {title}."]
        if tags:
            parts.append(f"Tags: {tags}.")

        raw_string = " ".join(parts)
        return normalize_text(raw_string)

    df["metadata_text"] = df.apply(extract_row_text, axis=1)
    
    # Save for later use - we keep track_id as the index
    output_path = PROCESSED_DIR / "metadata_texts.csv"
    # Convert to series or simple DF to avoid MultiIndex header issues
    df_out = pd.DataFrame(df["metadata_text"])
    df_out.to_csv(output_path)
    print(f"Saved metadata texts to {output_path}")
    return df

if __name__ == "__main__":
    build_metadata_strings()
