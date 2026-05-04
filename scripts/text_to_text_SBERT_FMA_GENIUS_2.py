"""
SBERT Lyrics Embeddings Generator

This script generates text embeddings for music tracks by combining metadata 
and fetched lyrics from Genius.com, using the Sentence-BERT model (all-MiniLM-L6-v2).
"""

import argparse
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import lyricsgenius

def load_metadata(csv_path, max_tracks=1200):
    """Load and preprocess FMA track metadata."""
    print(f"Loading metadata from {csv_path}...")
    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    
    df = pd.DataFrame({
        'track_id': tracks.index,
        'title': tracks['track']['title'],
        'artist': tracks['artist']['name'],
        'genres': tracks['track']['genre_top'],
        'tags': tracks['track']['tags'],
    })
    
    df = df.dropna(subset=['title', 'artist'])
    if max_tracks and max_tracks > 0:
        df = df.head(max_tracks)
        
    return df

def build_metadata_string(row):
    """Create a descriptive string from track metadata."""
    genres = row['genres'] if isinstance(row['genres'], str) else ''
    tags = row['tags'] if isinstance(row['tags'], str) else ''
    return f"{row['title']} by {row['artist']}. Genres: {genres}. Tags: {tags}."

def fetch_lyrics(genius, title, artist):
    """Fetch lyrics for a given track using the Genius API."""
    print(f"Fetching: {title} - {artist}")
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            # Skip the first line which often contains the title
            lyrics = song.lyrics.split('\n', 1)[-1]
            return lyrics[:1000]
    except Exception as e:
        print(f"  Error fetching lyrics: {e}")
    return ''

def main():
    parser = argparse.ArgumentParser(description="Generate SBERT text embeddings from lyrics.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to tracks.csv")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the FAISS index and data")
    parser.add_argument("--max-tracks", type=int, default=1200, help="Max tracks to process")
    args = parser.parse_args()

    # Load environment variables (like GENIUS_API_TOKEN)
    load_dotenv()
    genius_token = os.environ.get("GENIUS_API_TOKEN")
    
    if not genius_token:
        raise ValueError("GENIUS_API_TOKEN environment variable is not set. Please add it to a .env file or export it.")

    genius = lyricsgenius.Genius(genius_token, timeout=10)
    genius.verbose = False
    genius.remove_section_headers = True

    df = load_metadata(args.metadata, max_tracks=args.max_tracks)
    
    print("Building base metadata text...")
    df['text'] = df.apply(build_metadata_string, axis=1)
    
    print("Fetching lyrics from Genius...")
    df['lyrics'] = df.apply(lambda row: fetch_lyrics(genius, row['title'], row['artist']), axis=1)

    print("Combining text and lyrics...")
    def full_text(row):
        base = row['text']
        if row['lyrics']:
            return f"{base} Lyrics: {row['lyrics']}"
        return base
        
    df['full_text'] = df.apply(full_text, axis=1)

    print("Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df['full_text'].tolist()
    
    print("Encoding texts into embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"Index built. Total vectors: {index.ntotal}")
    
    # Save output if needed
    os.makedirs(args.output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(args.output_dir, "sbert_lyrics.index"))
    np.save(os.path.join(args.output_dir, "sbert_lyrics_embeddings.npy"), embeddings)
    df.to_csv(os.path.join(args.output_dir, "sbert_lyrics_metadata.csv"), index=False)
    
    print("Done! Data saved to", args.output_dir)

if __name__ == "__main__":
    main()
