"""
CLAP Embeddings Generator

This script generates audio and text embeddings using the LAION CLAP model.
It maps audio tracks to their corresponding metadata text and saves the resulting 
embedding representations as `.npy` files.
"""

import argparse
import os
import ast
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import ClapModel, ClapProcessor

def parse_tags(val):
    """Safely parse string representations of lists (tags)."""
    if not val or str(val).strip() in ('[]', 'nan', ''):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        inner = str(val).strip().strip('[]').replace("'", "")
        return [t.strip() for t in inner.split(',') if t.strip()]

def build_text_from_row(row):
    """Combine track title, artist name, and genre into a single string."""
    parts = [
        str(row.get(('track', 'title'), '')),
        str(row.get(('artist', 'name'), '')),
        str(row.get(('track', 'genre_top'), ''))
    ]
    # Filter out empty or 'nan' values
    valid_parts = [p for p in parts if p and p.lower() != 'nan']
    return " | ".join(valid_parts) if valid_parts else "music"

def get_audio_path(audio_dir, tid):
    """Resolve the path to an FMA mp3 file based on its Track ID."""
    tid_str = f"{tid:06d}"
    folder = tid_str[:3]
    return os.path.join(audio_dir, folder, f"{tid_str}.mp3")

def main():
    parser = argparse.ArgumentParser(description="Generate CLAP audio and text embeddings.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to our_2000_tracks.csv")
    parser.add_argument("--audio-dir", type=str, required=True, help="Path to fma_2000_tracks directory")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the embeddings")
    parser.add_argument("--model-id", type=str, default="laion/clap-htsat-unfused", help="HuggingFace model ID")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading CLAP model ({args.model_id})...")
    model = ClapModel.from_pretrained(args.model_id).to(device)
    processor = ClapProcessor.from_pretrained(args.model_id)
    model.eval()

    # Load Metadata
    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata, index_col=0, header=[0, 1])
    track_ids = df.index.tolist()

    audio_embeddings = {}
    text_embeddings = {}

    print(f"Processing {len(track_ids)} tracks...")
    
    for i, tid in enumerate(track_ids):
        audio_path = get_audio_path(args.audio_dir, tid)
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for track {tid} ({audio_path})")
            continue

        try:
            # Load audio (first 30 seconds)
            audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=30)
            
            # Load text
            text = build_text_from_row(df.loc[tid])

            # Process inputs
            inputs = processor(
                text=[text],
                audios=[audio], # Processors in newer transformers use 'audios'
                sampling_rate=48000,
                return_tensors="pt",
                padding=True
            ).to(device)

            # Extract embeddings
            with torch.no_grad():
                audio_emb = model.get_audio_features(
                    input_features=inputs['input_features']
                )
                text_emb = model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Check for pooler_output (model architecture specific)
                if hasattr(audio_emb, 'pooler_output'):
                    audio_emb = audio_emb.pooler_output
                if hasattr(text_emb, 'pooler_output'):
                    text_emb = text_emb.pooler_output

            audio_embeddings[tid] = audio_emb.squeeze().cpu().numpy()
            text_embeddings[tid] = text_emb.squeeze().cpu().numpy()

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(track_ids)} tracks")

        except Exception as e:
            print(f"Skipped {tid} due to error: {e}")

    # Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    
    audio_out = os.path.join(args.output_dir, 'clap_audio_embeddings.npy')
    text_out = os.path.join(args.output_dir, 'clap_text_embeddings.npy')
    
    print("Saving embeddings...")
    np.save(audio_out, audio_embeddings)
    np.save(text_out, text_embeddings)
    
    print("\nDone!")
    print(f"Saved {len(audio_embeddings)} audio embeddings to {audio_out}")
    print(f"Saved {len(text_embeddings)} text embeddings to {text_out}")

if __name__ == "__main__":
    main()