import pandas as pd
import requests
import time
import json
import sys
from pathlib import Path

def main():
    print("Loading FMA metadata...")
    df = pd.read_csv('data/fma_metadata/tracks.csv', header=[0,1], low_memory=False)
    small = df[df[('set', 'subset')] == 'small']
    
    # We take 1000 random tracks as requested
    sample = small.sample(1000, random_state=42)
    
    output_path = Path('data/processed/fma_to_mbid.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing maps if any
    mbid_map = {}
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                mbid_map = json.load(f)
            print(f"Loaded {len(mbid_map)} existing mappings.")
        except Exception:
            pass

    hits = 0
    total = 0
    print("Starting mapping for 1000 tracks...")
    
    try:
        for track_id, row in sample.iterrows():
            track_id_str = str(track_id)
            if track_id_str in mbid_map:
                total += 1
                if mbid_map[track_id_str]:
                    hits += 1
                continue
            
            title = str(row[('track', 'title')]).replace('\"', '')
            artist = str(row[('artist', 'name')]).replace('\"', '')
            
            q = f'recording:\"{title}\" AND artist:\"{artist}\"'
            url = 'https://musicbrainz.org/ws/2/recording/'
            headers = {'User-Agent': 'FMATest_1000/1.0'}
            
            matched_mbid = None
            try:
                r = requests.get(url, params={'query': q, 'fmt': 'json', 'limit': 1}, headers=headers)
                if r.status_code == 200:
                    recs = r.json().get('recordings', [])
                    if recs and int(recs[0].get('score', 0)) >= 80:
                        matched_mbid = recs[0]['id']
                        hits += 1
            except Exception as e:
                pass
            
            mbid_map[track_id_str] = matched_mbid
            total += 1
            
            # Checkpoint every 50 requests
            if total % 50 == 0:
                with open(output_path, 'w') as f:
                    json.dump(mbid_map, f, indent=2)
                sys.stdout.write(f"\rProgress: {total}/1000 | Matches: {hits}")
                sys.stdout.flush()
                
            time.sleep(1.1)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        with open(output_path, 'w') as f:
            json.dump(mbid_map, f, indent=2)
        print(f"\nFinal Progress: {total}/1000 | Matches: {hits}")

if __name__ == '__main__':
    main()
