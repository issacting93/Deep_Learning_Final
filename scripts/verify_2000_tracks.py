#!/usr/bin/env python3
"""Verify that the 2,000-track canonical subset is consistent across all data files."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LYRICS_ENRICHED = PROCESSED / "lyrics_enriched"
METADATA = ROOT / "data" / "fma_2000_metadata"
EVAL_SBERT = ROOT / "evaluation" / "SBERT"

CANONICAL_COUNT = 2000
passed = 0
failed = 0
warnings = 0


def ok(msg):
    global passed
    passed += 1
    print(f"  ✓ {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  ✗ {msg}")


def warn(msg):
    global warnings
    warnings += 1
    print(f"  ⚠ {msg}")


# ── 1. Canonical track IDs (source of truth) ────────────────────────────────
print("\n1. Canonical track IDs")
canonical_path = PROCESSED / "openl3_track_ids.npy"
if not canonical_path.exists():
    fail(f"Missing: {canonical_path.relative_to(ROOT)}")
    print("\nCannot continue without canonical IDs.")
    sys.exit(1)

canonical_ids = np.load(canonical_path).astype(int)
canonical_set = set(canonical_ids.tolist())

if len(canonical_ids) == CANONICAL_COUNT:
    ok(f"openl3_track_ids.npy has {len(canonical_ids)} IDs")
else:
    fail(f"openl3_track_ids.npy has {len(canonical_ids)} IDs (expected {CANONICAL_COUNT})")

if len(canonical_set) == len(canonical_ids):
    ok("All canonical IDs are unique")
else:
    fail(f"{len(canonical_ids) - len(canonical_set)} duplicate IDs")

# ── 2. Embedding files ──────────────────────────────────────────────────────
print("\n2. Embedding files (shape and ID alignment)")

embedding_pairs = [
    ("OpenL3", PROCESSED / "openl3_embeddings.npy", PROCESSED / "openl3_track_ids.npy", 512, True),
    ("SBERT", PROCESSED / "sbert_embeddings.npy", PROCESSED / "sbert_track_ids.npy", 384, True),
    ("CLAP", PROCESSED / "clap_embeddings.npy", PROCESSED / "clap_track_ids.npy", 512, False),
    ("SBERT-lyrics", LYRICS_ENRICHED / "sbert_lyrics_embeddings.npy", None, 384, True),
    ("Fused", LYRICS_ENRICHED / "fused_embeddings.npy", LYRICS_ENRICHED / "fused_track_ids.npy", None, True),
]

for name, emb_path, ids_path, expected_dim, expect_2000 in embedding_pairs:
    if not emb_path.exists():
        warn(f"{name}: {emb_path.relative_to(ROOT)} not found — skipped")
        continue

    emb = np.load(emb_path)
    n, d = emb.shape

    # Check count
    if expect_2000:
        if n == CANONICAL_COUNT:
            ok(f"{name} embeddings: ({n}, {d})")
        else:
            fail(f"{name} embeddings: ({n}, {d}) — expected {CANONICAL_COUNT} rows")
    else:
        ok(f"{name} embeddings: ({n}, {d}) (full FMA small, not restricted to 2000)")

    # Check dimension
    if expected_dim and d != expected_dim:
        fail(f"{name} dimension {d} ≠ expected {expected_dim}")

    # Check finite values
    if not np.isfinite(emb).all():
        fail(f"{name} contains NaN/Inf values")

    # Check ID alignment
    if ids_path and ids_path.exists():
        ids = np.load(ids_path).astype(int)
        if ids.shape[0] != n:
            fail(f"{name} ID count ({ids.shape[0]}) ≠ embedding rows ({n})")
        elif expect_2000 and set(ids.tolist()) == canonical_set:
            ok(f"{name} IDs match canonical set exactly")
        elif expect_2000:
            diff = set(ids.tolist()) - canonical_set
            missing = canonical_set - set(ids.tolist())
            if diff:
                fail(f"{name} has {len(diff)} IDs outside canonical set")
            if missing:
                fail(f"{name} is missing {len(missing)} canonical IDs")
        else:
            subset = set(ids.tolist())
            if canonical_set.issubset(subset):
                ok(f"{name} IDs are a superset of canonical ({len(subset)} ⊇ {CANONICAL_COUNT})")
            else:
                missing = canonical_set - subset
                warn(f"{name} is missing {len(missing)} canonical IDs")

# ── 3. FAISS index ID files ─────────────────────────────────────────────────
print("\n3. FAISS index ID files")

faiss_files = [
    ("CLAP FAISS", PROCESSED / "clap_faiss.ids.npy", False),
    ("SBERT FAISS", PROCESSED / "sbert_faiss.ids.npy", True),
    ("SBERT-lyrics FAISS", LYRICS_ENRICHED / "sbert_lyrics_faiss.ids.npy", True),
]

for name, path, expect_2000 in faiss_files:
    if not path.exists():
        warn(f"{name}: {path.relative_to(ROOT)} not found — skipped")
        continue
    ids = np.load(path).astype(int)
    if expect_2000:
        if len(ids) == CANONICAL_COUNT:
            ok(f"{name}: {len(ids)} IDs")
        else:
            fail(f"{name}: {len(ids)} IDs (expected {CANONICAL_COUNT})")
    else:
        ok(f"{name}: {len(ids)} IDs (full FMA small)")

# ── 4. Metadata CSVs ────────────────────────────────────────────────────────
print("\n4. Metadata CSVs")

# tracks.csv (FMA multi-level header: 3 header rows, index_col=0)
tracks_path = METADATA / "tracks.csv"
if tracks_path.exists():
    tracks = pd.read_csv(tracks_path, header=[0, 1], index_col=0, low_memory=False)
    n_tracks = len(tracks)
    if n_tracks == CANONICAL_COUNT:
        ok(f"tracks.csv: {n_tracks} rows")
    else:
        fail(f"tracks.csv: {n_tracks} rows (expected {CANONICAL_COUNT})")
    # Check IDs match canonical
    track_ids_csv = set(tracks.index.astype(int).tolist())
    if track_ids_csv == canonical_set:
        ok("tracks.csv IDs match canonical set exactly")
    else:
        extra = track_ids_csv - canonical_set
        missing = canonical_set - track_ids_csv
        if extra:
            fail(f"tracks.csv has {len(extra)} IDs outside canonical set")
        if missing:
            fail(f"tracks.csv is missing {len(missing)} canonical IDs")
else:
    warn(f"tracks.csv not found at {tracks_path.relative_to(ROOT)}")

# echonest.csv (FMA multi-level header: 3 header rows, index_col=0)
echonest_path = METADATA / "echonest.csv"
if echonest_path.exists():
    echonest = pd.read_csv(echonest_path, header=[0, 1, 2], index_col=0, low_memory=False)
    n_echo = len(echonest)
    echo_ids = set(echonest.index.astype(int).tolist())
    if echo_ids.issubset(canonical_set):
        ok(f"echonest.csv: {n_echo} rows (subset of canonical 2000)")
    else:
        extra = echo_ids - canonical_set
        fail(f"echonest.csv has {len(extra)} IDs outside canonical set")
else:
    warn(f"echonest.csv not found at {echonest_path.relative_to(ROOT)}")

# ── 5. Lyrics / SBERT CSVs ──────────────────────────────────────────────────
print("\n5. Lyrics / SBERT CSVs")

for name, path in [
    ("evaluation/SBERT/lyrics_df.csv", EVAL_SBERT / "lyrics_df.csv"),
    ("lyrics_enriched/lyrics_df.csv", LYRICS_ENRICHED / "lyrics_df.csv"),
]:
    if not path.exists():
        warn(f"{name} not found — skipped")
        continue
    df = pd.read_csv(path)
    n_rows = len(df)
    if n_rows == CANONICAL_COUNT:
        ok(f"{name}: {n_rows} rows")
    else:
        fail(f"{name}: {n_rows} rows (expected {CANONICAL_COUNT})")
    if "track_id" in df.columns:
        csv_ids = set(df["track_id"].astype(int).tolist())
        if csv_ids == canonical_set:
            ok(f"{name} track_ids match canonical set")
        elif csv_ids.issubset(canonical_set):
            warn(f"{name} has {CANONICAL_COUNT - len(csv_ids)} fewer IDs than canonical")
        else:
            extra = csv_ids - canonical_set
            fail(f"{name} has {len(extra)} IDs outside canonical set")

# ── 6. Cross-view intersection (simulates app.py startup) ───────────────────
print("\n6. Cross-view intersection (simulates app.py)")

view_ids = {}
for name, ids_path in [
    ("OpenL3", PROCESSED / "openl3_track_ids.npy"),
    ("SBERT", PROCESSED / "sbert_track_ids.npy"),
    ("CLAP", PROCESSED / "clap_track_ids.npy"),
]:
    if ids_path.exists():
        view_ids[name] = set(np.load(ids_path).astype(int).tolist())

if len(view_ids) == 3:
    common = view_ids["OpenL3"] & view_ids["SBERT"] & view_ids["CLAP"]
    if len(common) == CANONICAL_COUNT:
        ok(f"3-view intersection: {len(common)} tracks (all 2000 present)")
    elif len(common) >= CANONICAL_COUNT:
        ok(f"3-view intersection: {len(common)} tracks (≥ 2000)")
    else:
        fail(f"3-view intersection: {len(common)} tracks (expected {CANONICAL_COUNT})")
    # Show per-view sizes
    for name, ids in view_ids.items():
        print(f"    {name}: {len(ids)} tracks")
else:
    warn(f"Only {len(view_ids)}/3 views found — cannot test intersection")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")
if failed:
    print("SOME CHECKS FAILED — review output above.")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
