#!/usr/bin/env python3
"""
scripts/prepare_manifests.py

Prepares audio manifests for Multi-Task Audio Spectrogram Transformer (AST) classification.
Processes GTZAN (music domain, 10 classes) and ESC-50 (environmental domain, 50 classes).

Key features:
1. Resamples all incoming audio to 16 kHz mono.
2. GTZAN: Splits tracks at the song level FIRST (80/10/10) to prevent cross-split song leakage,
   then slices each 30s track into six non-overlapping 5.0s chunks.
   Gracefully handles and skips corrupt audio files (such as jazz.00054.wav).
3. ESC-50: Respects official 5 folds to construct balanced 80/10/10 splits across native 5.0s clips.
   Supports auto-downloading official ESC-50 release if missing.
4. Outputs:
   - data/manifests/train.csv
   - data/manifests/val.csv
   - data/manifests/test.csv
   Columns: filepath, domain, label (domain: 0 for music, 1 for environmental).
   - data/manifests/music_classes.json
   - data/manifests/env_classes.json
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ESC50_DOWNLOAD_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION)  # 80,000 samples


def download_esc50_dataset(destination_dir: Path) -> Path:
    """Downloads and extracts the official ESC-50 dataset if not present."""
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = destination_dir.parent / "esc50_temp.zip"

    logger.info(f"Downloading ESC-50 dataset from {ESC50_DOWNLOAD_URL}...")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading ESC-50") as t:
        urllib.request.urlretrieve(
            ESC50_DOWNLOAD_URL,
            filename=temp_zip,
            reporthook=t.update_to,
        )

    logger.info(f"Extracting ESC-50 archive to {destination_dir.parent}...")
    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(destination_dir.parent)

    if temp_zip.exists():
        temp_zip.unlink()

    extracted_dir = destination_dir.parent / "ESC-50-master"
    if extracted_dir != destination_dir and not destination_dir.exists():
        if extracted_dir.exists():
            return extracted_dir

    return destination_dir


def load_and_resample(
    filepath: Union[str, Path], target_sr: int = TARGET_SAMPLE_RATE
) -> Optional[np.ndarray]:
    """
    Reads an audio file, converts to mono, and resamples to target_sr.
    Returns float32 1D numpy array, or None if file is corrupted/unreadable.
    """
    filepath = Path(filepath)
    try:
        data, sr = sf.read(str(filepath), dtype="float32")
    except Exception as e:
        logger.warning(f"Skipping corrupt or unreadable audio file '{filepath.name}': {e}")
        return None

    # Convert stereo or multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rate doesn't match
    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        up = target_sr // gcd
        down = sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)

    return data


def ensure_chunk_length(chunk: np.ndarray, target_length: int = CHUNK_SAMPLES) -> np.ndarray:
    """Pads with zeros or truncates to ensure exact target sample count."""
    if len(chunk) < target_length:
        chunk = np.pad(chunk, (0, target_length - len(chunk)), mode="constant")
    elif len(chunk) > target_length:
        chunk = chunk[:target_length]
    return chunk.astype(np.float32)


def process_gtzan(
    gtzan_dir: Path,
    processed_dir: Path,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int]]:
    """
    Processes the GTZAN dataset:
    1. Splits tracks at song level FIRST (80/10/10 split) to prevent leakage.
    2. Slices each 30s track into six non-overlapping 5.0s chunks (80,000 samples each).
    3. Handles corrupt files like jazz.00054.wav via try/except.
    4. Saves processed 16 kHz mono chunks to disk.
    """
    logger.info("=" * 60)
    logger.info("Processing GTZAN dataset (Music Domain)...")
    logger.info("=" * 60)

    if not gtzan_dir.exists():
        raise FileNotFoundError(f"GTZAN directory not found: {gtzan_dir}")

    # Identify all genres (subdirectories)
    genre_dirs = sorted([d for d in gtzan_dir.iterdir() if d.is_dir()])
    if not genre_dirs:
        raise ValueError(f"No genre subdirectories found in {gtzan_dir}")

    music_classes = {d.name: idx for idx, d in enumerate(genre_dirs)}
    logger.info(f"Found {len(music_classes)} genres: {list(music_classes.keys())}")

    rng = random.Random(seed)
    train_records = []
    val_records = []
    test_records = []
    total_skipped = 0

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        label_id = music_classes[genre]
        wav_files = sorted(list(genre_dir.glob("*.wav")))

        # Deterministic shuffle at song level
        song_list = list(wav_files)
        rng.shuffle(song_list)

        n_total = len(song_list)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)

        split_assignment = {
            "train": song_list[:n_train],
            "val": song_list[n_train : n_train + n_val],
            "test": song_list[n_train + n_val :],
        }

        logger.info(
            f"Genre [{genre:<10}]: {len(split_assignment['train'])} train songs, "
            f"{len(split_assignment['val'])} val songs, {len(split_assignment['test'])} test songs"
        )

        for split_name, songs in split_assignment.items():
            records_list = (
                train_records
                if split_name == "train"
                else (val_records if split_name == "val" else test_records)
            )

            out_genre_dir = processed_dir / "gtzan" / split_name / genre
            out_genre_dir.mkdir(parents=True, exist_ok=True)

            for song_path in songs:
                audio = load_and_resample(song_path, TARGET_SAMPLE_RATE)
                if audio is None:
                    total_skipped += 1
                    continue

                # Slice into six non-overlapping 5.0s chunks
                # Chunk 0: 0-5s, Chunk 1: 5-10s, ..., Chunk 5: 25-30s
                for chunk_idx in range(6):
                    start_sample = chunk_idx * CHUNK_SAMPLES
                    end_sample = start_sample + CHUNK_SAMPLES
                    raw_chunk = audio[start_sample:end_sample]
                    chunk = ensure_chunk_length(raw_chunk, CHUNK_SAMPLES)

                    chunk_filename = f"{song_path.stem}_chunk{chunk_idx}.wav"
                    chunk_path = out_genre_dir / chunk_filename
                    sf.write(str(chunk_path), chunk, TARGET_SAMPLE_RATE)

                    # Store relative path for portability
                    rel_path = chunk_path.as_posix()
                    records_list.append({
                        "filepath": rel_path,
                        "domain": 0,  # 0 for music
                        "label": label_id,
                    })

    logger.info(
        f"GTZAN completed: {len(train_records)} train chunks, {len(val_records)} val chunks, "
        f"{len(test_records)} test chunks. (Corrupt files skipped: {total_skipped})"
    )
    return train_records, val_records, test_records, music_classes


def process_esc50(
    esc50_dir: Path,
    processed_dir: Path,
    auto_download: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int]]:
    """
    Processes the ESC-50 dataset:
    1. Locates or downloads the official ESC-50 repository.
    2. Respects official 5 folds or constructs balanced 80/10/10 splits across native 5.0s clips:
       Folds 1, 2, 3, 4 -> Train (1,600 clips, 80%)
       Fold 5 divided evenly -> Val (200 clips, 10%), Test (200 clips, 10%)
    3. Resamples to 16 kHz mono (80,000 samples).
    4. Saves processed audio and returns records.
    """
    logger.info("=" * 60)
    logger.info("Processing ESC-50 dataset (Environmental Domain)...")
    logger.info("=" * 60)

    # Check alternative common names if specified directory not directly present
    meta_csv = esc50_dir / "meta" / "esc50.csv"
    if not meta_csv.exists():
        alt_dirs = [
            esc50_dir.parent / "ESC-50-master",
            esc50_dir.parent / "ESC-50",
            Path("Data/ESC-50-master"),
            Path("Data/ESC-50"),
        ]
        for alt in alt_dirs:
            if (alt / "meta" / "esc50.csv").exists():
                esc50_dir = alt
                meta_csv = esc50_dir / "meta" / "esc50.csv"
                break

    if not meta_csv.exists():
        if auto_download:
            esc50_dir = download_esc50_dataset(esc50_dir)
            meta_csv = esc50_dir / "meta" / "esc50.csv"
        else:
            raise FileNotFoundError(
                f"ESC-50 metadata not found at {meta_csv}. Pass --download-esc50 to download automatically."
            )

    if not meta_csv.exists():
        raise FileNotFoundError(f"ESC-50 metadata CSV could not be found at {meta_csv}")

    df_meta = pd.read_csv(meta_csv)
    audio_src_dir = esc50_dir / "audio"

    # Export mapping: category name -> integer target (0-49)
    # ESC-50 official metadata has category and target columns
    category_target_df = df_meta[["category", "target"]].drop_duplicates().sort_values("target")
    env_classes = dict(zip(category_target_df["category"], category_target_df["target"].astype(int)))
    logger.info(f"Found {len(env_classes)} environmental classes.")

    rng = random.Random(seed)
    train_records = []
    val_records = []
    test_records = []
    skipped = 0

    out_esc_dir = processed_dir / "esc50"
    out_esc_dir.mkdir(parents=True, exist_ok=True)

    # Partition:
    # Folds 1, 2, 3, 4 -> Train (4 folds * 400 clips = 1,600 clips, exactly 80%)
    # Fold 5 (400 clips) -> 200 Val (10%), 200 Test (10%), balanced across all 50 categories
    fold5_df = df_meta[df_meta["fold"] == 5]
    fold5_val_indices = set()
    fold5_test_indices = set()

    for category, group in fold5_df.groupby("category"):
        indices = list(group.index)
        rng.shuffle(indices)
        half = len(indices) // 2
        fold5_val_indices.update(indices[:half])
        fold5_test_indices.update(indices[half:])

    logger.info("Resampling and formatting ESC-50 audio clips to 16 kHz mono...")
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="ESC-50 clips"):
        filename = row["filename"]
        fold = row["fold"]
        label = int(row["target"])
        src_path = audio_src_dir / filename

        audio = load_and_resample(src_path, TARGET_SAMPLE_RATE)
        if audio is None:
            skipped += 1
            continue

        chunk = ensure_chunk_length(audio, CHUNK_SAMPLES)
        chunk_path = out_esc_dir / filename
        sf.write(str(chunk_path), chunk, TARGET_SAMPLE_RATE)

        record = {
            "filepath": chunk_path.as_posix(),
            "domain": 1,  # 1 for environmental
            "label": label,
        }

        if fold in [1, 2, 3, 4]:
            train_records.append(record)
        elif idx in fold5_val_indices:
            val_records.append(record)
        else:
            test_records.append(record)

    logger.info(
        f"ESC-50 completed: {len(train_records)} train clips, {len(val_records)} val clips, "
        f"{len(test_records)} test clips. (Skipped: {skipped})"
    )
    return train_records, val_records, test_records, env_classes


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GTZAN & ESC-50 manifests for Multi-Task AST."
    )
    parser.add_argument(
        "--gtzan-dir",
        type=Path,
        default=Path("Data/genres_original"),
        help="Path to GTZAN genres_original directory.",
    )
    parser.add_argument(
        "--esc50-dir",
        type=Path,
        default=Path("Data/ESC-50-master"),
        help="Path to ESC-50 dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for CSV manifests and class JSON files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to store resampled and sliced 16 kHz audio files.",
    )
    parser.add_argument(
        "--download-esc50",
        action="store_true",
        default=True,
        help="Automatically download ESC-50 if not found.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic song-level splitting.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process GTZAN
    gtzan_train, gtzan_val, gtzan_test, music_classes = process_gtzan(
        gtzan_dir=args.gtzan_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
    )

    # 2. Process ESC-50
    esc_train, esc_val, esc_test, env_classes = process_esc50(
        esc50_dir=args.esc50_dir,
        processed_dir=args.processed_dir,
        auto_download=args.download_esc50,
        seed=args.seed,
    )

    # 3. Combine records
    train_all = gtzan_train + esc_train
    val_all = gtzan_val + esc_val
    test_all = gtzan_test + esc_test

    # Shuffle training set so music and environmental clips are interleaved
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    # 4. Save manifest CSVs with required columns: filepath, domain, label
    columns = ["filepath", "domain", "label"]
    df_train = pd.DataFrame(train_all)[columns]
    df_val = pd.DataFrame(val_all)[columns]
    df_test = pd.DataFrame(test_all)[columns]

    train_csv = args.output_dir / "train.csv"
    val_csv = args.output_dir / "val.csv"
    test_csv = args.output_dir / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    logger.info(f"Saved manifests:")
    logger.info(f" - Train: {train_csv} ({len(df_train)} rows)")
    logger.info(f" - Val:   {val_csv} ({len(df_val)} rows)")
    logger.info(f" - Test:  {test_csv} ({len(df_test)} rows)")

    # 5. Export class mappings to JSON
    music_classes_json = args.output_dir / "music_classes.json"
    env_classes_json = args.output_dir / "env_classes.json"

    with open(music_classes_json, "w", encoding="utf-8") as f:
        json.dump(music_classes, f, indent=2)

    with open(env_classes_json, "w", encoding="utf-8") as f:
        json.dump(env_classes, f, indent=2)

    logger.info(f"Saved class mappings:")
    logger.info(f" - Music: {music_classes_json} ({len(music_classes)} classes)")
    logger.info(f" - Env:   {env_classes_json} ({len(env_classes)} classes)")
    logger.info("Manifest preparation complete!")


if __name__ == "__main__":
    main()
