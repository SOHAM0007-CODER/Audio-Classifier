"""
tests/test_dataset.py

Unit tests for Multi-Task AST Dataset and Manifest Preparation:
1. Validates manifest structure (columns, types, domains, label ranges).
2. Verifies GTZAN zero cross-split song leakage.
3. Tests AudioDataset item loading, feature extractor output shape (1024, 128), and keys.
4. Tests DataLoader batch collation shape (B, 1024, 128).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.dataset import AudioDataset, create_dataloader


@pytest.fixture
def temp_audio_and_manifest(tmp_path):
    """Creates a minimal synthetic test environment with 16kHz audio clips and manifest."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    manifest_data = []
    # Create 4 synthetic 5.0s wav files (2 music domain=0, 2 env domain=1)
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    for i in range(4):
        # Sine wave tone
        freq = 440.0 * (i + 1)
        audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        clip_path = audio_dir / f"clip_{i}.wav"
        sf.write(str(clip_path), audio, sr)

        domain = 0 if i < 2 else 1
        label = i if domain == 0 else (i + 10)
        manifest_data.append({
            "filepath": str(clip_path),
            "domain": domain,
            "label": label,
        })

    manifest_path = tmp_path / "test_manifest.csv"
    pd.DataFrame(manifest_data).to_csv(manifest_path, index=False)

    return manifest_path, manifest_data


def test_dataset_item_keys_and_shapes(temp_audio_and_manifest):
    """Verifies that dataset returns input_values, domain, and label with correct shapes."""
    manifest_path, _ = temp_audio_and_manifest

    # Use default AST feature extractor
    dataset = AudioDataset(manifest_path=manifest_path)
    assert len(dataset) == 4

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "input_values" in sample
    assert "domain" in sample
    assert "label" in sample

    input_values = sample["input_values"]
    domain = sample["domain"]
    label = sample["label"]

    assert isinstance(input_values, torch.Tensor)
    assert isinstance(domain, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    # AST spectrogram shape: (1024, 128)
    assert input_values.ndim == 2
    assert input_values.shape == (1024, 128)
    assert domain.ndim == 0
    assert label.ndim == 0
    assert domain.dtype == torch.long
    assert label.dtype == torch.long


def test_dataloader_batch_collation(temp_audio_and_manifest):
    """Verifies that DataLoader properly collates samples into (batch_size, 1024, 128)."""
    manifest_path, _ = temp_audio_and_manifest
    dataset = AudioDataset(manifest_path=manifest_path)

    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert "input_values" in batch
    assert "domain" in batch
    assert "label" in batch

    assert batch["input_values"].shape == (2, 1024, 128)
    assert batch["domain"].shape == (2,)
    assert batch["label"].shape == (2,)


def test_real_manifests_if_present():
    """Validates real generated manifests in data/manifests/ if already generated."""
    manifest_dir = Path("data/manifests")
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"
    music_json = manifest_dir / "music_classes.json"
    env_json = manifest_dir / "env_classes.json"

    if not train_csv.exists():
        pytest.skip("Manifests not yet generated; skipping real manifests test.")

    for split_file in [train_csv, val_csv, test_csv]:
        df = pd.read_csv(split_file)
        assert "filepath" in df.columns
        assert "domain" in df.columns
        assert "label" in df.columns
        assert set(df["domain"].unique()).issubset({0, 1})

        # Check label bounds
        music_df = df[df["domain"] == 0]
        if len(music_df) > 0:
            assert music_df["label"].min() >= 0
            assert music_df["label"].max() <= 9

        env_df = df[df["domain"] == 1]
        if len(env_df) > 0:
            assert env_df["label"].min() >= 0
            assert env_df["label"].max() <= 49

    # Check class mapping files
    with open(music_json, "r") as f:
        music_classes = json.load(f)
    assert len(music_classes) == 10

    with open(env_json, "r") as f:
        env_classes = json.load(f)
    assert len(env_classes) == 50


def test_zero_gtzan_leakage_if_manifests_present():
    """Ensures that no GTZAN song track appears in more than one split."""
    manifest_dir = Path("data/manifests")
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        pytest.skip("Manifests not yet generated; skipping leakage test.")

    def get_song_stems(csv_path):
        df = pd.read_csv(csv_path)
        music_df = df[df["domain"] == 0]
        # Filename format: e.g. "blues.00001_chunk3.wav" -> song stem "blues.00001"
        stems = set()
        for fp in music_df["filepath"]:
            p = Path(fp)
            stem = p.stem.split("_chunk")[0]
            stems.add(stem)
        return stems

    train_stems = get_song_stems(train_csv)
    val_stems = get_song_stems(val_csv)
    test_stems = get_song_stems(test_csv)

    assert len(train_stems & val_stems) == 0, "Leakage detected between train and val!"
    assert len(train_stems & test_stems) == 0, "Leakage detected between train and test!"
    assert len(val_stems & test_stems) == 0, "Leakage detected between val and test!"
