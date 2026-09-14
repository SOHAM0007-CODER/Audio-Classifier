"""
src/dataset.py

PyTorch Dataset implementation for Multi-Task Audio Spectrogram Transformer (AST) classification.
Reads manifest CSV files containing audio filepaths, domains (music vs environmental), and labels.
Loads 5-second 16 kHz waveforms into memory and extracts log-mel spectrogram features using
HuggingFace's AutoFeatureExtractor (MIT/ast-finetuned-audioset-10-10-0.4593).
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor

DEFAULT_AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION = 5.0  # seconds


def load_audio(filepath: Union[str, Path], target_sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Loads an audio file as a mono float32 waveform at target_sr (channels averaged, polyphase resampling)."""
    try:
        waveform, sr = sf.read(str(filepath), dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file '{filepath}': {e}")

    # Convert multi-channel to mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample to target_sr if required
    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        waveform = resample_poly(waveform, target_sr // gcd, sr // gcd)

    return waveform.astype(np.float32)


class AudioDataset(Dataset):
    """
    Multi-Task Audio Spectrogram Transformer Dataset.

    Reads a CSV manifest with columns:
      - filepath: Path to the 5.0s audio clip.
      - domain: Integer domain ID (0 for music, 1 for environmental).
      - label: Integer class ID (0-9 for music, 0-49 for environmental).

    Extracts log-mel spectrogram features using AutoFeatureExtractor.

    Returns dictionary:
      - 'input_values': Tensor of shape (1024, 128) containing log-mel spectrogram features.
      - 'domain': LongTensor scalar.
      - 'label': LongTensor scalar.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        feature_extractor: Optional[Union[str, Any]] = DEFAULT_AST_MODEL,
        target_sr: int = DEFAULT_SAMPLE_RATE,
        target_duration: float = DEFAULT_DURATION,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            manifest_path: Path to the manifest CSV file (e.g. data/manifests/train.csv).
            feature_extractor: HuggingFace model name or an instantiated ASTFeatureExtractor.
            target_sr: Target audio sampling rate in Hz (default 16,000 Hz).
            target_duration: Target clip duration in seconds (default 5.0s).
            base_dir: Optional base directory to resolve relative filepaths against.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        required_cols = {"filepath", "domain", "label"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Manifest CSV missing required columns: {missing_cols}")

        self.target_sr = int(target_sr)
        self.target_samples = int(self.target_sr * target_duration)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # Initialize HuggingFace AST feature extractor
        if isinstance(feature_extractor, str):
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor)
        elif feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(DEFAULT_AST_MODEL)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_filepath(self, filepath_str: str) -> Path:
        """Resolves relative filepaths against base_dir or manifest directory."""
        path = Path(filepath_str)
        if path.is_absolute() and path.exists():
            return path
        # Try relative to base_dir
        candidate = (self.base_dir / path).resolve()
        if candidate.exists():
            return candidate
        # Try relative to manifest directory
        candidate = (self.manifest_path.parent / path).resolve()
        if candidate.exists():
            return candidate
        # Try relative to repo root (manifest_path.parent.parent.parent)
        candidate = (self.manifest_path.resolve().parents[2] / path).resolve()
        if candidate.exists():
            return candidate
        return candidate

    def _load_audio(self, filepath: Path) -> np.ndarray:
        """
        Loads audio as mono float32 at target_sr and pads or truncates to exact
        target_samples (80,000 for 5.0s @ 16kHz).
        """
        waveform = load_audio(filepath, self.target_sr)

        # Pad or truncate to target_samples (80,000)
        num_samples = len(waveform)
        if num_samples < self.target_samples:
            waveform = np.pad(
                waveform, (0, self.target_samples - num_samples), mode="constant"
            )
        elif num_samples > self.target_samples:
            waveform = waveform[: self.target_samples]

        return waveform.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = self._resolve_filepath(str(row["filepath"]))
        waveform = self._load_audio(audio_path)

        # Feature extractor returns a dictionary with 'input_values' of shape (1, 1024, 128)
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )

        # Squeeze batch dimension so DataLoader can collate to (batch_size, 1024, 128)
        input_values = inputs["input_values"].squeeze(0)  # Shape: (1024, 128)

        domain = torch.tensor(int(row["domain"]), dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "input_values": input_values,
            "domain": domain,
            "label": label,
        }


def create_dataloader(
    dataset: AudioDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader optimized for consumer hardware (e.g. GTX 1650 4GB).

    Args:
        dataset: AudioDataset instance.
        batch_size: Mini-batch size (default 8; 4-8 recommended for 4GB VRAM).
        shuffle: Whether to shuffle data every epoch.
        num_workers: Subprocesses for data loading (0 recommended on Windows).
        pin_memory: Speeds up host-to-GPU memory transfer when CUDA is available.
        drop_last: Whether to drop incomplete trailing batch.

    Returns:
        DataLoader collating batches into:
          - input_values: (batch_size, 1024, 128)
          - domain: (batch_size,)
          - label: (batch_size,)
    """
    cuda_available = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(pin_memory and cuda_available),
        drop_last=drop_last,
    )
