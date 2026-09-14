"""
tests/test_evaluate.py

Tests for src/evaluate.py with a Trainer-saved checkpoint of the 2-layer backbone and a manifest of
synthetic audio clips:
1. The CLI reports per-task metrics identical to evaluate_model and writes them to JSON.
2. Missing checkpoint / manifest files fail with exit code 1.
"""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from src.dataset import DEFAULT_AST_MODEL, AudioDataset, create_dataloader
from src.evaluate import main
from src.models.multitask_ast import MultiTaskAST
from src.predict import load_checkpoint_model
from src.trainer import BEST_MODEL_NAME, TrainConfig, Trainer, evaluate_model


@pytest.fixture(scope="module")
def eval_artifacts(tmp_path_factory, tiny_config):
    root = tmp_path_factory.mktemp("evaluate")
    sr = 16000
    t = np.arange(5 * sr) / sr
    rows = []
    for i, (domain, label) in enumerate([(0, 3), (1, 42), (0, 9), (1, 7)]):
        path = root / f"clip_{i}.wav"
        sf.write(str(path), (0.3 * np.sin(2 * np.pi * 220 * (i + 1) * t)).astype(np.float32), sr)
        rows.append({"filepath": str(path), "domain": domain, "label": label})
    manifest = root / "test.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    torch.manual_seed(0)
    trainer = Trainer(
        MultiTaskAST(backbone_config=tiny_config),
        DataLoader([0]),
        DataLoader([0]),
        TrainConfig(output_dir=str(root / "checkpoints")),
        device=torch.device("cpu"),
    )
    trainer._save_best_model(epoch=0, metrics={})
    return root, manifest, root / "checkpoints" / BEST_MODEL_NAME


def test_evaluate_cli_matches_evaluate_model(eval_artifacts, capsys):
    root, manifest, checkpoint = eval_artifacts
    output = root / "metrics" / "test_metrics.json"
    argv = [
        "--checkpoint", str(checkpoint),
        "--manifest", str(manifest),
        "--batch_size", "2",
        "--num_workers", "0",
        "--device", "cpu",
        "--output", str(output),
    ]
    assert main(argv) == 0
    assert "Combined metric" in capsys.readouterr().out

    metrics = json.loads(output.read_text())
    assert metrics["checkpoint"] == str(checkpoint)
    assert metrics["manifest"] == str(manifest)
    assert metrics["music_num_samples"] == 2 and metrics["env_num_samples"] == 2
    assert metrics["combined_metric"] == pytest.approx(0.5 * metrics["music_acc"] + 0.5 * metrics["env_acc"])

    model, _ = load_checkpoint_model(checkpoint)
    feature_extractor = AutoFeatureExtractor.from_pretrained(DEFAULT_AST_MODEL)
    loader = create_dataloader(AudioDataset(manifest, feature_extractor=feature_extractor), batch_size=2, shuffle=False)
    expected = evaluate_model(model, loader, torch.device("cpu"))
    for key, value in expected.items():
        if isinstance(value, float):
            assert metrics[key] == pytest.approx(value, abs=1e-6), key
        else:
            assert metrics[key] == value, key


def test_evaluate_missing_inputs(eval_artifacts, capsys):
    root, manifest, checkpoint = eval_artifacts

    assert main(["--checkpoint", str(root / "missing.pt"), "--manifest", str(manifest), "--device", "cpu"]) == 1
    assert "Checkpoint not found" in capsys.readouterr().err

    assert main(["--checkpoint", str(checkpoint), "--manifest", str(root / "missing.csv"), "--device", "cpu"]) == 1
    assert "Manifest not found" in capsys.readouterr().err
