"""
tests/test_predict.py

Tests for src/predict.py on synthetic audio files, using the 2-layer AST backbone fixture:
1. Domain parsing, class-name loading and 5.0 s windowing.
2. Python API: AudioClassifier.predict and predict() on a checkpoint written by the Trainer, for the
   music / env / both domains, including stereo 44.1 kHz input resampled to 16 kHz mono.
3. CLI: formatted text output, --json output (in-process and via `python -m src.predict`), error exits.
4. Fallback without a checkpoint (pretrained backbone, untrained heads; skipped if weights unavailable).
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.models.multitask_ast import MultiTaskAST
from src.predict import (
    DEFAULT_CHECKPOINT,
    AudioClassifier,
    format_results,
    load_class_names,
    main,
    parse_domain,
    predict,
    split_into_chunks,
)
from src.trainer import BEST_MODEL_NAME, TrainConfig, Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, tiny_config):
    """Synthetic audio files, class mappings, and a best_model.pt saved by the Trainer for the 2-layer backbone."""
    root = tmp_path_factory.mktemp("predict")

    # 12 s stereo 44.1 kHz tone -> two full 5 s windows; the 2 s tail is shorter than half a window and dropped.
    sr = 44100
    t = np.arange(12 * sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    long_audio = root / "long_stereo.wav"
    sf.write(str(long_audio), np.stack([tone, 0.5 * tone], axis=1).astype(np.float32), sr)

    # 3 s mono 16 kHz noise -> a single zero-padded window.
    short_audio = root / "short.wav"
    noise = 0.1 * np.random.default_rng(0).standard_normal(3 * 16000)
    sf.write(str(short_audio), noise.astype(np.float32), 16000)

    music_classes = root / "music_classes.json"
    music_classes.write_text(json.dumps({name: i for i, name in enumerate(GENRES)}))
    env_classes = root / "env_classes.json"
    env_classes.write_text(json.dumps({f"sound_{i}": i for i in range(50)}))

    torch.manual_seed(0)
    model = MultiTaskAST(backbone_config=tiny_config)
    output_dir = root / "checkpoints"
    trainer = Trainer(
        model, DataLoader([0]), DataLoader([0]), TrainConfig(output_dir=str(output_dir)), device=torch.device("cpu")
    )
    trainer._save_best_model(epoch=0, metrics={})

    return SimpleNamespace(
        root=root,
        long_audio=long_audio,
        short_audio=short_audio,
        music_classes=music_classes,
        env_classes=env_classes,
        checkpoint=output_dir / BEST_MODEL_NAME,
        model=model,
    )


@pytest.fixture(scope="module")
def classifier(artifacts):
    return AudioClassifier.from_checkpoint(
        artifacts.checkpoint,
        device="cpu",
        music_classes=artifacts.music_classes,
        env_classes=artifacts.env_classes,
    )


def _cli_args(artifacts, audio, *extra):
    return [
        "--audio", str(audio),
        "--checkpoint", str(artifacts.checkpoint),
        "--music_classes", str(artifacts.music_classes),
        "--env_classes", str(artifacts.env_classes),
        "--device", "cpu",
        *extra,
    ]


def _check_predictions(predictions, class_names, expected_len):
    assert len(predictions) == expected_len
    assert [p["rank"] for p in predictions] == list(range(1, expected_len + 1))
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    for p in predictions:
        assert class_names[p["index"]] == p["label"]


def test_parse_domain():
    for value in ("music", "MUSIC", "0", 0):
        assert parse_domain(value) == ["music"]
    for value in ("env", "environmental", "1", 1):
        assert parse_domain(value) == ["env"]
    for value in (None, "auto", "both", "Both"):
        assert parse_domain(value) == ["music", "env"]
    with pytest.raises(ValueError):
        parse_domain("speech")


def test_split_into_chunks():
    chunks = split_into_chunks(np.ones(25, dtype=np.float32), chunk_samples=10)
    assert [len(c) for c in chunks] == [10, 10, 10]
    assert chunks[-1].tolist() == [1.0] * 5 + [0.0] * 5

    assert len(split_into_chunks(np.ones(24, dtype=np.float32), chunk_samples=10)) == 2  # 4-sample tail dropped
    short = split_into_chunks(np.ones(3, dtype=np.float32), chunk_samples=10)
    assert len(short) == 1 and len(short[0]) == 10
    with pytest.raises(ValueError):
        split_into_chunks(np.zeros(0, dtype=np.float32), chunk_samples=10)


def test_load_class_names(artifacts, tmp_path):
    names = load_class_names(artifacts.music_classes, 10)
    assert names == GENRES

    assert load_class_names(tmp_path / "missing.json", 3) == ["class_0", "class_1", "class_2"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": 0, "b": 2}))
    with pytest.raises(ValueError):
        load_class_names(bad, 2)


def test_predict_both_domains(classifier, artifacts):
    """Stereo 44.1 kHz input is resampled to 16 kHz mono and split into two 5 s windows."""
    result = classifier.predict(artifacts.long_audio, top_k=3)

    assert set(result["predictions"]) == {"music", "env"}
    assert result["duration_sec"] == pytest.approx(12.0)
    assert result["num_chunks"] == 2
    assert result["audio"] == str(artifacts.long_audio)
    _check_predictions(result["predictions"]["music"], GENRES, 3)
    _check_predictions(result["predictions"]["env"], classifier.class_names["env"], 3)
    json.dumps(result)  # JSON-serializable


def test_predict_single_domain_and_top_k(classifier, artifacts):
    music = classifier.predict(artifacts.short_audio, domain="music", top_k=100)
    assert set(music["predictions"]) == {"music"}
    assert music["num_chunks"] == 1 and music["duration_sec"] == pytest.approx(3.0)
    _check_predictions(music["predictions"]["music"], GENRES, 10)  # top_k capped at the number of classes
    assert sum(p["confidence"] for p in music["predictions"]["music"]) == pytest.approx(1.0, abs=1e-5)

    env = classifier.predict(artifacts.short_audio, domain=1, top_k=5)
    assert set(env["predictions"]) == {"env"}
    _check_predictions(env["predictions"]["env"], classifier.class_names["env"], 5)

    with pytest.raises(FileNotFoundError):
        classifier.predict(artifacts.root / "missing.wav")


def test_predict_matches_manual_forward(classifier, artifacts):
    """A single padded window must match running the feature extractor and model by hand."""
    waveform, sr = sf.read(str(artifacts.short_audio), dtype="float32")
    padded = np.pad(waveform, (0, 5 * sr - len(waveform)))
    features = classifier.feature_extractor(padded, sampling_rate=sr, return_tensors="pt")["input_values"]
    with torch.no_grad():
        expected = classifier.model(features)["music_logits"].softmax(dim=-1)[0]

    result = classifier.predict(artifacts.short_audio, domain="music", top_k=10)
    actual = torch.zeros(10)
    for p in result["predictions"]["music"]:
        actual[p["index"]] = p["confidence"]
    assert torch.allclose(actual, expected, atol=1e-5)


def test_predict_function_with_checkpoint(artifacts):
    result = predict(
        artifacts.short_audio,
        checkpoint=artifacts.checkpoint,
        domain="env",
        top_k=5,
        device="cpu",
        music_classes=artifacts.music_classes,
        env_classes=artifacts.env_classes,
    )
    assert result["checkpoint"] == str(artifacts.checkpoint)
    assert set(result["predictions"]) == {"env"}
    assert len(result["predictions"]["env"]) == 5


def test_cli_text_output(artifacts, capsys):
    assert main(_cli_args(artifacts, artifacts.long_audio)) == 0
    out = capsys.readouterr().out
    assert "Music genre (GTZAN)" in out and "Environmental sound (ESC-50)" in out
    assert str(artifacts.checkpoint) in out
    assert out.count("%") == 6  # top-3 for each head


def test_cli_json_output(artifacts, capsys):
    assert main(_cli_args(artifacts, artifacts.short_audio, "--domain", "music", "--top_k", "5", "--json")) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["predictions"]) == {"music"}
    _check_predictions(result["predictions"]["music"], GENRES, 5)


def test_cli_subprocess_json(artifacts):
    """End-to-end `python -m src.predict --json`: stdout must be valid JSON only."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.predict", *_cli_args(artifacts, artifacts.long_audio, "--domain", "both", "--json")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert set(result["predictions"]) == {"music", "env"}
    assert result["num_chunks"] == 2


def test_cli_errors(artifacts, capsys):
    assert main(_cli_args(artifacts, artifacts.root / "missing.wav")) == 1
    assert "Audio file not found" in capsys.readouterr().err

    args = _cli_args(artifacts, artifacts.short_audio)
    args[args.index("--checkpoint") + 1] = str(artifacts.root / "missing.pt")
    assert main(args) == 1
    assert "Checkpoint not found" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        main(_cli_args(artifacts, artifacts.short_audio, "--domain", "speech"))
    assert excinfo.value.code == 2


def test_fallback_without_checkpoint(artifacts, monkeypatch, tmp_path):
    """A missing default checkpoint falls back to the pretrained backbone with untrained heads."""
    monkeypatch.chdir(tmp_path)  # empty directory: no checkpoints/best_model.pt
    try:
        fallback = AudioClassifier.from_checkpoint(
            DEFAULT_CHECKPOINT,
            device="cpu",
            music_classes=artifacts.music_classes,
            env_classes=artifacts.env_classes,
        )
    except OSError as e:  # offline / no cached weights
        pytest.skip(f"Pretrained AST checkpoint unavailable: {e}")

    assert fallback.model.backbone.config.num_hidden_layers == 12
    result = fallback.predict(artifacts.short_audio, top_k=3)
    assert result["checkpoint"] is None
    _check_predictions(result["predictions"]["music"], GENRES, 3)
    _check_predictions(result["predictions"]["env"], fallback.class_names["env"], 3)
    assert "untrained heads" in format_results(result)
