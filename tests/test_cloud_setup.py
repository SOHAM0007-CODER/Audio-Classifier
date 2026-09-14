"""
tests/test_cloud_setup.py

Static checks for the cloud training assets:
1. configs/cloud_train.yaml parses into the expected TrainConfig (and CLI flags still override it).
2. notebooks/train_colab.ipynb is clean nbformat-4 JSON whose code cells compile and run, in order, the
   environment check, installation, data preparation, cloud training, evaluation and prediction.
3. scripts/train_cloud.sh keeps LF line endings and runs the same steps.
4. scripts/package_for_cloud.py zips the code without datasets, checkpoints, caches or virtual environments.
"""

import importlib.util
import json
import re
import zipfile
from pathlib import Path

import pytest

from src.train import parse_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_CONFIG = REPO_ROOT / "configs" / "cloud_train.yaml"
EXPECTED_STEPS = [
    "nvidia-smi",
    "pip install -q -r requirements.txt",
    "scripts/prepare_manifests.py",
    "src.train --config",
    "src.evaluate --checkpoint",
    "src.predict --audio",
]


def _load_package_script():
    spec = importlib.util.spec_from_file_location("package_for_cloud", REPO_ROOT / "scripts" / "package_for_cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_in_order(text, steps):
    positions = [text.find(step) for step in steps]
    assert all(position >= 0 for position in positions), dict(zip(steps, positions))
    assert positions == sorted(positions), dict(zip(steps, positions))


def test_cloud_train_config():
    config = parse_config(["--config", str(CLOUD_CONFIG)])
    assert config.batch_size == 16
    assert config.grad_accum_steps == 1
    assert config.mixed_precision is True
    assert config.gradient_checkpointing is False
    assert config.num_workers == 4
    assert config.epochs == 20
    assert config.lr_backbone == pytest.approx(2e-5)
    assert config.lr_heads == pytest.approx(5e-4)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.warmup_epochs == pytest.approx(2.0)
    assert config.output_dir == "checkpoints/cloud_run"
    assert config.music_loss_weight == 1.0 and config.env_loss_weight == 1.0

    assert parse_config(["--config", str(CLOUD_CONFIG), "--num_workers", "2"]).num_workers == 2


def test_colab_notebook():
    notebook = json.loads((REPO_ROOT / "notebooks" / "train_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"

    code_sources = []
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            # IPython shell lines (`!cmd`) are not Python; replace them before compiling.
            compile(re.sub(r"^(\s*)!.*$", r"\1pass", source, flags=re.MULTILINE), "<notebook cell>", "exec")
            code_sources.append(source)

    assert len(code_sources) >= 5
    _assert_in_order("\n".join(code_sources), EXPECTED_STEPS)
    assert "configs/cloud_train.yaml --resume auto" in code_sources[3]


def test_cloud_shell_script():
    script = (REPO_ROOT / "scripts" / "train_cloud.sh").read_bytes()
    assert b"\r\n" not in script
    text = script.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    _assert_in_order(text, ["nvidia-smi", "pip install -q -r requirements.txt", "scripts/prepare_manifests.py",
                            "src.train --config", "src.evaluate --checkpoint", "src.predict --audio"])


def test_package_for_cloud(tmp_path):
    output = tmp_path / "code.zip"
    names = _load_package_script().package_project(output)

    for required in (
        "requirements.txt",
        "configs/cloud_train.yaml",
        "notebooks/train_colab.ipynb",
        "scripts/prepare_manifests.py",
        "scripts/train_cloud.sh",
        "src/train.py",
        "src/evaluate.py",
        "src/predict.py",
        "tests/conftest.py",
    ):
        assert required in names

    excluded_prefixes = ("Data/", "data/", ".venv/", "checkpoints/", "dist/")
    assert not [n for n in names if n.startswith(excluded_prefixes) or "__pycache__" in n or n.endswith(".pyc")]
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == sorted(names)
