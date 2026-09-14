"""
tests/test_train.py

Fast tests for the training / validation pipeline on synthetic spectrograms, using the
2-layer AST backbone fixture from conftest.py:
1. Accuracy / macro-F1 helper against hand-computed values.
2. Warmup + cosine learning-rate schedule and backbone / head parameter groups.
3. One training step + one validation step: logged losses and learning rates, per-task metrics,
   combined metric, best_model.pt and latest_checkpoint.pt.
4. Gradient accumulation step counting and resuming from latest_checkpoint.pt.
5. Config parsing from YAML with command-line overrides.
"""

import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.multitask_ast import MultiTaskAST
from src.train import parse_config
from src.trainer import (
    BEST_MODEL_NAME,
    LATEST_CHECKPOINT_NAME,
    METRICS_LOG_NAME,
    TrainConfig,
    Trainer,
    build_scheduler,
    classification_metrics,
)

MAX_LENGTH = 1024
NUM_MEL_BINS = 128


class SyntheticSpectrogramDataset(Dataset):
    """Random (1024, 128) spectrograms alternating music (domain 0) and environmental (domain 1) samples."""

    def __init__(self, num_samples: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.input_values = torch.randn(num_samples, MAX_LENGTH, NUM_MEL_BINS, generator=generator)
        self.domain = torch.arange(num_samples) % 2
        music_labels = torch.randint(0, 10, (num_samples,), generator=generator)
        env_labels = torch.randint(0, 50, (num_samples,), generator=generator)
        self.label = torch.where(self.domain == 0, music_labels, env_labels)

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, idx):
        return {"input_values": self.input_values[idx], "domain": self.domain[idx], "label": self.label[idx]}


@pytest.fixture
def make_trainer(tiny_config, tmp_path):
    """Builds a CPU Trainer around the 2-layer backbone with synthetic train / val loaders."""

    def _make(num_train_samples=4, num_val_samples=4, **overrides):
        config_kwargs = dict(
            output_dir=str(tmp_path / "checkpoints"),
            epochs=1,
            batch_size=4,
            warmup_epochs=0,
            log_every=1,
            mixed_precision=False,
            num_workers=0,
        )
        config_kwargs.update(overrides)
        config = TrainConfig(**config_kwargs)
        torch.manual_seed(0)
        model = MultiTaskAST(
            backbone_config=tiny_config,
            music_loss_weight=config.music_loss_weight,
            env_loss_weight=config.env_loss_weight,
        )
        train_loader = DataLoader(SyntheticSpectrogramDataset(num_train_samples, seed=0), batch_size=config.batch_size)
        val_loader = DataLoader(SyntheticSpectrogramDataset(num_val_samples, seed=1), batch_size=config.batch_size)
        return Trainer(model, train_loader, val_loader, config, device=torch.device("cpu"))

    return _make


def test_classification_metrics():
    preds = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])
    metrics = classification_metrics(preds, targets, num_classes=4)

    assert metrics["acc"] == pytest.approx(3 / 4)
    # Per-class F1: class 0 = 1, class 1 = 2/3, class 2 = 2/3; class 3 never appears and is excluded.
    assert metrics["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
    assert metrics["num_samples"] == 4

    empty = classification_metrics(torch.tensor([]), torch.tensor([]), num_classes=4)
    assert empty == {"acc": 0.0, "macro_f1": 0.0, "num_samples": 0}


def test_warmup_cosine_schedule():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = build_scheduler(optimizer, num_training_steps=10, num_warmup_steps=2)

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert all(later <= earlier for earlier, later in zip(lrs[2:], lrs[3:]))
    assert lrs[-1] == pytest.approx(0.5 * (1 + math.cos(math.pi * 7 / 8)))


def test_optimizer_param_groups(make_trainer):
    trainer = make_trainer(lr_backbone=1e-5, lr_heads=1e-3, weight_decay=0.05)
    groups = {group["name"]: group for group in trainer.optimizer.param_groups}

    assert set(groups) == {"backbone_decay", "backbone_no_decay", "heads_decay", "heads_no_decay"}
    for name, group in groups.items():
        assert group["lr"] == pytest.approx(1e-5 if name.startswith("backbone") else 1e-3)
        assert group["weight_decay"] == (0.0 if name.endswith("no_decay") else 0.05)

    grouped_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert grouped_ids == {id(p) for p in trainer.model.parameters()}
    no_decay_ids = {id(p) for p in groups["backbone_no_decay"]["params"]}
    assert id(trainer.model.backbone.embeddings.cls_token) in no_decay_ids


def test_one_train_and_val_step_saves_checkpoints(make_trainer):
    trainer = make_trainer()
    model = trainer.model
    music_head_before = model.music_head.net[-1].weight.detach().clone()
    env_head_before = model.env_head.net[-1].weight.detach().clone()
    backbone_before = model.backbone.layers[0].attention.q_proj.weight.detach().clone()

    history = trainer.fit()

    # One training step and one validation step.
    assert trainer.global_step == 1
    assert len(history) == 1
    record = history[0]
    assert math.isfinite(record["train_loss"]) and math.isfinite(record["val_loss"])
    for task in ("music", "env"):
        assert record[f"val_{task}_num_samples"] == 2
        assert 0.0 <= record[f"val_{task}_acc"] <= 1.0
        assert 0.0 <= record[f"val_{task}_macro_f1"] <= 1.0
    assert record["val_combined_metric"] == pytest.approx(
        0.5 * record["val_music_acc"] + 0.5 * record["val_env_acc"]
    )
    assert record["is_best"]

    assert not torch.equal(music_head_before, model.music_head.net[-1].weight)
    assert not torch.equal(env_head_before, model.env_head.net[-1].weight)
    assert not torch.equal(backbone_before, model.backbone.layers[0].attention.q_proj.weight)

    output_dir = trainer.output_dir
    best = torch.load(output_dir / BEST_MODEL_NAME, map_location="cpu", weights_only=True)
    assert best["epoch"] == 0
    assert best["backbone_config"]["num_hidden_layers"] == 2
    assert best["metrics"]["combined_metric"] == pytest.approx(record["val_combined_metric"])
    model.load_state_dict(best["model_state_dict"])

    latest = torch.load(output_dir / LATEST_CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    assert latest["epoch"] == 0 and latest["global_step"] == 1
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"):
        assert key in latest

    records = [json.loads(line) for line in (output_dir / METRICS_LOG_NAME).read_text().splitlines()]
    train_records = [r for r in records if r["type"] == "train"]
    assert len(train_records) == 1
    for key in ("loss", "music_loss", "env_loss", "grad_norm"):
        assert math.isfinite(train_records[0][key])
    assert train_records[0]["lr_backbone"] == pytest.approx(2e-5)
    assert train_records[0]["lr_heads"] == pytest.approx(5e-4)
    assert [r["epoch"] for r in records if r["type"] == "epoch"] == [1]


def test_grad_accumulation_and_resume(make_trainer):
    # 3 micro-batches with grad_accum_steps=2 -> one full window plus one shorter trailing window.
    trainer = make_trainer(num_train_samples=6, batch_size=2, grad_accum_steps=2, log_every=100)
    assert trainer.steps_per_epoch == 2
    trainer.fit()
    assert trainer.global_step == 2

    resumed = make_trainer(num_train_samples=6, batch_size=2, grad_accum_steps=2, log_every=100, epochs=2)
    resumed.load_checkpoint(trainer.output_dir / LATEST_CHECKPOINT_NAME)
    assert resumed.start_epoch == 1 and resumed.global_step == 2
    assert resumed.optimizer.state_dict()["state"]
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])

    history = resumed.fit()
    assert resumed.global_step == 4
    assert [r["epoch"] for r in history] == [1, 2]


def test_parse_config_yaml_and_cli_overrides(tmp_path):
    assert parse_config([]) == TrainConfig()

    config_path = tmp_path / "train.yaml"
    config_path.write_text("batch_size: 16\nlr_heads: 1e-3\nmixed_precision: false\nhead_hidden_dim: null\n")
    config = parse_config(["--config", str(config_path), "--batch_size", "4", "--output_dir", "runs/x"])
    assert config.batch_size == 4
    assert isinstance(config.lr_heads, float) and config.lr_heads == pytest.approx(1e-3)
    assert config.mixed_precision is False
    assert config.epochs == TrainConfig().epochs
    assert config.output_dir == "runs/x"

    assert parse_config(["--config", str(config_path), "--fp16"]).mixed_precision is True
    assert parse_config(["--no-fp16"]).mixed_precision is False

    config_path.write_text("not_a_field: 1\n")
    with pytest.raises(ValueError):
        parse_config(["--config", str(config_path)])
