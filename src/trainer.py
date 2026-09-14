"""
src/trainer.py

Training and validation engine for the Multi-Task AST classifier.
Handles AdamW with separate backbone / head learning rates, linear warmup + cosine decay,
CUDA mixed precision, gradient accumulation and clipping, per-task validation metrics
(accuracy and macro-F1 for GTZAN and ESC-50), and best / latest checkpointing.
"""

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DEFAULT_AST_MODEL
from .models.multitask_ast import ENV_DOMAIN, MUSIC_DOMAIN

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = "best_model.pt"
LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
METRICS_LOG_NAME = "metrics.jsonl"
TASKS = ("music", "env")

# Parameters excluded from weight decay in addition to all 1-D parameters (biases, LayerNorms).
_NO_DECAY_KEYWORDS = ("cls_token", "distillation_token", "position_embeddings")


def _arg(default: Any, help: str, **metadata: Any) -> Any:
    """Dataclass field carrying CLI help text (and optional flag aliases) for src/train.py."""
    return field(default=default, metadata={"help": help, **metadata})


@dataclass
class TrainConfig:
    """Training hyperparameters. Every field is exposed as a command-line flag by src/train.py."""

    # Data
    train_manifest: str = _arg("data/manifests/train.csv", "Training manifest CSV.")
    val_manifest: str = _arg("data/manifests/val.csv", "Validation manifest CSV.")
    test_manifest: str = _arg(
        "data/manifests/test.csv", "Test manifest CSV, evaluated with best_model.pt after training."
    )
    batch_size: int = _arg(8, "Mini-batch size per micro-step.")
    num_workers: int = _arg(0 if os.name == "nt" else 4, "DataLoader worker processes (0 on Windows).")

    # Optimization
    lr_backbone: float = _arg(2e-5, "Peak AdamW learning rate for the AST backbone.")
    lr_heads: float = _arg(5e-4, "Peak AdamW learning rate for the music / env heads.")
    weight_decay: float = _arg(1e-4, "AdamW weight decay (not applied to biases, norms and tokens).")
    epochs: int = _arg(20, "Number of training epochs.")
    warmup_epochs: float = _arg(2.0, "Linear warmup length in epochs, followed by cosine decay.")
    min_lr_ratio: float = _arg(0.0, "Final learning rate as a fraction of the peak learning rate.")
    grad_accum_steps: int = _arg(1, "Micro-batches accumulated per optimizer step.")
    max_grad_norm: float = _arg(1.0, "Gradient clipping max norm.")
    # Off by default: GTX 16xx GPUs (no tensor cores) run float16 ~5x slower than float32.
    mixed_precision: bool = _arg(
        False, "float16 autocast + GradScaler on CUDA (ignored on CPU; slow on GTX 16xx).", aliases=("--fp16",)
    )
    # On by default: without it only batch size <= 2 fits on a 4GB GPU.
    gradient_checkpointing: bool = _arg(
        True, "Recompute backbone activations during backward to reduce GPU memory."
    )

    # Model
    pretrained_model_name: str = _arg(DEFAULT_AST_MODEL, "HuggingFace AST checkpoint for the backbone.")
    head_hidden_dim: Optional[int] = _arg(None, "Hidden width for MLP heads (default: LayerNorm + Linear).")
    head_dropout: float = _arg(0.1, "Dropout inside the classification heads.")
    music_loss_weight: float = _arg(1.0, "Weight of the music CrossEntropy term.")
    env_loss_weight: float = _arg(1.0, "Weight of the environmental CrossEntropy term.")

    # Run
    output_dir: str = _arg("checkpoints", "Directory for checkpoints, logs and metrics.")
    log_every: int = _arg(50, "Log losses and learning rates every N optimizer steps.")
    seed: int = _arg(42, "Random seed.")
    resume: Optional[str] = _arg(
        None, "Checkpoint to resume from, or 'auto' to use <output_dir>/latest_checkpoint.pt if present."
    )
    eval_test: bool = _arg(True, "Evaluate best_model.pt on test_manifest after training.")
    max_train_batches: Optional[int] = _arg(None, "Limit training batches per epoch (debugging).")
    max_eval_batches: Optional[int] = _arg(None, "Limit evaluation batches (debugging).")

    def __post_init__(self):
        for name in ("batch_size", "epochs", "grad_accum_steps", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classification_metrics(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Dict[str, Union[float, int]]:
    """
    Accuracy and macro-F1. Like sklearn's f1_score(average='macro'), the macro average is taken
    over classes that appear in the targets or the predictions.
    """
    preds = preds.detach().long().cpu().view(-1)
    targets = targets.detach().long().cpu().view(-1)
    num_samples = targets.numel()
    if num_samples == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "num_samples": 0}

    # confusion[true_class, predicted_class]
    confusion = torch.bincount(targets * num_classes + preds, minlength=num_classes * num_classes)
    confusion = confusion.view(num_classes, num_classes).double()
    true_positives = confusion.diag()
    # support + predicted count = 2*TP + FP + FN, so F1 = 2*TP / (support + predicted count)
    support_plus_predicted = confusion.sum(dim=1) + confusion.sum(dim=0)
    present = support_plus_predicted > 0
    f1_per_class = 2 * true_positives[present] / support_plus_predicted[present]
    return {
        "acc": true_positives.sum().item() / num_samples,
        "macro_f1": f1_per_class.mean().item(),
        "num_samples": num_samples,
    }


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW with separate learning rates for backbone and heads, and no decay on 1-D params / tokens."""
    groups: Dict[str, Dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        component = "backbone" if name.startswith("backbone.") else "heads"
        no_decay = param.ndim < 2 or any(keyword in name for keyword in _NO_DECAY_KEYWORDS)
        group_name = f"{component}_{'no_decay' if no_decay else 'decay'}"
        if group_name not in groups:
            groups[group_name] = {
                "name": group_name,
                "params": [],
                "lr": config.lr_backbone if component == "backbone" else config.lr_heads,
                "weight_decay": 0.0 if no_decay else config.weight_decay,
            }
        groups[group_name]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Per-optimizer-step linear warmup to each group's peak LR, then cosine decay to min_lr_ratio * peak."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return (step + 1) / num_warmup_steps
        progress = min(1.0, (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class _LossMeter:
    """Running averages: total loss per batch, task losses weighted by the task's sample count."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.num_batches = 0
        self.task_sums = {task: 0.0 for task in TASKS}
        self.task_counts = {task: 0 for task in TASKS}

    def update(self, outputs: Dict[str, Any]) -> None:
        self.loss_sum += outputs["loss"].item()
        self.num_batches += 1
        for task in TASKS:
            if outputs[f"{task}_loss"] is not None:
                count = outputs[f"{task}_indices"].numel()
                self.task_sums[task] += outputs[f"{task}_loss"].item() * count
                self.task_counts[task] += count

    def averages(self) -> Dict[str, Optional[float]]:
        result = {"loss": self.loss_sum / max(1, self.num_batches)}
        for task in TASKS:
            count = self.task_counts[task]
            result[f"{task}_loss"] = self.task_sums[task] / count if count else None
        return result


def _num_batches(loader: DataLoader, max_batches: Optional[int]) -> int:
    num_batches = len(loader)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    if num_batches == 0:
        raise ValueError("DataLoader yields no batches.")
    return num_batches


def _atomic_save(obj: Dict[str, Any], path: Path) -> None:
    """Writes to a temporary file first so an interrupted save never corrupts an existing checkpoint."""
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def forward_batch(
    model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device, use_amp: bool = False
) -> Dict[str, Any]:
    """Moves a dataset batch to device and runs the model with routed losses (float16 autocast if use_amp)."""
    input_values = batch["input_values"].to(device, non_blocking=True)
    domain = batch["domain"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        return model(input_values, domain=domain, label=label)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
    desc: str = "eval",
) -> Dict[str, Any]:
    """
    Evaluates a MultiTaskAST on a loader of mixed-domain batches. Returns loss, per-task accuracy /
    macro-F1 / sample counts, and combined_metric = 0.5 * music_acc + 0.5 * env_acc.
    """
    model.eval()
    num_batches = _num_batches(loader, max_batches)
    preds: Dict[str, List[torch.Tensor]] = {task: [] for task in TASKS}
    targets: Dict[str, List[torch.Tensor]] = {task: [] for task in TASKS}
    loss_meter = _LossMeter()

    progress = tqdm(total=num_batches, desc=desc, leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        outputs = forward_batch(model, batch, device, use_amp)
        loss_meter.update(outputs)
        domain, label = batch["domain"], batch["label"]
        preds["music"].append(outputs["music_logits"].argmax(dim=-1).cpu())
        targets["music"].append(label[domain == MUSIC_DOMAIN])
        preds["env"].append(outputs["env_logits"].argmax(dim=-1).cpu())
        targets["env"].append(label[domain == ENV_DOMAIN])
        progress.update(1)
    progress.close()

    num_classes = {"music": model.num_music_classes, "env": model.num_env_classes}
    metrics: Dict[str, Any] = loss_meter.averages()
    for task in TASKS:
        task_metrics = classification_metrics(torch.cat(preds[task]), torch.cat(targets[task]), num_classes[task])
        if task_metrics["num_samples"] == 0:
            logger.warning("No %s samples in '%s'; its accuracy and macro-F1 are reported as 0.", task, desc)
        metrics.update({f"{task}_{key}": value for key, value in task_metrics.items()})
    metrics["combined_metric"] = 0.5 * metrics["music_acc"] + 0.5 * metrics["env_acc"]
    return metrics


class Trainer:
    """Runs training epochs, per-epoch validation and checkpointing for MultiTaskAST."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        if config.mixed_precision and not self.use_amp:
            logger.info("Mixed precision requested but not on CUDA; training in float32 on %s.", self.device)
        elif self.use_amp and "GTX 16" in torch.cuda.get_device_name(self.device):
            logger.warning(
                "float16 autocast is usually slower than float32 on %s (no tensor cores); consider --no-fp16.",
                torch.cuda.get_device_name(self.device),
            )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = build_optimizer(self.model, config)
        self.train_batches_per_epoch = _num_batches(train_loader, config.max_train_batches)
        self.steps_per_epoch = math.ceil(self.train_batches_per_epoch / config.grad_accum_steps)
        self.total_steps = self.steps_per_epoch * config.epochs
        self.warmup_steps = round(config.warmup_epochs * self.steps_per_epoch)
        self.scheduler = build_scheduler(
            self.optimizer, self.total_steps, self.warmup_steps, config.min_lr_ratio
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_path = self.output_dir / METRICS_LOG_NAME

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf")
        self.history: List[Dict[str, Any]] = []

    def _learning_rates(self) -> Dict[str, float]:
        lrs: Dict[str, float] = {}
        for group in self.optimizer.param_groups:
            component = group["name"].split("_")[0]
            lrs.setdefault(f"lr_{component}", group["lr"])
        return lrs

    def _write_metrics(self, record: Dict[str, Any]) -> None:
        with open(self.metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def train_one_epoch(self, epoch: int) -> Dict[str, Optional[float]]:
        config = self.config
        num_batches = self.train_batches_per_epoch
        accum = config.grad_accum_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        epoch_meter, window_meter = _LossMeter(), _LossMeter()

        progress = tqdm(
            total=num_batches, desc=f"Epoch {epoch + 1}/{config.epochs} [train]", leave=False, dynamic_ncols=True
        )
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= num_batches:
                break
            # The last accumulation window of an epoch may be shorter than grad_accum_steps.
            window_start = batch_idx - batch_idx % accum
            window_size = min(accum, num_batches - window_start)

            outputs = forward_batch(self.model, batch, self.device, self.use_amp)
            loss = outputs["loss"]
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss (%s) at epoch %d, batch %d.", loss.item(), epoch + 1, batch_idx)
            self.scaler.scale(loss / window_size).backward()
            epoch_meter.update(outputs)
            window_meter.update(outputs)
            progress.update(1)

            if batch_idx + 1 - window_start < window_size:
                continue

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
            lrs = self._learning_rates()
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            # GradScaler skips the optimizer step (and lowers the scale) when gradients overflow.
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            self.global_step += 1

            if self.global_step % config.log_every == 0 or batch_idx + 1 == num_batches:
                record = {
                    "type": "train",
                    "epoch": epoch + 1,
                    "step": self.global_step,
                    **window_meter.averages(),
                    **lrs,
                    "grad_norm": grad_norm.item(),
                }
                self._write_metrics(record)
                logger.info(
                    "Epoch %d | step %d | loss %s | music_loss %s | env_loss %s | "
                    "lr_backbone %.2e | lr_heads %.2e | grad_norm %.3f",
                    epoch + 1,
                    self.global_step,
                    _fmt(record["loss"]),
                    _fmt(record["music_loss"]),
                    _fmt(record["env_loss"]),
                    record["lr_backbone"],
                    record["lr_heads"],
                    record["grad_norm"],
                )
                progress.set_postfix(loss=f"{record['loss']:.4f}")
                window_meter.reset()
        progress.close()
        return epoch_meter.averages()

    def evaluate(
        self, loader: DataLoader, max_batches: Optional[int] = None, desc: str = "val"
    ) -> Dict[str, Any]:
        """Returns loss, per-task accuracy / macro-F1 / sample counts, and combined_metric."""
        return evaluate_model(self.model, loader, self.device, self.use_amp, max_batches, desc)

    def fit(self) -> List[Dict[str, Any]]:
        config = self.config
        logger.info(
            "Training on %s | AMP: %s | %d batches/epoch | %d optimizer steps/epoch | %d total steps (%d warmup)",
            self.device,
            self.use_amp,
            self.train_batches_per_epoch,
            self.steps_per_epoch,
            self.total_steps,
            self.warmup_steps,
        )
        for epoch in range(self.start_epoch, config.epochs):
            start_time = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            train_stats = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(
                self.val_loader, config.max_eval_batches, desc=f"Epoch {epoch + 1}/{config.epochs} [val]"
            )

            is_best = val_metrics["combined_metric"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["combined_metric"]
                self._save_best_model(epoch, val_metrics)

            record = {
                "type": "epoch",
                "epoch": epoch + 1,
                "step": self.global_step,
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "is_best": is_best,
                "best_combined_metric": self.best_metric,
                "epoch_time_sec": time.time() - start_time,
            }
            if self.device.type == "cuda":
                record["max_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
            self.history.append(record)
            self._write_metrics(record)
            self.save_checkpoint(epoch)

            logger.info(
                "Epoch %d/%d | train_loss %.4f | val_loss %.4f | music acc %.4f f1 %.4f | "
                "env acc %.4f f1 %.4f | combined %.4f%s | %.0fs",
                epoch + 1,
                config.epochs,
                record["train_loss"],
                record["val_loss"],
                record["val_music_acc"],
                record["val_music_macro_f1"],
                record["val_env_acc"],
                record["val_env_macro_f1"],
                record["val_combined_metric"],
                " (new best)" if is_best else "",
                record["epoch_time_sec"],
            )
        return self.history

    def _save_best_model(self, epoch: int, metrics: Dict[str, Any]) -> None:
        _atomic_save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "config": asdict(self.config),
                # Lets src/predict.py rebuild the backbone without fetching its config from the Hub.
                "backbone_config": self.model.backbone.config.to_dict(),
            },
            self.output_dir / BEST_MODEL_NAME,
        )

    def save_checkpoint(self, epoch: int) -> Path:
        """Saves everything needed to resume after `epoch` (0-based) to latest_checkpoint.pt."""
        path = self.output_dir / LATEST_CHECKPOINT_NAME
        _atomic_save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "history": self.history,
                "config": asdict(self.config),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restores model, optimizer, scheduler and scaler state; training continues at the next epoch."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # A scaler saved while disabled (e.g. CPU run) has an empty state dict.
        if checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        self.history = checkpoint["history"]
        logger.info(
            "Resumed from %s (completed epoch %d, step %d, best combined_metric %.4f).",
            path,
            self.start_epoch,
            self.global_step,
            self.best_metric,
        )

    def load_best_model(self) -> Dict[str, Any]:
        checkpoint = torch.load(self.output_dir / BEST_MODEL_NAME, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
