"""
src/train.py

Command-line entry point for training the Multi-Task AST classifier on GTZAN + ESC-50.
Every TrainConfig field (src/trainer.py) is a flag; an optional YAML file supplies values
that command-line flags override.

Usage (from the repository root):
    python -m src.train
    python -m src.train --batch_size 4 --grad_accum_steps 2 --gradient_checkpointing
    python -m src.train --config my_config.yaml --epochs 10 --no-fp16
    python -m src.train --resume auto
"""

import argparse
import json
import logging
import sys
import typing
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoFeatureExtractor

from .dataset import AudioDataset, create_dataloader
from .models.multitask_ast import MultiTaskAST
from .trainer import BEST_MODEL_NAME, LATEST_CHECKPOINT_NAME, TrainConfig, Trainer, set_seed

logger = logging.getLogger("src.train")


def _is_optional(field_type: Any) -> bool:
    return typing.get_origin(field_type) is typing.Union and type(None) in typing.get_args(field_type)


def _base_type(field_type: Any) -> Any:
    """Unwraps Optional[X] to X."""
    if _is_optional(field_type):
        return next(arg for arg in typing.get_args(field_type) if arg is not type(None))
    return field_type


def _coerce(name: str, field_type: Any, value: Any) -> Any:
    """Converts a YAML value to the field's type (PyYAML reads e.g. `1e-4` as a string)."""
    if value is None:
        if _is_optional(field_type):
            return None
        raise ValueError(f"Config field '{name}' cannot be null.")
    base = _base_type(field_type)
    if base is bool:
        if not isinstance(value, bool):
            raise ValueError(f"Config field '{name}' must be true or false, got {value!r}.")
        return value
    try:
        return base(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Config field '{name}' expects {base.__name__}, got {value!r}.") from e


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Multi-Task AST classifier (GTZAN + ESC-50).")
    parser.add_argument(
        "--config", type=str, default=None, help="YAML file of config fields; command-line flags take precedence."
    )
    for f in fields(TrainConfig):
        flags = [f"--{f.name}", *f.metadata.get("aliases", ())]
        help_text = f"{f.metadata['help']} (default: {f.default})"
        base = _base_type(f.type)
        # SUPPRESS keeps unset flags out of the namespace so they don't override YAML values.
        if base is bool:
            parser.add_argument(
                *flags, dest=f.name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=help_text
            )
        else:
            parser.add_argument(
                *flags, dest=f.name, type=base, default=argparse.SUPPRESS, metavar=base.__name__.upper(), help=help_text
            )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    """Builds a TrainConfig from dataclass defaults, then the optional YAML file, then CLI flags."""
    args = vars(build_arg_parser().parse_args(argv))
    config_path = args.pop("config")
    values: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            file_values = yaml.safe_load(f) or {}
        if not isinstance(file_values, dict):
            raise ValueError(f"Config file {config_path} must contain a mapping of field names to values.")
        field_types = {f.name: f.type for f in fields(TrainConfig)}
        unknown = sorted(set(file_values) - set(field_types))
        if unknown:
            raise ValueError(f"Unknown keys in {config_path}: {unknown}")
        values.update({key: _coerce(key, field_types[key], value) for key, value in file_values.items()})
    values.update(args)
    return TrainConfig(**values)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
        force=True,
    )
    # Hugging Face Hub logs every HTTP request at INFO level.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _resolve_resume_path(config: TrainConfig) -> Optional[Path]:
    if not config.resume:
        return None
    if config.resume == "auto":
        path = Path(config.output_dir) / LATEST_CHECKPOINT_NAME
        return path if path.exists() else None
    path = Path(config.resume)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    config = parse_config(argv)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
    logger.info("Config:\n%s", json.dumps(asdict(config), indent=2))
    set_seed(config.seed)

    # Share one feature extractor across splits instead of loading it per dataset.
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.pretrained_model_name)

    def make_loader(manifest: str, shuffle: bool):
        dataset = AudioDataset(manifest, feature_extractor=feature_extractor)
        return create_dataloader(
            dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers
        )

    train_loader = make_loader(config.train_manifest, shuffle=True)
    val_loader = make_loader(config.val_manifest, shuffle=False)
    logger.info("Train samples: %d | Val samples: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = MultiTaskAST(
        pretrained_model_name=config.pretrained_model_name,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        music_loss_weight=config.music_loss_weight,
        env_loss_weight=config.env_loss_weight,
    )
    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()

    trainer = Trainer(model, train_loader, val_loader, config)
    resume_path = _resolve_resume_path(config)
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)

    results: Dict[str, Any] = {"best_combined_metric": None, "test_metrics": None}
    with logging_redirect_tqdm():
        trainer.fit()
        results["best_combined_metric"] = trainer.best_metric

        if config.eval_test and Path(config.test_manifest).exists() and (output_dir / BEST_MODEL_NAME).exists():
            test_loader = make_loader(config.test_manifest, shuffle=False)
            best = trainer.load_best_model()
            test_metrics = trainer.evaluate(test_loader, config.max_eval_batches, desc="test")
            test_metrics["best_epoch"] = best["epoch"] + 1
            logger.info(
                "Test (best model, epoch %d) | music acc %.4f f1 %.4f | env acc %.4f f1 %.4f | combined %.4f",
                test_metrics["best_epoch"],
                test_metrics["music_acc"],
                test_metrics["music_macro_f1"],
                test_metrics["env_acc"],
                test_metrics["env_macro_f1"],
                test_metrics["combined_metric"],
            )
            with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
            results["test_metrics"] = test_metrics
    return results


if __name__ == "__main__":
    main()
