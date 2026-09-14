"""
src/evaluate.py

Evaluates a trained Multi-Task AST checkpoint on a manifest (default: the test split) and reports
loss, per-task accuracy and macro-F1, and combined_metric = 0.5 * music_acc + 0.5 * env_acc.

Usage (from the repository root):
    python -m src.evaluate --checkpoint checkpoints/cloud_run/best_model.pt --fp16
    python -m src.evaluate --checkpoint checkpoints/best_model.pt --manifest data/manifests/val.csv --output val_metrics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoFeatureExtractor

from .dataset import AudioDataset, create_dataloader
from .predict import DEFAULT_CHECKPOINT, TASK_TITLES, load_checkpoint_model
from .trainer import TrainConfig, evaluate_model

DEFAULT_MANIFEST = "data/manifests/test.csv"


def format_metrics(metrics: Dict[str, Any]) -> str:
    lines = [
        f"Checkpoint: {metrics['checkpoint']}",
        f"Manifest:   {metrics['manifest']}",
        f"Loss:       {metrics['loss']:.4f}",
    ]
    width = max(len(title) for title in TASK_TITLES.values())
    for task, title in TASK_TITLES.items():
        lines.append(
            f"{title:<{width}}  acc {metrics[f'{task}_acc']:.4f} | macro-F1 {metrics[f'{task}_macro_f1']:.4f} "
            f"| n={metrics[f'{task}_num_samples']}"
        )
    lines.append(f"Combined metric (0.5 * music_acc + 0.5 * env_acc): {metrics['combined_metric']:.4f}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Multi-Task AST checkpoint on a manifest.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help=f"best_model.pt (default: {DEFAULT_CHECKPOINT}).")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"Manifest CSV (default: {DEFAULT_MANIFEST}).")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size (default: 16).")
    parser.add_argument(
        "--num_workers", type=int, default=TrainConfig().num_workers, help="DataLoader workers (default: 0 on Windows, else 4)."
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available).")
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=False, help="float16 autocast on CUDA (default: off)."
    )
    parser.add_argument("--max_batches", type=int, default=None, help="Limit evaluation batches (debugging).")
    parser.add_argument("--output", default=None, help="Optional JSON file to write the metrics to.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)

    for name, path in (("Checkpoint", args.checkpoint), ("Manifest", args.manifest)):
        if not Path(path).is_file():
            print(f"error: {name} not found: {path}", file=sys.stderr)
            return 1

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pretrained_model_name = load_checkpoint_model(args.checkpoint)
    model.to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
    loader = create_dataloader(
        AudioDataset(args.manifest, feature_extractor=feature_extractor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate_model(
        model,
        loader,
        device,
        use_amp=args.fp16 and device.type == "cuda",
        max_batches=args.max_batches,
        desc=Path(args.manifest).stem,
    )
    metrics = {"checkpoint": str(args.checkpoint), "manifest": str(args.manifest), **metrics}

    print(format_metrics(metrics))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
