"""
src/predict.py

Inference for the Multi-Task AST classifier.

Audio of any sample rate / channel count is converted to 16 kHz mono and split into 5.0 s windows
(a trailing partial window is zero-padded, and kept if it is at least 2.5 s long or is the only window).
Each window goes through the AST feature extractor and model, and the softmax probabilities of the
requested head(s) are averaged over the windows.

Usage (from the repository root):
    python -m src.predict --audio clip.wav
    python -m src.predict --audio song.wav --domain music --top_k 5
    python -m src.predict --audio dog.wav --domain env --json
    python -m src.predict --audio clip.wav --checkpoint none   # pretrained backbone, untrained heads
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import ASTConfig, AutoFeatureExtractor

from .dataset import DEFAULT_AST_MODEL, DEFAULT_DURATION, load_audio
from .models.multitask_ast import ENV_DOMAIN, MUSIC_DOMAIN, MultiTaskAST

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"
DEFAULT_MUSIC_CLASSES = "data/manifests/music_classes.json"
DEFAULT_ENV_CLASSES = "data/manifests/env_classes.json"
DEFAULT_TOP_K = 3
DEFAULT_MAX_CHUNKS = 6  # first 30 s
TASK_TITLES = {"music": "Music genre (GTZAN)", "env": "Environmental sound (ESC-50)"}


def parse_domain(domain: Union[str, int, None]) -> List[str]:
    """Maps music/0 -> ['music'], env/1 -> ['env'], and auto/both/None -> ['music', 'env']."""
    if domain is None:
        return ["music", "env"]
    key = str(domain).strip().lower()
    if key in ("music", str(MUSIC_DOMAIN)):
        return ["music"]
    if key in ("env", "environmental", str(ENV_DOMAIN)):
        return ["env"]
    if key in ("auto", "both"):
        return ["music", "env"]
    raise ValueError(f"Unknown domain {domain!r}; expected music/0, env/1 or auto/both.")


def load_class_names(path: Union[str, Path], num_classes: int) -> List[str]:
    """Reads a {class_name: index} JSON mapping into a list ordered by index."""
    path = Path(path)
    if not path.is_file():
        logger.warning("Class mapping %s not found; using generic class names.", path)
        return [f"class_{i}" for i in range(num_classes)]

    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    names: List[Optional[str]] = [None] * num_classes
    for name, index in mapping.items():
        index = int(index)
        if not 0 <= index < num_classes or names[index] is not None:
            raise ValueError(
                f"{path}: invalid or duplicate class index {index} for '{name}' (expected 0-{num_classes - 1})."
            )
        names[index] = name
    missing = [i for i, name in enumerate(names) if name is None]
    if missing:
        raise ValueError(f"{path}: no class name for indices {missing}.")
    return names


def split_into_chunks(waveform: np.ndarray, chunk_samples: int) -> List[np.ndarray]:
    """
    Splits a waveform into non-overlapping chunk_samples windows. The trailing partial window is
    zero-padded and kept only if it covers at least half a window or is the only window, so a short
    tail of padding silence does not dilute the averaged prediction.
    """
    if len(waveform) == 0:
        raise ValueError("Audio contains no samples.")
    num_full = len(waveform) // chunk_samples
    chunks = [waveform[i * chunk_samples : (i + 1) * chunk_samples] for i in range(num_full)]
    remainder = waveform[num_full * chunk_samples :]
    if len(remainder) > 0 and (num_full == 0 or len(remainder) >= chunk_samples // 2):
        chunks.append(np.pad(remainder, (0, chunk_samples - len(remainder))))
    return chunks


def load_checkpoint_model(checkpoint_path: Union[str, Path]) -> Tuple[MultiTaskAST, str]:
    """
    Rebuilds a MultiTaskAST from a best_model.pt written by src.train and loads its weights on CPU.
    Returns the model and the name of the pretrained AST checkpoint it was fine-tuned from, whose
    feature extractor must be used to prepare inputs.
    """
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_config = checkpoint_data.get("config", {})
    pretrained_model_name = train_config.get("pretrained_model_name", DEFAULT_AST_MODEL)
    if "backbone_config" in checkpoint_data:
        backbone_config = ASTConfig.from_dict(checkpoint_data["backbone_config"])
    else:
        backbone_config = ASTConfig.from_pretrained(pretrained_model_name)
    model = MultiTaskAST(head_hidden_dim=train_config.get("head_hidden_dim"), backbone_config=backbone_config)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    return model, pretrained_model_name


class AudioClassifier:
    """Wraps a MultiTaskAST model, its feature extractor and class names for clip-level prediction."""

    def __init__(
        self,
        model: MultiTaskAST,
        feature_extractor: Optional[Any] = None,
        music_classes: Union[str, Path] = DEFAULT_MUSIC_CLASSES,
        env_classes: Union[str, Path] = DEFAULT_ENV_CLASSES,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        max_chunks: Optional[int] = DEFAULT_MAX_CHUNKS,
        batch_size: int = 8,
    ):
        """
        Args:
            model: MultiTaskAST with loaded weights.
            feature_extractor: AST feature extractor (default: loaded from DEFAULT_AST_MODEL).
            music_classes / env_classes: {class_name: index} JSON mappings.
            device: 'cpu' or 'cuda' (default: cuda if available).
            checkpoint_path: Reported in results; None means untrained heads.
            max_chunks: Maximum number of 5.0 s windows averaged per clip (None = all).
            batch_size: Windows per forward pass.
        """
        if max_chunks is not None and max_chunks < 1:
            raise ValueError(f"max_chunks must be >= 1 or None, got {max_chunks}")
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        if feature_extractor is None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(DEFAULT_AST_MODEL)
        self.feature_extractor = feature_extractor
        self.target_sr = int(feature_extractor.sampling_rate)
        self.class_names = {
            "music": load_class_names(music_classes, model.num_music_classes),
            "env": load_class_names(env_classes, model.num_env_classes),
        }
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.max_chunks = max_chunks
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Optional[Union[str, Path]] = DEFAULT_CHECKPOINT,
        device: Optional[Union[str, torch.device]] = None,
        pretrained_model_name: str = DEFAULT_AST_MODEL,
        **kwargs: Any,
    ) -> "AudioClassifier":
        """
        Builds a classifier from a best_model.pt written by src.train.

        With checkpoint=None or 'none', or when the default checkpoint does not exist, falls back to the
        pretrained AST backbone with randomly initialized heads (only useful to test the pipeline).
        Remaining keyword arguments are passed to AudioClassifier.__init__.
        """
        path = None if checkpoint is None or str(checkpoint).lower() == "none" else Path(checkpoint)
        if path is not None and not path.is_file():
            if path != Path(DEFAULT_CHECKPOINT):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            logger.warning("Default checkpoint %s not found.", path)
            path = None

        if path is None:
            logger.warning("Using the pretrained AST backbone with untrained heads; predictions are not meaningful.")
            model = MultiTaskAST(pretrained_model_name=pretrained_model_name)
        else:
            model, pretrained_model_name = load_checkpoint_model(path)

        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        return cls(model, feature_extractor=feature_extractor, device=device, checkpoint_path=path, **kwargs)

    @torch.no_grad()
    def predict(
        self, audio_path: Union[str, Path], domain: Union[str, int, None] = None, top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Classifies one audio file.

        Returns a JSON-serializable dict with 'audio', 'checkpoint', 'duration_sec', 'chunk_sec',
        'num_chunks' and 'predictions': {task: [{'rank', 'label', 'index', 'confidence'}, ...]}
        for task 'music' and/or 'env', sorted by confidence.
        """
        tasks = parse_domain(domain)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = load_audio(audio_path, self.target_sr)
        chunks = split_into_chunks(waveform, int(round(self.target_sr * DEFAULT_DURATION)))
        if self.max_chunks is not None and len(chunks) > self.max_chunks:
            logger.info("%s: using the first %d of %d windows.", audio_path, self.max_chunks, len(chunks))
            chunks = chunks[: self.max_chunks]

        probs: Dict[str, List[torch.Tensor]] = {task: [] for task in tasks}
        for start in range(0, len(chunks), self.batch_size):
            features = self.feature_extractor(
                chunks[start : start + self.batch_size], sampling_rate=self.target_sr, return_tensors="pt"
            )["input_values"].to(self.device)
            outputs = self.model(features)
            for task in tasks:
                probs[task].append(outputs[f"{task}_logits"].float().softmax(dim=-1).cpu())

        predictions = {}
        for task in tasks:
            mean_probs = torch.cat(probs[task]).mean(dim=0)
            confidences, indices = mean_probs.topk(min(top_k, mean_probs.numel()))
            predictions[task] = [
                {
                    "rank": rank + 1,
                    "label": self.class_names[task][int(index)],
                    "index": int(index),
                    "confidence": float(confidence),
                }
                for rank, (confidence, index) in enumerate(zip(confidences, indices))
            ]

        return {
            "audio": str(audio_path),
            "checkpoint": self.checkpoint_path,
            "duration_sec": round(len(waveform) / self.target_sr, 3),
            "chunk_sec": DEFAULT_DURATION,
            "num_chunks": len(chunks),
            "predictions": predictions,
        }


def predict(
    audio_path: Union[str, Path],
    checkpoint: Optional[Union[str, Path]] = DEFAULT_CHECKPOINT,
    domain: Union[str, int, None] = None,
    top_k: int = DEFAULT_TOP_K,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    One-call prediction: loads the checkpoint and classifies audio_path (see AudioClassifier.predict).
    To classify many files, build AudioClassifier.from_checkpoint(...) once and call its predict().
    """
    classifier = AudioClassifier.from_checkpoint(checkpoint, device=device, **kwargs)
    return classifier.predict(audio_path, domain=domain, top_k=top_k)


def format_results(result: Dict[str, Any]) -> str:
    checkpoint = result["checkpoint"] or "none (pretrained backbone, untrained heads: predictions are not meaningful)"
    lines = [
        f"Audio:      {result['audio']} ({result['duration_sec']:.1f} s, "
        f"{result['num_chunks']} x {result['chunk_sec']:.1f} s window(s))",
        f"Checkpoint: {checkpoint}",
    ]
    for task, task_predictions in result["predictions"].items():
        width = max(len(p["label"]) for p in task_predictions)
        lines += ["", TASK_TITLES[task]]
        lines += [
            f"  {p['rank']:>2}. {p['label']:<{width}}  {100 * p['confidence']:6.2f}%" for p in task_predictions
        ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify an audio clip with the Multi-Task AST classifier.")
    parser.add_argument("--audio", required=True, help="Audio file to classify (any format / sample rate soundfile reads).")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"best_model.pt from src.train (default: {DEFAULT_CHECKPOINT}). 'none' uses the pretrained backbone "
        "with untrained heads; a missing default checkpoint falls back to the same.",
    )
    parser.add_argument(
        "--domain",
        default="auto",
        choices=["music", "0", "env", "1", "auto", "both"],
        help="music/0: GTZAN genres, env/1: ESC-50 sounds, auto/both: both heads (default: auto).",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help=f"Predictions per head (default: {DEFAULT_TOP_K}).")
    parser.add_argument("--music_classes", default=DEFAULT_MUSIC_CLASSES, help="Music {class_name: index} JSON.")
    parser.add_argument("--env_classes", default=DEFAULT_ENV_CLASSES, help="Environmental {class_name: index} JSON.")
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help=f"Maximum 5.0 s windows averaged (default: {DEFAULT_MAX_CHUNKS}, i.e. the first 30 s).",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available).")
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)

    try:
        if not Path(args.audio).is_file():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        classifier = AudioClassifier.from_checkpoint(
            args.checkpoint,
            device=args.device,
            music_classes=args.music_classes,
            env_classes=args.env_classes,
            max_chunks=args.max_chunks,
        )
        result = classifier.predict(args.audio, domain=args.domain, top_k=args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2) if args.json else format_results(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
