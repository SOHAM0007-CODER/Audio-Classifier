"""
src/models/multitask_ast.py

Multi-Task Audio Spectrogram Transformer (AST) classifier.
A single AST backbone (MIT/ast-finetuned-audioset-10-10-0.4593) produces a shared pooled
representation (mean of the CLS and distillation tokens, hidden size 768), which is routed to
one of two task-specific heads based on the sample's domain:
  - domain 0 (music):         music_head -> 10 GTZAN genres
  - domain 1 (environmental): env_head   -> 50 ESC-50 categories
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTConfig, ASTModel

from ..dataset import DEFAULT_AST_MODEL

MUSIC_DOMAIN = 0
ENV_DOMAIN = 1
NUM_MUSIC_CLASSES = 10
NUM_ENV_CLASSES = 50


class ClassificationHead(nn.Module):
    """
    LayerNorm -> Linear head (mirrors the original ASTMLPHead).
    If hidden_dim is given, becomes LayerNorm -> Linear -> GELU -> Dropout -> Linear.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        layers = [nn.LayerNorm(in_dim, eps=layer_norm_eps)]
        if hidden_dim:
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        else:
            layers += [nn.Dropout(dropout), nn.Linear(in_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskAST(nn.Module):
    """
    Shared AST backbone with domain-routed music / environmental classification heads.

    forward() returns a dictionary:
      - 'loss':         Weighted combination of per-task losses (only if domain and label given).
      - 'music_loss':   CrossEntropy over domain-0 samples, or None if the batch has none.
      - 'env_loss':     CrossEntropy over domain-1 samples, or None if the batch has none.
      - 'music_logits': (N_music, 10) logits for domain-0 samples, or (B, 10) for the full batch.
      - 'env_logits':   (N_env, 50) logits for domain-1 samples, or (B, 50) for the full batch.
      - 'music_indices' / 'env_indices': Batch positions the routed logits correspond to.
      - 'pooled_output': (B, hidden_size) shared representation.
    """

    def __init__(
        self,
        pretrained_model_name: str = DEFAULT_AST_MODEL,
        num_music_classes: int = NUM_MUSIC_CLASSES,
        num_env_classes: int = NUM_ENV_CLASSES,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.1,
        music_loss_weight: float = 1.0,
        env_loss_weight: float = 1.0,
        backbone_config: Optional[ASTConfig] = None,
    ):
        """
        Args:
            pretrained_model_name: HuggingFace checkpoint used to initialize the AST backbone.
            num_music_classes: Number of GTZAN genres (default 10).
            num_env_classes: Number of ESC-50 categories (default 50).
            head_hidden_dim: If set, heads become 2-layer MLPs with this hidden width.
            head_dropout: Dropout probability inside the heads.
            music_loss_weight: Weight of the music CrossEntropy term in the combined loss.
            env_loss_weight: Weight of the environmental CrossEntropy term in the combined loss.
            backbone_config: If given, builds a randomly initialized backbone from this config
                instead of loading pretrained weights (useful for tests).
        """
        super().__init__()
        if backbone_config is not None:
            self.backbone = ASTModel(backbone_config)
        else:
            self.backbone = ASTModel.from_pretrained(pretrained_model_name)

        config = self.backbone.config
        hidden_size = config.hidden_size
        self.music_head = ClassificationHead(
            hidden_size, num_music_classes, head_hidden_dim, head_dropout, config.layer_norm_eps
        )
        self.env_head = ClassificationHead(
            hidden_size, num_env_classes, head_hidden_dim, head_dropout, config.layer_norm_eps
        )

        self.num_music_classes = num_music_classes
        self.num_env_classes = num_env_classes
        self.music_loss_weight = music_loss_weight
        self.env_loss_weight = env_loss_weight

    def forward(
        self,
        input_values: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        return_full_batch_logits: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            input_values: (B, 1024, 128) log-mel spectrograms from the AST feature extractor.
            domain: Optional (B,) LongTensor of domain IDs (0 = music, 1 = environmental).
            label: Optional (B,) LongTensor of per-domain class IDs. Requires `domain`.
            return_full_batch_logits: If True, both heads score every sample in the batch.
                Always the case when `domain` is None (e.g. inference with unknown domain).
        """
        if label is not None and domain is None:
            raise ValueError("`label` was provided without `domain`; cannot route samples to heads.")

        pooled_output = self.backbone(input_values=input_values).pooler_output  # (B, hidden)
        batch_size = pooled_output.shape[0]
        all_indices = torch.arange(batch_size, device=pooled_output.device)

        outputs: Dict[str, Optional[torch.Tensor]] = {
            "loss": None,
            "music_loss": None,
            "env_loss": None,
            "pooled_output": pooled_output,
        }

        if domain is None or return_full_batch_logits:
            outputs["music_logits"] = self.music_head(pooled_output)
            outputs["env_logits"] = self.env_head(pooled_output)
            outputs["music_indices"] = all_indices
            outputs["env_indices"] = all_indices
            if domain is None:
                return outputs

        domain = domain.to(pooled_output.device).view(-1)
        music_mask = domain == MUSIC_DOMAIN
        env_mask = domain == ENV_DOMAIN
        if not bool((music_mask | env_mask).all()):
            raise ValueError(f"`domain` must contain only {MUSIC_DOMAIN} or {ENV_DOMAIN}.")

        if return_full_batch_logits:
            music_logits = outputs["music_logits"][music_mask]
            env_logits = outputs["env_logits"][env_mask]
        else:
            music_logits = self.music_head(pooled_output[music_mask])
            env_logits = self.env_head(pooled_output[env_mask])
            outputs["music_logits"] = music_logits
            outputs["env_logits"] = env_logits
            outputs["music_indices"] = all_indices[music_mask]
            outputs["env_indices"] = all_indices[env_mask]

        if label is None:
            return outputs

        label = label.to(pooled_output.device).view(-1)
        weighted_losses = []
        total_weight = 0.0

        # Skip absent domains: CrossEntropy over zero samples returns NaN.
        if music_logits.shape[0] > 0:
            music_loss = F.cross_entropy(music_logits, label[music_mask])
            outputs["music_loss"] = music_loss
            weighted_losses.append(self.music_loss_weight * music_loss)
            total_weight += self.music_loss_weight
        if env_logits.shape[0] > 0:
            env_loss = F.cross_entropy(env_logits, label[env_mask])
            outputs["env_loss"] = env_loss
            weighted_losses.append(self.env_loss_weight * env_loss)
            total_weight += self.env_loss_weight

        # Normalize by the weights of tasks present so single-domain batches keep the same scale.
        outputs["loss"] = torch.stack(weighted_losses).sum() / total_weight
        return outputs
