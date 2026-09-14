"""
tests/test_model.py

Unit tests for the Multi-Task AST model:
1. Forward pass on a mixed-domain dummy batch (4, 1024, 128) with domain routing.
2. Loss computation and backward pass (gradients reach both heads and the backbone).
3. Single-domain batches produce a finite loss and only update the relevant head.
4. Inference mode (no domain / labels) returns full-batch logits from both heads.
5. Pretrained checkpoint loads with hidden size 768 (skipped if unavailable).

Most tests use a 2-layer randomly initialized backbone with the real AST input geometry
(1024 frames x 128 mel bins, hidden size 768) so they run quickly without downloading weights.
"""

import pytest
import torch

from src.models.multitask_ast import MultiTaskAST

BATCH_SIZE = 4
MAX_LENGTH = 1024
NUM_MEL_BINS = 128


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    return MultiTaskAST(backbone_config=tiny_config)


@pytest.fixture
def mixed_batch():
    torch.manual_seed(0)
    return {
        "input_values": torch.randn(BATCH_SIZE, MAX_LENGTH, NUM_MEL_BINS),
        "domain": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "label": torch.tensor([3, 42, 9, 0], dtype=torch.long),
    }


def _grad_is_nonzero(param: torch.nn.Parameter) -> bool:
    return param.grad is not None and bool(param.grad.abs().sum() > 0)


def test_forward_mixed_domain_batch(model, mixed_batch):
    """Routes domain-0 samples to music_head and domain-1 samples to env_head."""
    model.train()
    outputs = model(**mixed_batch)

    assert outputs["music_logits"].shape == (2, 10)
    assert outputs["env_logits"].shape == (2, 50)
    assert outputs["music_indices"].tolist() == [0, 2]
    assert outputs["env_indices"].tolist() == [1, 3]
    assert outputs["pooled_output"].shape == (BATCH_SIZE, 768)

    loss = outputs["loss"]
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    assert torch.isfinite(outputs["music_loss"]) and torch.isfinite(outputs["env_loss"])
    # Equal task weights -> combined loss is the mean of both task losses.
    expected = (outputs["music_loss"] + outputs["env_loss"]) / 2
    assert torch.allclose(loss, expected)


def test_backward_gradient_flow(model, mixed_batch):
    """Gradients flow through both heads and every part of the backbone."""
    model.train()
    model.zero_grad()
    outputs = model(**mixed_batch)
    outputs["loss"].backward()

    for head in (model.music_head, model.env_head):
        for name, param in head.named_parameters():
            assert _grad_is_nonzero(param), f"No gradient in head parameter {name}"

    backbone = model.backbone
    assert _grad_is_nonzero(backbone.embeddings.patch_embeddings.projection.weight)
    assert _grad_is_nonzero(backbone.layers[0].attention.q_proj.weight)
    assert _grad_is_nonzero(backbone.layers[-1].mlp.fc2.weight)
    assert _grad_is_nonzero(backbone.layernorm.weight)
    params_without_grad = [n for n, p in backbone.named_parameters() if p.grad is None]
    assert not params_without_grad, f"Backbone parameters without gradient: {params_without_grad}"


def test_single_domain_batch(model):
    """A batch with only one domain yields a finite loss and leaves the other head untouched."""
    model.train()
    model.zero_grad()
    torch.manual_seed(1)
    input_values = torch.randn(BATCH_SIZE, MAX_LENGTH, NUM_MEL_BINS)
    domain = torch.ones(BATCH_SIZE, dtype=torch.long)
    label = torch.tensor([0, 17, 33, 49], dtype=torch.long)

    outputs = model(input_values, domain=domain, label=label)
    assert outputs["music_logits"].shape == (0, 10)
    assert outputs["env_logits"].shape == (BATCH_SIZE, 50)
    assert outputs["music_loss"] is None
    assert torch.isfinite(outputs["loss"])
    assert torch.allclose(outputs["loss"], outputs["env_loss"])

    outputs["loss"].backward()
    assert all(_grad_is_nonzero(p) for p in model.env_head.parameters())
    assert all(p.grad is None or bool(p.grad.abs().sum() == 0) for p in model.music_head.parameters())


def test_inference_mode_without_labels(model, mixed_batch):
    """Without domain/labels, both heads score the full batch and no loss is computed."""
    model.eval()
    with torch.no_grad():
        outputs = model(mixed_batch["input_values"])

    assert outputs["loss"] is None
    assert outputs["music_logits"].shape == (BATCH_SIZE, 10)
    assert outputs["env_logits"].shape == (BATCH_SIZE, 50)

    # Domain-aware inference: routed logits match the corresponding full-batch rows.
    with torch.no_grad():
        routed = model(mixed_batch["input_values"], domain=mixed_batch["domain"])
    assert routed["loss"] is None
    assert torch.allclose(routed["music_logits"], outputs["music_logits"][[0, 2]], atol=1e-5)
    assert torch.allclose(routed["env_logits"], outputs["env_logits"][[1, 3]], atol=1e-5)


def test_full_batch_logits_with_labels(model, mixed_batch):
    """return_full_batch_logits keeps (B, C) logits while still computing the routed loss."""
    model.eval()
    with torch.no_grad():
        full = model(**mixed_batch, return_full_batch_logits=True)
        routed = model(**mixed_batch)
    assert full["music_logits"].shape == (BATCH_SIZE, 10)
    assert full["env_logits"].shape == (BATCH_SIZE, 50)
    assert torch.allclose(full["loss"], routed["loss"], atol=1e-5)


def test_label_without_domain_raises(model, mixed_batch):
    with pytest.raises(ValueError):
        model(mixed_batch["input_values"], label=mixed_batch["label"])


def test_pretrained_backbone_loads():
    """Loads the real MIT AST checkpoint and runs inference on the dummy batch shape."""
    try:
        model = MultiTaskAST()
    except Exception as e:  # offline / no cached weights
        pytest.skip(f"Pretrained AST checkpoint unavailable: {e}")

    assert model.backbone.config.hidden_size == 768
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(BATCH_SIZE, MAX_LENGTH, NUM_MEL_BINS))
    assert outputs["music_logits"].shape == (BATCH_SIZE, 10)
    assert outputs["env_logits"].shape == (BATCH_SIZE, 50)
