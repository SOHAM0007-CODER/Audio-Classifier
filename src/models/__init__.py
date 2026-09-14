"""
Model architectures for the Multi-Task Audio Spectrogram Transformer (AST) classifier.
"""

from .multitask_ast import ClassificationHead, MultiTaskAST

__all__ = ["ClassificationHead", "MultiTaskAST"]
