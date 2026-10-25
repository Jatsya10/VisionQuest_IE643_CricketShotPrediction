"""Evaluation metrics: top-k accuracy, per-class F1, MRR, earliness, confusion."""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, ks=(1, 3, 5)) -> dict:
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)   # (B,maxk)
    correct = pred.eq(target.view(-1, 1))
    out = {}
    for k in ks:
        kk = min(k, maxk)
        out[f"top{k}"] = correct[:, :kk].any(1).float().mean().item()
    return out


@torch.no_grad()
def mean_reciprocal_rank(logits: torch.Tensor, target: torch.Tensor) -> float:
    order = logits.argsort(dim=1, descending=True)                 # (B,C)
    ranks = (order == target.view(-1, 1)).float().argmax(1) + 1    # 1-indexed
    return (1.0 / ranks.float()).mean().item()


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    return macro, {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()}


def confusion(y_true, y_pred, num_classes):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def earliness_score(acc_by_ratio: dict, margin: float = 0.03) -> float:
    """
    Smallest observation ratio whose top-1 accuracy is within `margin` of the
    full-clip (ratio=1.0) top-1 accuracy. Lower is better (earlier reliable call).
    Returns 1.0 if no earlier ratio qualifies.
    """
    full = acc_by_ratio.get(1.0, max(acc_by_ratio.values()))
    qualifying = [r for r, a in sorted(acc_by_ratio.items()) if a >= full - margin]
    return min(qualifying) if qualifying else 1.0
