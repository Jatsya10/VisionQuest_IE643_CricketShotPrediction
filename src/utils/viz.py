"""Qualitative visualisation: pose overlay on frames, confidence-vs-time curve."""
from __future__ import annotations
import numpy as np

from src.models.pose_branch import POSE_EDGES


def draw_pose_on_frame(frame_rgb: np.ndarray, kp_xy: np.ndarray, color=(179, 35, 28)):
    """frame_rgb: (H,W,3) uint8. kp_xy: (33,2) in [0,1] image coords."""
    import cv2
    img = frame_rgb.copy()
    H, W = img.shape[:2]
    pts = (kp_xy * np.array([W, H])).astype(int)
    for i, j in POSE_EDGES:
        if (kp_xy[i] > 0).any() and (kp_xy[j] > 0).any():
            cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 2)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (20, 61, 43), -1)
    return img


def confidence_curve(per_ratio_probs: dict, classes, out_path, true_label=None):
    """
    per_ratio_probs: {ratio: prob_vector(np.ndarray[C])}.
    Plots each class's predicted probability as the observed fraction grows,
    i.e. 'how the model's belief evolves as the shot unfolds'.
    """
    import matplotlib.pyplot as plt
    ratios = sorted(per_ratio_probs)
    P = np.stack([per_ratio_probs[r] for r in ratios])   # (R,C)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in range(len(classes)):
        lw = 2.5 if (true_label is not None and c == true_label) else 1.0
        ax.plot([int(r * 100) for r in ratios], P[:, c], label=classes[c], linewidth=lw)
    ax.set_xlabel("Observed clip fraction (%)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Prediction confidence as the shot unfolds")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
