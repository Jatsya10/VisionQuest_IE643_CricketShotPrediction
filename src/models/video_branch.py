"""
Video branch for Setting A (early Top-k shot prediction).

Primary backbone: R3D-18 (ResNet-18 3D) pretrained on Kinetics-400 from
torchvision. It is a well-established spatiotemporal CNN that is light enough
to fine-tune on a single Colab T4 in the available time, unlike SlowFast.
We swap its final FC for our K-class head.

Alternative backbone: CNN-GRU (per-frame ResNet-18 2D features -> GRU),
following the CricShotClassify precedent. Selected via cfg.model.video.backbone.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class R3DVideoNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        try:
            self.backbone = r3d_18(weights=weights)
        except Exception as e:  # offline / no network for pretrained download
            print(f"[warn] could not load pretrained r3d_18 ({e}); using random init")
            self.backbone = r3d_18(weights=None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, num_classes),
        )

    def forward(self, x):            # x: (B,3,T,H,W)
        feat = self.backbone(x)      # (B, in_feat)
        return self.head(feat)       # (B, num_classes) logits


class CNNGRUVideoNet(nn.Module):
    """Per-frame 2D CNN features -> GRU -> classifier (alternative backbone)."""
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5,
                 hidden: int = 256):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            cnn = resnet18(weights=weights)
        except Exception as e:
            print(f"[warn] could not load pretrained resnet18 ({e}); using random init")
            cnn = resnet18(weights=None)
        self.feat_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.gru = nn.GRU(self.feat_dim, hidden, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x):            # x: (B,3,T,H,W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        f = self.cnn(x).view(B, T, self.feat_dim)   # (B,T,feat)
        out, _ = self.gru(f)
        return self.head(out[:, -1])                # last step logits


def build_video_model(cfg, num_classes: int) -> nn.Module:
    m = cfg.model.video
    if m.backbone == "r3d_18":
        return R3DVideoNet(num_classes, m.pretrained, m.dropout)
    if m.backbone == "cnn_gru":
        return CNNGRUVideoNet(num_classes, m.pretrained, m.dropout)
    raise ValueError(f"unknown video backbone {m.backbone}")
