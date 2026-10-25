"""
Pose branch for Setting B (player-specific, likely/unlikely shot scoring).

Architecture: a compact ST-GCN (spatial-temporal graph conv network, Yan et al.
2018) over the MediaPipe 33-joint skeleton, plus an OPTIONAL player embedding
concatenated before the classifier. Output is per-class logits; at inference we
apply a sigmoid and threshold to get likely/unlikely, OR softmax for ranking.

Graph: edges follow the MediaPipe BlazePose skeleton connectivity. The
normalised adjacency A (with self-loops) is fixed (not learned) in this compact
version; a learnable edge-importance mask is included as in the original paper.

Why ST-GCN: the skeleton is naturally a graph; ST-GCN exploits both the spatial
joint structure (within a frame) and temporal dynamics (across frames), which is
exactly the backlift/trigger kinematics we want to read early. Far fewer params
than the video branch, so it trains fast and resists overfitting on pose.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe BlazePose 33-landmark skeleton edges (subset covering the body).
POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),      # arms + shoulders
    (11, 23), (12, 24), (23, 24),                          # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                # legs
    (27, 29), (29, 31), (28, 30), (30, 32),                # feet
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (0, 11), (0, 12),                                      # nose -> shoulders
]
NUM_JOINTS = 33


def build_adjacency(num_joints=NUM_JOINTS, edges=POSE_EDGES) -> torch.Tensor:
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # symmetric normalisation D^-1/2 A D^-1/2
    deg = A.sum(1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-6))
    return torch.from_numpy(dinv @ A @ dinv)


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, dropout=0.4):
        super().__init__()
        self.register_buffer("A", A)
        self.edge_imp = nn.Parameter(torch.ones_like(A))  # learnable edge mask
        self.gconv = nn.Conv2d(in_c, out_c, kernel_size=1)             # spatial
        self.tconv = nn.Sequential(                                    # temporal
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_c), nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if (in_c == out_c and stride == 1)
            else nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=(stride, 1)), nn.BatchNorm2d(out_c))
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                      # x: (B,C,T,V)
        res = self.residual(x)
        A = self.A * self.edge_imp
        x = torch.einsum("bctv,vw->bctw", x, A)  # spatial graph conv
        x = self.gconv(x)
        x = self.tconv(x)
        return self.relu(x + res)


class PoseSTGCN(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden=128, num_layers=3,
                 dropout=0.4, num_players=1, player_dim=16):
        super().__init__()
        A = build_adjacency()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        chans = [in_channels] + [hidden] * num_layers
        strides = [1] + [2 if l == num_layers - 1 else 1 for l in range(num_layers - 1)] if num_layers > 1 else [1]
        strides = ([1] * num_layers)
        self.blocks = nn.ModuleList([
            STGCNBlock(chans[l], chans[l + 1], A, stride=strides[l], dropout=dropout)
            for l in range(num_layers)
        ])
        self.use_player = num_players > 1 and player_dim > 0
        if self.use_player:
            self.player_emb = nn.Embedding(num_players, player_dim)
        clf_in = hidden + (player_dim if self.use_player else 0)
        self.classifier = nn.Linear(clf_in, num_classes)

    def forward(self, x, player_idx=None):     # x: (B,T,V,C)
        B, T, V, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()               # (B,C,T,V)
        x = x.view(B, C * V, T)
        x = self.data_bn(x).view(B, C, V, T).permute(0, 1, 3, 2).contiguous()  # (B,C,T,V)
        for blk in self.blocks:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1).view(B, -1)          # (B,hidden)
        if self.use_player:
            if player_idx is None:
                player_idx = torch.zeros(B, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.player_emb(player_idx)], dim=1)
        return self.classifier(x)                            # (B,num_classes) logits


class PoseGRU(nn.Module):
    """Fallback: flatten joints -> GRU. Used if cfg.model.pose.arch == 'gru'."""
    def __init__(self, num_classes, in_dim=NUM_JOINTS * 3, hidden=128, num_layers=2,
                 dropout=0.4, num_players=1, player_dim=16):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.use_player = num_players > 1 and player_dim > 0
        if self.use_player:
            self.player_emb = nn.Embedding(num_players, player_dim)
        self.head = nn.Linear(hidden + (player_dim if self.use_player else 0), num_classes)

    def forward(self, x, player_idx=None):     # x: (B,T,V,C)
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)
        out, _ = self.gru(x)
        h = out[:, -1]
        if self.use_player:
            if player_idx is None:
                player_idx = torch.zeros(B, dtype=torch.long, device=x.device)
            h = torch.cat([h, self.player_emb(player_idx)], dim=1)
        return self.head(h)


def build_pose_model(cfg, num_classes: int, num_players: int = 1) -> nn.Module:
    p = cfg.model.pose
    if p.arch == "stgcn":
        return PoseSTGCN(num_classes, 3, p.hidden, p.num_layers, p.dropout,
                         num_players, p.player_embedding_dim)
    if p.arch == "gru":
        return PoseGRU(num_classes, NUM_JOINTS * 3, p.hidden, p.num_layers, p.dropout,
                       num_players, p.player_embedding_dim)
    raise ValueError(f"unknown pose arch {p.arch}")
