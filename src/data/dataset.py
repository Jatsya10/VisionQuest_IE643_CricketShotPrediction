"""
Torch Datasets for the two branches, both observation-ratio aware.

Partial observation is the heart of the project. A clip is sampled into
`num_frames` ordered frames. An observation ratio g in (0,1] means the model
is only allowed to see the FIRST round(g*num_frames) frames — i.e. the early
part of the shot. Frames after that are masked:
  * video branch: the unseen tail is replaced by a repeat of the last seen
    frame (keeps tensor shape fixed for the 3D-CNN) — the model gets no future
    information but the temporal length the backbone expects is preserved.
  * pose branch: the unseen tail is zeroed and excluded via a length, so the
    sequence model (ST-GCN/GRU) only aggregates over seen steps.

Training uses a *random* ratio drawn from cfg.data.train_ratios each __getitem__
so one model generalises across early-prediction horizons (cheaper and stronger
than training a separate model per ratio). Evaluation pins a fixed ratio.
"""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.common import load_json

# ImageNet/Kinetics normalisation stats for the pretrained R3D-18 stem.
_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def seen_count(num_frames: int, ratio: float) -> int:
    return max(1, int(round(ratio * num_frames)))


class _Base(Dataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        index = load_json(cfg.paths.index_file)
        splits = load_json(cfg.paths.splits_file)
        self.ids = [c for c in splits[split] if c in index]
        self.index = index
        self.classes = list(cfg.classes)
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_frames = cfg.data.num_frames
        self.train_ratios = list(cfg.data.train_ratios)
        self.is_train = split == "train"
        self.fixed_ratio = None  # eval pins this

    def __len__(self):
        return len(self.ids)

    def _ratio(self):
        if self.fixed_ratio is not None:
            return self.fixed_ratio
        return random.choice(self.train_ratios) if self.is_train else 1.0

    def label_of(self, cid):
        return self.cls2idx[self.index[cid]["label"]]


class VideoClipDataset(_Base):
    """Returns (clip_tensor (3,T,H,W), label, ratio)."""

    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.frames_dir = Path(cfg.paths.frames_dir)
        self.img_size = cfg.data.img_size

    def _augment(self, frames):
        # light train-time augmentation: horizontal flip (shot handedness is
        # preserved as a cue, so flip is applied to the whole clip consistently)
        if self.is_train and random.random() < 0.5:
            frames = frames[:, :, ::-1, :].copy()
        return frames

    def __getitem__(self, i):
        cid = self.ids[i]
        arr = np.load(self.frames_dir / f"{cid}.npy")  # (T,H,W,3) uint8
        arr = self._augment(arr)
        ratio = self._ratio()
        k = seen_count(self.num_frames, ratio)
        if k < self.num_frames:
            # mask the future: repeat last seen frame to keep length T
            tail = np.repeat(arr[k - 1:k], self.num_frames - k, axis=0)
            arr = np.concatenate([arr[:k], tail], axis=0)
        x = arr.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD                 # (T,H,W,3)
        x = torch.from_numpy(x).permute(3, 0, 1, 2).contiguous()  # (3,T,H,W)
        return x, self.label_of(cid), float(ratio)


class PoseSeqDataset(_Base):
    """Returns (pose (T,33,3), length_seen, label, player_idx, ratio)."""

    def __init__(self, cfg, split, player2idx=None):
        super().__init__(cfg, split)
        self.pose_dir = Path(cfg.paths.pose_dir)
        self.player2idx = player2idx or {"unknown": 0}

    def __getitem__(self, i):
        cid = self.ids[i]
        kp = np.load(self.pose_dir / f"{cid}.npy").astype(np.float32)  # (T,33,3)
        ratio = self._ratio()
        k = seen_count(self.num_frames, ratio)
        seq = np.zeros_like(kp)
        seq[:k] = kp[:k]                       # zero the unseen tail
        player = self.index[cid]["player"]
        pidx = self.player2idx.get(player, 0)
        return (torch.from_numpy(seq), int(k),
                self.label_of(cid), int(pidx), float(ratio))


def build_player_vocab(cfg):
    """Players with >= min_clips_per_player get their own id; rest -> 'unknown'(0)."""
    from collections import Counter
    index = load_json(cfg.paths.index_file)
    splits = load_json(cfg.paths.splits_file)
    train_ids = set(splits["train"])
    cnt = Counter(index[c]["player"] for c in index if c in train_ids)
    vocab = {"unknown": 0}
    for p, n in cnt.items():
        if p != "unknown" and n >= cfg.data.min_clips_per_player:
            vocab[p] = len(vocab)
    return vocab
