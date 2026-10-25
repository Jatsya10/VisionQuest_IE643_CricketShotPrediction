"""
Train the pose branch (Setting B): player-specific likely/unlikely scoring.

The pose model outputs per-class logits. We train with cross-entropy on the
true shot (ranking objective), which also yields calibrated per-class scores:
at inference, softmax -> ranking (top-k) and sigmoid+threshold -> likely/unlikely.

Player-specific modelling: a player-id embedding is concatenated before the
classifier. Only players with >= min_clips_per_player get a dedicated id; the
rest share id 0 ('unknown'), so the model degrades gracefully to a global pose
model when player data is scarce.

Usage
-----
python -m src.train.train_pose --config configs/default.yaml
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.common import load_config, set_seed, get_device, ensure_dir, save_json, get_logger, to_plain
from src.data.dataset import PoseSeqDataset, build_player_vocab
from src.models.pose_branch import build_pose_model
from src.eval.metrics import topk_accuracy, mean_reciprocal_rank

log = get_logger("train_pose")


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tot = {"top1": 0, "top3": 0}
    mrr, n = 0.0, 0
    for seq, length, y, pidx, _ in dl:
        seq, y, pidx = seq.to(device), y.to(device), pidx.to(device)
        logits = model(seq, pidx)
        acc = topk_accuracy(logits, y, (1, 3))
        bs = y.size(0)
        tot["top1"] += acc["top1"] * bs
        tot["top3"] += acc["top3"] * bs
        mrr += mean_reciprocal_rank(logits, y) * bs
        n += bs
    return {"top1": tot["top1"] / n, "top3": tot["top3"] / n, "mrr": mrr / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    player2idx = build_player_vocab(cfg)
    log.info("Player vocab size: %d (players with >=%d clips get own id)",
             len(player2idx), cfg.data.min_clips_per_player)

    tr = PoseSeqDataset(cfg, "train", player2idx)
    va = PoseSeqDataset(cfg, "val", player2idx)
    dl_tr = DataLoader(tr, batch_size=cfg.train.batch_size * 4, shuffle=True,
                       num_workers=cfg.train.num_workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=cfg.train.batch_size * 4, shuffle=False,
                       num_workers=cfg.train.num_workers)

    num_classes = len(cfg.classes)
    model = build_pose_model(cfg, num_classes, num_players=len(player2idx)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Pose model '%s' params: %.2fM | train=%d val=%d",
             cfg.model.pose.arch, n_params, len(tr), len(va))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr * 3,
                            weight_decay=cfg.train.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    ckpt_dir = ensure_dir(cfg.paths.ckpt_dir)
    best, best_ep, bad = -1.0, -1, 0
    history = []
    for ep in range(cfg.train.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for seq, length, y, pidx, _ in dl_tr:
            seq, y, pidx = seq.to(device), y.to(device), pidx.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(seq, pidx)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            run += loss.item()
        sch.step()
        val = evaluate(model, dl_va, device)
        log.info("ep %02d | loss %.3f | val top1 %.3f top3 %.3f mrr %.3f | %.0fs",
                 ep, run / len(dl_tr), val["top1"], val["top3"], val["mrr"], time.time() - t0)
        history.append({"epoch": ep, "loss": run / len(dl_tr), **val})
        if val["top1"] > best:
            best, best_ep, bad = val["top1"], ep, 0
            torch.save({"model": model.state_dict(), "cfg": to_plain(cfg),
                        "classes": list(cfg.classes), "player2idx": player2idx,
                        "epoch": ep, "val": val}, Path(ckpt_dir) / "pose_best.pt")
        else:
            bad += 1
            if bad >= cfg.train.early_stop_patience:
                log.info("Early stop ep %d (best %d top1 %.3f)", ep, best_ep, best)
                break

    save_json(history, Path(cfg.paths.results_dir) / "pose_history.json")
    log.info("Done. Best val top1 %.3f @ ep %d. Checkpoint: %s/pose_best.pt",
             best, best_ep, ckpt_dir)


if __name__ == "__main__":
    main()
