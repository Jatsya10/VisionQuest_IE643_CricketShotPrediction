"""
Train the video branch (Setting A).

Loss: earliness-aware cross-entropy. For each batch the Dataset already draws a
random observation ratio per sample, so standard CE over that mixed-ratio batch
trains the model to be correct across horizons. We ADDITIONALLY, every
`earliness_every` steps, draw one extra forced-early pass (ratio from the lower
end) and add earliness_lambda * CE on it. This explicitly rewards getting the
answer right from few frames, following Aliakbarian et al. (2017).

    L = CE(full-batch, mixed ratio)  +  earliness_lambda * CE(forced-early batch)

Optimiser: AdamW, cosine schedule with warmup, mixed precision on CUDA.

Usage
-----
python -m src.train.train_video --config configs/default.yaml
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
from src.data.dataset import VideoClipDataset
from src.models.video_branch import build_video_model
from src.eval.metrics import topk_accuracy

log = get_logger("train_video")


def make_loaders(cfg):
    tr = VideoClipDataset(cfg, "train")
    va = VideoClipDataset(cfg, "val")
    dl_tr = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False,
                       num_workers=cfg.train.num_workers, pin_memory=True)
    return tr, va, dl_tr, dl_va


def build_optim(cfg, model, steps_per_epoch):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total = cfg.train.epochs * steps_per_epoch
    warm = cfg.train.warmup_epochs * steps_per_epoch
    if cfg.train.scheduler == "cosine":
        def lr_lambda(step):
            if step < warm:
                return step / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + np.cos(np.pi * prog))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif cfg.train.scheduler == "step":
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.train.epochs // 3), gamma=0.3)
    else:
        sch = None
    return opt, sch


@torch.no_grad()
def evaluate(model, dl, device, num_classes):
    model.eval()
    tot = {"top1": 0, "top3": 0, "top5": 0}
    n = 0
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        acc = topk_accuracy(logits, y, (1, 3, 5))
        bs = y.size(0)
        for k in tot:
            tot[k] += acc[k] * bs
        n += bs
    return {k: v / max(1, n) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log.info("Device: %s", device)

    tr, va, dl_tr, dl_va = make_loaders(cfg)
    num_classes = len(cfg.classes)
    model = build_video_model(cfg, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Video model '%s' params: %.2fM | train=%d val=%d",
             cfg.model.video.backbone, n_params, len(tr), len(va))

    opt, sch = build_optim(cfg, model, len(dl_tr))
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = ensure_dir(cfg.paths.ckpt_dir)
    best_top1, best_ep, bad = -1.0, -1, 0
    history = []

    low_ratio = min(cfg.data.train_ratios)
    for ep in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for it, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y)
                # earliness term: force an early view on this same batch
                if cfg.train.earliness_lambda > 0 and it % 2 == 0:
                    k = max(1, int(round(low_ratio * x.size(2))))
                    xe = x.clone()
                    xe[:, :, k:] = x[:, :, k - 1:k]      # mask future frames
                    loss = loss + cfg.train.earliness_lambda * ce(model(xe), y)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            if sch:
                sch.step()
            run += loss.item()
        val = evaluate(model, dl_va, device, num_classes)
        dt = time.time() - t0
        log.info("ep %02d | loss %.3f | val top1 %.3f top3 %.3f top5 %.3f | %.0fs",
                 ep, run / len(dl_tr), val["top1"], val["top3"], val["top5"], dt)
        history.append({"epoch": ep, "loss": run / len(dl_tr), **val})

        if val["top1"] > best_top1:
            best_top1, best_ep, bad = val["top1"], ep, 0
            torch.save({"model": model.state_dict(), "cfg": to_plain(cfg),
                        "classes": list(cfg.classes), "epoch": ep, "val": val},
                       Path(ckpt_dir) / "video_best.pt")
        else:
            bad += 1
            if bad >= cfg.train.early_stop_patience:
                log.info("Early stop at ep %d (best ep %d, top1 %.3f)", ep, best_ep, best_top1)
                break

    save_json(history, Path(cfg.paths.results_dir) / "video_history.json")
    log.info("Done. Best val top1 %.3f @ ep %d. Checkpoint: %s/video_best.pt",
             best_top1, best_ep, ckpt_dir)


if __name__ == "__main__":
    main()
