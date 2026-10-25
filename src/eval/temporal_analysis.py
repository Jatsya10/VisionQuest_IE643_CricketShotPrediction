"""
Subtask 4: temporal early-prediction analysis.

Loads the trained video and pose checkpoints and evaluates each on the TEST
split at every observation ratio in cfg.data.observation_ratios. Produces:
  * results/temporal_results.json  : acc_by_ratio for both branches + earliness
  * results/figures/accuracy_vs_ratio.png
  * results/figures/confusion_video_full.png
  * results/figures/perclass_f1.png
  * results/test_metrics.json       : full-clip test metrics, per-class F1

This is the script whose numbers feed the report's Results section and the
slides. Run it after both branches are trained.

Usage
-----
python -m src.eval.temporal_analysis --config configs/default.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.common import load_config, get_device, ensure_dir, save_json, get_logger
from src.data.dataset import VideoClipDataset, PoseSeqDataset
from src.models.video_branch import build_video_model
from src.models.pose_branch import build_pose_model
from src.eval.metrics import (topk_accuracy, mean_reciprocal_rank, per_class_f1,
                              confusion, earliness_score)

log = get_logger("temporal")


@torch.no_grad()
def eval_video_at_ratio(model, cfg, device, ratio):
    ds = VideoClipDataset(cfg, "test")
    ds.fixed_ratio = ratio
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers)
    all_logits, all_y = [], []
    for x, y, _ in dl:
        all_logits.append(model(x.to(device)).cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    acc = topk_accuracy(logits, y, tuple(cfg.eval.topk))
    return acc, logits, y


@torch.no_grad()
def eval_pose_at_ratio(model, cfg, device, ratio, player2idx):
    ds = PoseSeqDataset(cfg, "test", player2idx)
    ds.fixed_ratio = ratio
    dl = DataLoader(ds, batch_size=cfg.train.batch_size * 4, shuffle=False,
                    num_workers=cfg.train.num_workers)
    all_logits, all_y = [], []
    for seq, length, y, pidx, _ in dl:
        all_logits.append(model(seq.to(device), pidx.to(device)).cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    acc = topk_accuracy(logits, y, (1, 3))
    acc["mrr"] = mean_reciprocal_rank(logits, y)
    return acc, logits, y


def plot_accuracy_vs_ratio(video_acc, pose_acc, ratios, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rr = [int(r * 100) for r in ratios]
    ax.plot(rr, [video_acc[r]["top1"] for r in ratios], "-o", label="Video Top-1", color="#B3231C")
    ax.plot(rr, [video_acc[r]["top3"] for r in ratios], "--o", label="Video Top-3", color="#B3231C", alpha=0.6)
    if pose_acc:
        ax.plot(rr, [pose_acc[r]["top1"] for r in ratios], "-s", label="Pose Top-1", color="#143D2B")
        ax.plot(rr, [pose_acc[r]["top3"] for r in ratios], "--s", label="Pose Top-3", color="#143D2B", alpha=0.6)
    ax.set_xlabel("Observed clip fraction (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Early-prediction accuracy vs. observation ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(Path(out).parent)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(cm, classes, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (video, full clip)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_perclass_f1(f1s, classes, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(classes, f1s, color="#143D2B")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 (video, full clip)")
    ax.set_xticklabels(classes, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(cfg.device)
    classes = list(cfg.classes)
    ratios = list(cfg.data.observation_ratios)
    fig_dir = ensure_dir(cfg.paths.figures_dir)

    # ---- video ----
    vpath = Path(cfg.paths.ckpt_dir) / "video_best.pt"
    video_acc = {}
    full_logits = full_y = None
    if vpath.exists():
        ck = torch.load(vpath, map_location=device, weights_only=False)
        vm = build_video_model(cfg, len(classes)).to(device)
        vm.load_state_dict(ck["model"]); vm.eval()
        for r in ratios:
            acc, logits, y = eval_video_at_ratio(vm, cfg, device, r)
            video_acc[r] = acc
            log.info("VIDEO ratio %.2f | " + " ".join(f"{k} %.3f" for k in acc), r, *acc.values())
            if r == 1.0:
                full_logits, full_y = logits, y
    else:
        log.warning("No video checkpoint at %s — skipping video.", vpath)

    # ---- pose ----
    ppath = Path(cfg.paths.ckpt_dir) / "pose_best.pt"
    pose_acc = {}
    if ppath.exists():
        ck = torch.load(ppath, map_location=device, weights_only=False)
        player2idx = ck.get("player2idx", {"unknown": 0})
        pm = build_pose_model(cfg, len(classes), num_players=len(player2idx)).to(device)
        pm.load_state_dict(ck["model"]); pm.eval()
        for r in ratios:
            acc, _, _ = eval_pose_at_ratio(pm, cfg, device, r, player2idx)
            pose_acc[r] = acc
            log.info("POSE  ratio %.2f | top1 %.3f top3 %.3f mrr %.3f",
                     r, acc["top1"], acc["top3"], acc["mrr"])
    else:
        log.warning("No pose checkpoint at %s — skipping pose.", ppath)

    # ---- figures + derived metrics ----
    results = {"classes": classes, "ratios": ratios,
               "video_acc": {str(k): v for k, v in video_acc.items()},
               "pose_acc": {str(k): v for k, v in pose_acc.items()}}
    if video_acc:
        results["video_earliness"] = earliness_score({r: a["top1"] for r, a in video_acc.items()})
    if pose_acc:
        results["pose_earliness"] = earliness_score({r: a["top1"] for r, a in pose_acc.items()})

    if video_acc:
        plot_accuracy_vs_ratio(video_acc, pose_acc, ratios, fig_dir / "accuracy_vs_ratio.png")
    if full_logits is not None:
        y_pred = full_logits.argmax(1).numpy()
        y_true = full_y.numpy()
        macro, percls = per_class_f1(y_true, y_pred, len(classes))
        results["video_full_macro_f1"] = macro
        results["video_full_per_class"] = percls
        cm = confusion(y_true, y_pred, len(classes))
        plot_confusion(cm, classes, fig_dir / "confusion_video_full.png")
        plot_perclass_f1(percls["f1"], classes, fig_dir / "perclass_f1.png")

    save_json(results, Path(cfg.paths.results_dir) / "temporal_results.json")
    log.info("Wrote results + figures to %s", cfg.paths.results_dir)


if __name__ == "__main__":
    main()
