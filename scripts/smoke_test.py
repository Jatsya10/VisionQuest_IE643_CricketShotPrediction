"""
Self-contained smoke test: fabricates a tiny synthetic dataset (random frames,
random pose, 7 classes) and runs the FULL pipeline end-to-end for 1 epoch each
branch + temporal analysis. No real data / GPU needed. Purpose: catch shape and
wiring bugs before committing a long training run.

Run:  python scripts/smoke_test.py
Pass => "SMOKE TEST PASSED" printed and exit 0.
"""
from __future__ import annotations
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import load_config, save_json, ensure_dir  # noqa: E402


def build_fake(cfg, n_per_class=6):
    classes = list(cfg.classes)
    nf, sz = cfg.data.num_frames, cfg.data.img_size
    frames_dir = ensure_dir(cfg.paths.frames_dir)
    pose_dir = ensure_dir(cfg.paths.pose_dir)
    index = {}
    players = ["kohli", "rohit", "unknown"]
    for ci, c in enumerate(classes):
        for k in range(n_per_class):
            cid = f"{c}__fake{k}"
            # frames: random but class-tinted so the model can actually learn a bit
            arr = (np.random.rand(nf, sz, sz, 3) * 60 + ci * 20).clip(0, 255).astype(np.uint8)
            np.save(frames_dir / f"{cid}.npy", arr)
            pose = np.random.rand(nf, 33, 3).astype(np.float32)
            pose[:, :, :2] += ci * 0.1          # class signal in pose too
            np.save(pose_dir / f"{cid}.npy", pose)
            np.save(pose_dir / f"{cid}.mask.npy", np.ones(nf, np.float32))
            index[cid] = {"path": f"/fake/{cid}.mp4", "label": c,
                          "player": players[k % len(players)]}
    save_json(index, cfg.paths.index_file)
    # split
    from collections import defaultdict
    import random
    random.seed(0)
    by = defaultdict(list)
    for cid, m in index.items():
        by[m["label"]].append(cid)
    tr, va, te = [], [], []
    for c, ids in by.items():
        random.shuffle(ids)
        tr += ids[:4]; va += ids[4:5]; te += ids[5:]
    save_json({"train": tr, "val": va, "test": te}, cfg.paths.splits_file)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vq_smoke_"))
    cfg = load_config(str(ROOT / "configs/default.yaml"))
    # redirect all paths into tmp + shrink everything
    for key in ["frames_dir", "pose_dir", "ckpt_dir", "results_dir", "figures_dir"]:
        cfg.paths[key] = str(tmp / key)
    cfg.paths.index_file = str(tmp / "index.json")
    cfg.paths.splits_file = str(tmp / "splits.json")
    cfg.model.video.backbone = "cnn_gru"   # lighter than r3d for CPU smoke
    cfg.model.video.pretrained = False
    cfg.model.pose.pretrained = False
    cfg.train.epochs = 1
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0
    cfg.train.amp = False
    cfg.data.min_clips_per_player = 2
    cfg.device = "cpu"

    print("building fake data ...")
    build_fake(cfg)

    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import VideoClipDataset, PoseSeqDataset, build_player_vocab
    from src.models.video_branch import build_video_model
    from src.models.pose_branch import build_pose_model
    from src.eval.metrics import topk_accuracy, mean_reciprocal_rank, per_class_f1, earliness_score

    # ---- video forward/backward ----
    print("video branch ...")
    ds = VideoClipDataset(cfg, "train")
    dl = DataLoader(ds, batch_size=2)
    vm = build_video_model(cfg, len(cfg.classes))
    x, y, r = next(iter(dl))
    out = vm(x)
    assert out.shape == (2, len(cfg.classes)), out.shape
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    acc = topk_accuracy(out, y, (1, 3, 5))
    assert set(acc) == {"top1", "top3", "top5"}

    # ---- pose forward/backward (with player embedding) ----
    print("pose branch ...")
    p2i = build_player_vocab(cfg)
    pds = PoseSeqDataset(cfg, "train", p2i)
    pdl = DataLoader(pds, batch_size=4)
    pm = build_pose_model(cfg, len(cfg.classes), num_players=max(2, len(p2i)))
    seq, length, yy, pidx, rr = next(iter(pdl))
    po = pm(seq, pidx)
    assert po.shape[1] == len(cfg.classes), po.shape
    torch.nn.functional.cross_entropy(po, yy).backward()
    _ = mean_reciprocal_rank(po, yy)

    # ---- partial observation: shapes stay fixed across ratios ----
    print("partial-observation shape check ...")
    for rt in [0.25, 0.5, 0.75, 1.0]:
        ds.fixed_ratio = rt
        xx, _, got = ds[0]
        assert xx.shape == (3, cfg.data.num_frames, cfg.data.img_size, cfg.data.img_size)
        assert abs(got - rt) < 1e-6

    # ---- full training scripts (1 epoch) ----
    print("train_video (1 epoch) ...")
    import importlib
    import src.train.train_video as tv
    sys.argv = ["x", "--config", str(ROOT / "configs/default.yaml")]
    # monkey-patch config loader to our tmp cfg
    tv.load_config = lambda *a, **k: cfg
    tv.main()
    assert (Path(cfg.paths.ckpt_dir) / "video_best.pt").exists()

    print("train_pose (1 epoch) ...")
    import src.train.train_pose as tp
    tp.load_config = lambda *a, **k: cfg
    tp.main()
    assert (Path(cfg.paths.ckpt_dir) / "pose_best.pt").exists()

    print("temporal_analysis ...")
    import src.eval.temporal_analysis as ta
    ta.load_config = lambda *a, **k: cfg
    ta.main()
    res = json.loads((Path(cfg.paths.results_dir) / "temporal_results.json").read_text())
    assert "video_acc" in res and "pose_acc" in res
    assert (Path(cfg.paths.figures_dir) / "accuracy_vs_ratio.png").exists()

    # ---- earliness util ----
    assert 0 < earliness_score({0.25: 0.5, 0.5: 0.7, 0.75: 0.75, 1.0: 0.76}) <= 1.0

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
