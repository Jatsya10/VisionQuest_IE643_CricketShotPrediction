"""
Stage 1 of preprocessing: build the clip index, a stratified train/val/test
split, and (optionally) cache uniformly-sampled RGB frames per clip.

Why frame caching: decoding video every epoch is the training bottleneck.
We sample `num_frames` frames ONCE, resize to img_size, store as uint8 .npy
of shape (num_frames, H, W, 3). The Dataset then just loads the .npy.

Player identity: the Kaggle set does not ship player labels. We attempt to
parse a player id from the filename if present (e.g. 'kohli_pull_012.mp4' ->
'kohli'); otherwise player='unknown'. Player-specific modelling then only
activates for any player that clears min_clips_per_player. This keeps the
pipeline honest: it never fabricates player labels.

Usage
-----
python -m src.data.extract_frames --config configs/default.yaml
python -m src.data.extract_frames --config configs/default.yaml --no-cache   # index+split only
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.common import load_config, set_seed, save_json, ensure_dir, get_logger

log = get_logger("frames")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# --------------------------------------------------------------------------
# Frame sampling backend (decord preferred, cv2 fallback)
# --------------------------------------------------------------------------
def _read_frames_decord(path, num_frames, size):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(path))
    total = len(vr)
    if total == 0:
        return None
    idx = np.linspace(0, total - 1, num_frames).astype(int)
    frames = vr.get_batch(idx).asnumpy()  # (T,H,W,3) RGB
    return _resize_stack(frames, size)


def _read_frames_cv2(path, num_frames, size):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # some containers report 0; read sequentially
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None
        arr = np.stack(frames)
        idx = np.linspace(0, len(arr) - 1, num_frames).astype(int)
        return _resize_stack(arr[idx], size)
    idx = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idx:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not out:
        return None
    arr = np.stack(out)
    if len(arr) < num_frames:  # pad by repeating last
        pad = np.repeat(arr[-1:], num_frames - len(arr), axis=0)
        arr = np.concatenate([arr, pad], 0)
    return _resize_stack(arr[:num_frames], size)


def _resize_stack(frames, size):
    import cv2
    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    return out


def read_frames(path, num_frames, size):
    try:
        return _read_frames_decord(path, num_frames, size)
    except Exception:
        try:
            return _read_frames_cv2(path, num_frames, size)
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            return None


# --------------------------------------------------------------------------
# Player id heuristic
# --------------------------------------------------------------------------
def guess_player(filename: str, classes: list[str]) -> str:
    stem = Path(filename).stem.lower()
    # strip class names and trailing digits; if a leading token survives, use it
    for c in classes:
        stem = stem.replace(c, " ")
    tokens = re.split(r"[^a-z]+", stem)
    tokens = [t for t in tokens if len(t) >= 3]
    return tokens[0] if tokens else "unknown"


# --------------------------------------------------------------------------
# Index + split
# --------------------------------------------------------------------------
def find_class_root(raw: Path, classes: list[str]) -> Path:
    candidates = [raw] + [p for p in raw.rglob("*") if p.is_dir()]
    for c in candidates:
        names = {d.name.lower() for d in c.iterdir() if d.is_dir()}
        if sum(cl in names for cl in classes) >= max(2, len(classes) // 2):
            return c
    return raw


def build_index(raw: Path, classes: list[str]) -> dict:
    root = find_class_root(raw, classes)
    log.info("Class root: %s", root)
    index = {}
    cls_lower = {c.lower(): c for c in classes}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.lower() not in cls_lower:
            continue
        label = cls_lower[d.name.lower()]
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS:
                cid = f"{label}__{f.stem}"
                index[cid] = {
                    "path": str(f.resolve()),
                    "label": label,
                    "player": guess_player(f.name, [c.lower() for c in classes]),
                }
    log.info("Indexed %d clips across %d classes.", len(index), len(classes))
    return index


def stratified_split(index: dict, ratios: dict, stratify: bool, seed: int) -> dict:
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for cid, meta in index.items():
        by_class[meta["label"] if stratify else "all"].append(cid)
    train, val, test = [], [], []
    for _, cids in by_class.items():
        cids = list(cids)
        rng.shuffle(cids)
        n = len(cids)
        n_tr = int(round(ratios["train"] * n))
        n_va = int(round(ratios["val"] * n))
        train += cids[:n_tr]
        val += cids[n_tr:n_tr + n_va]
        test += cids[n_tr + n_va:]
    log.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


def cache_frames(index: dict, frames_dir: Path, num_frames: int, size: int):
    ensure_dir(frames_dir)
    ok, bad = 0, 0
    for cid, meta in tqdm(index.items(), desc="caching frames"):
        out = frames_dir / f"{cid}.npy"
        if out.exists():
            ok += 1
            continue
        arr = read_frames(meta["path"], num_frames, size)
        if arr is None:
            bad += 1
            continue
        np.save(out, arr)
        ok += 1
    log.info("Cached %d clips (%d unreadable).", ok, bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-cache", action="store_true", help="build index+split only, skip frame caching")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)

    raw = Path(cfg.paths.raw_dataset)
    index = build_index(raw, list(cfg.classes))
    if not index:
        log.error("Empty index — check paths.raw_dataset and classes in config.")
        return
    splits = stratified_split(index, dict(cfg.data.split), cfg.data.stratify, cfg.seed)

    save_json(index, cfg.paths.index_file)
    save_json(splits, cfg.paths.splits_file)
    log.info("Wrote %s and %s", cfg.paths.index_file, cfg.paths.splits_file)

    if not args.no_cache:
        cache_frames(index, Path(cfg.paths.frames_dir), cfg.data.num_frames, cfg.data.img_size)


if __name__ == "__main__":
    main()
