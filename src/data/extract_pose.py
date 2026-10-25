"""
Stage 2 of preprocessing: extract batter body-pose keypoints per clip.

For each clip we run MediaPipe BlazePose on the `num_frames` sampled frames
(reusing the cached frames from extract_frames.py so sampling is identical to
the video branch) and store a (num_frames, 33, 3) array = (x, y, visibility),
normalised to be translation/scale invariant.

If MediaPipe finds no person in a frame, that frame's keypoints are zeros and
a per-frame validity mask records it (saved alongside).

Usage
-----
python -m src.data.extract_pose --config configs/default.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.common import load_config, load_json, ensure_dir, get_logger

log = get_logger("pose")


def normalize_pose(kp: np.ndarray) -> np.ndarray:
    """
    kp: (33, 3) raw (x,y,visibility) in [0,1] image coords.
    Center on the mid-hip, scale by torso length (mid-shoulder<->mid-hip).
    Makes the representation invariant to where/how big the batter is in frame.
    """
    LHIP, RHIP, LSH, RSH = 23, 24, 11, 12
    xy = kp[:, :2].copy()
    vis = kp[:, 2:3]
    mid_hip = (xy[LHIP] + xy[RHIP]) / 2.0
    mid_sh = (xy[LSH] + xy[RSH]) / 2.0
    xy = xy - mid_hip
    torso = np.linalg.norm(mid_sh - mid_hip) + 1e-6
    xy = xy / torso
    return np.concatenate([xy, vis], axis=1).astype(np.float32)


def extract_from_frames(frames: np.ndarray, pose_detector, normalize: bool):
    """frames: (T,H,W,3) uint8 RGB -> (T,33,3), mask (T,)"""
    T = len(frames)
    out = np.zeros((T, 33, 3), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        res = pose_detector.process(frames[t])
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            kp = np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)  # (33,3)
            out[t] = normalize_pose(kp) if normalize else kp
            mask[t] = 1.0
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    index = load_json(cfg.paths.index_file)
    frames_dir = Path(cfg.paths.frames_dir)
    pose_dir = ensure_dir(cfg.paths.pose_dir)

    try:
        import mediapipe as mp
    except ImportError:
        log.error("mediapipe not installed. pip install mediapipe")
        return
    mp_pose = mp.solutions.pose
    detector = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                            min_detection_confidence=0.5)

    from src.data.extract_frames import read_frames
    ok, bad = 0, 0
    for cid, meta in tqdm(index.items(), desc="pose"):
        out_kp = pose_dir / f"{cid}.npy"
        out_mask = pose_dir / f"{cid}.mask.npy"
        if out_kp.exists():
            ok += 1
            continue
        fp = frames_dir / f"{cid}.npy"
        if fp.exists():
            frames = np.load(fp)
        else:  # fall back to decoding the raw clip
            frames = read_frames(meta["path"], cfg.data.num_frames, cfg.data.img_size)
            if frames is None:
                bad += 1
                continue
        kp, mask = extract_from_frames(frames, detector, cfg.pose.normalize)
        np.save(out_kp, kp)
        np.save(out_mask, mask)
        ok += 1
    detector.close()
    log.info("Pose extracted for %d clips (%d failed). Mean frame-detection rate varies by clip.", ok, bad)


if __name__ == "__main__":
    main()
