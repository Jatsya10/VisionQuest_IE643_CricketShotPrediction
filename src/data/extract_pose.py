"""
Stage 2 of preprocessing: extract batter body-pose keypoints per clip.

For each clip we run MediaPipe BlazePose on the `num_frames` sampled frames
(reusing the cached frames from extract_frames.py so sampling is identical to
the video branch) and store a (num_frames, 33, 3) array = (x, y, visibility),
normalised to be translation/scale invariant.

Works with BOTH MediaPipe APIs:
  * old  (<=0.10.18 ish): mp.solutions.pose.Pose
  * new  (>=0.10.30):     mediapipe.tasks ... PoseLandmarker
The new API needs a small model file (pose_landmarker_lite.task); it is
auto-downloaded on first use to models/pose_landmarker_lite.task.

Usage
-----
python -m src.data.extract_pose --config configs/default.yaml
"""
from __future__ import annotations
import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.common import load_config, load_json, ensure_dir, get_logger

log = get_logger("pose")

_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
_TASK_PATH = Path("models/pose_landmarker_lite.task")


def normalize_pose(kp: np.ndarray) -> np.ndarray:
    """
    kp: (33, 3) raw (x,y,visibility) in [0,1] image coords.
    Center on the mid-hip, scale by torso length (mid-shoulder<->mid-hip).
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


# --------------------------------------------------------------------------
# Backend wrappers: both expose detect(frame_rgb_uint8) -> (33,3) or None
# --------------------------------------------------------------------------
class _OldBackend:
    def __init__(self):
        import mediapipe as mp
        self.d = mp.solutions.pose.Pose(static_image_mode=True,
                                        model_complexity=1,
                                        min_detection_confidence=0.5)

    def detect(self, frame):
        res = self.d.process(frame)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        return np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)

    def close(self):
        self.d.close()


class _NewBackend:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self._mp = mp
        if not _TASK_PATH.exists():
            ensure_dir(_TASK_PATH.parent)
            log.info("Downloading pose model -> %s", _TASK_PATH)
            last = None
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(_TASK_URL, _TASK_PATH)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    log.warning("download attempt %d failed: %s", attempt + 1, e)
            if not _TASK_PATH.exists():
                raise RuntimeError(
                    f"Could not download the pose model after 3 tries ({last}).\n"
                    f"Manual fix: download {_TASK_URL}\n"
                    f"and place it at {_TASK_PATH.resolve()}")
        base = python.BaseOptions(model_asset_path=str(_TASK_PATH))
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5)
        self.d = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame):
        # frame: (H,W,3) uint8 RGB
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(frame))
        res = self.d.detect(mp_img)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]  # first (only) person
        return np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)

    def close(self):
        self.d.close()


def make_backend():
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        log.info("Using MediaPipe legacy solutions.pose API.")
        return _OldBackend()
    log.info("Using MediaPipe Tasks PoseLandmarker API (v%s).", getattr(mp, "__version__", "?"))
    return _NewBackend()


def extract_from_frames(frames: np.ndarray, backend, normalize: bool):
    """frames: (T,H,W,3) uint8 RGB -> (T,33,3), mask (T,)"""
    T = len(frames)
    out = np.zeros((T, 33, 3), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        kp = backend.detect(frames[t])
        if kp is not None and kp.shape == (33, 3):
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
        import mediapipe  # noqa: F401
    except ImportError:
        log.error("mediapipe not installed. pip install mediapipe")
        return

    backend = make_backend()
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
        else:
            frames = read_frames(meta["path"], cfg.data.num_frames, cfg.data.img_size)
            if frames is None:
                bad += 1
                continue
        kp, mask = extract_from_frames(frames, backend, cfg.pose.normalize)
        np.save(out_kp, kp)
        np.save(out_mask, mask)
        ok += 1
    backend.close()
    log.info("Pose extracted for %d clips (%d failed).", ok, bad)


if __name__ == "__main__":
    main()
