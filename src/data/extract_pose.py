"""
Stage 2 of preprocessing: extract batter body-pose keypoints per clip.

IMPROVED two-stage, high-resolution + crop pipeline (fixes low detection rate
on small broadcast batters):

  1. Decode each clip at HIGH resolution (pose_read_size, default 256) directly
     from the original video -- NOT the 112px video cache.
  2. Stage-1 detect on the full high-res frame to locate the person.
  3. Crop a padded box around that person, upscale the crop, and re-run pose on
     it. The batter now fills the frame -> far more reliable keypoints. The
     crop-space landmarks are mapped back to full-frame [0,1] coords so the
     normalisation and downstream model are unchanged.
  4. If no person is found at all, the frame's keypoints stay zero (mask=0).
  5. Short detection gaps are linearly interpolated so a mostly-detected
     sequence is not half-zeros.

Output format is unchanged: (num_frames, 33, 3) = (x, y, visibility), torso-
normalised, plus a (num_frames,) validity mask.

Works with both MediaPipe APIs (legacy solutions.pose and new Tasks
PoseLandmarker). The new API needs a small model file, auto-downloaded.

Usage
-----
python -m src.data.extract_pose --config configs/default.yaml
# tune resolution / crop if needed:
python -m src.data.extract_pose --config configs/default.yaml --set pose.read_size=320 pose.crop_margin=0.35
"""
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.common import load_config, load_json, ensure_dir, get_logger

log = get_logger("pose")

_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
_TASK_PATH = Path("models/pose_landmarker_heavy.task")


def normalize_pose(kp: np.ndarray) -> np.ndarray:
    """kp: (33,3) (x,y,visibility) in [0,1]. Centre on mid-hip, scale by torso."""
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
# Backends: detect(frame_rgb_uint8) -> (33,3) raw [0,1] coords, or None
# --------------------------------------------------------------------------
class _OldBackend:
    def __init__(self):
        import mediapipe as mp
        self.d = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                        min_detection_confidence=0.3)

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
            log.info("Downloading pose model (heavy) -> %s", _TASK_PATH)
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
                    f"Could not download the pose model ({last}). Manually download\n"
                    f"{_TASK_URL}\nto {_TASK_PATH.resolve()}")
        base = python.BaseOptions(model_asset_path=str(_TASK_PATH))
        opts = vision.PoseLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3)
        self.d = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame):
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(frame))
        res = self.d.detect(mp_img)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        return np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)

    def close(self):
        self.d.close()


def make_backend():
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        log.info("Using MediaPipe legacy solutions.pose API (complexity 2).")
        return _OldBackend()
    log.info("Using MediaPipe Tasks PoseLandmarker API (heavy, v%s).",
             getattr(mp, "__version__", "?"))
    return _NewBackend()


# --------------------------------------------------------------------------
# Two-stage crop detection on one high-res frame
# --------------------------------------------------------------------------
def _bbox_from_kp(kp, H, W, margin):
    vis = kp[:, 2] > 0.3
    if vis.sum() < 4:
        return None
    xs = kp[vis, 0] * W
    ys = kp[vis, 1] * H
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 -= margin * bw; x1 += margin * bw
    y0 -= margin * bh; y1 += margin * bh
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W, int(x1)); y1 = min(H, int(y1))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return x0, y0, x1, y1


def detect_frame_twostage(frame, backend, crop_size, margin):
    """frame: (H,W,3) uint8 high-res. Returns (33,3) in full-frame [0,1] or None."""
    import cv2
    H, W = frame.shape[:2]
    kp1 = backend.detect(frame)                     # stage-1 full frame
    if kp1 is None:
        return None
    box = _bbox_from_kp(kp1, H, W, margin)
    if box is None:
        return kp1                                  # use stage-1 as-is
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    crop_up = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    kp2 = backend.detect(crop_up)                   # stage-2 on zoomed crop
    if kp2 is None:
        return kp1
    # map crop-space [0,1] back to full-frame [0,1]
    cw, ch = (x1 - x0), (y1 - y0)
    out = kp2.copy()
    out[:, 0] = (kp2[:, 0] * cw + x0) / W
    out[:, 1] = (kp2[:, 1] * ch + y0) / H
    return out


def interpolate_gaps(seq, mask):
    """Linearly interpolate (x,y,vis) over short zero-detection gaps."""
    T = len(seq)
    valid = np.where(mask > 0)[0]
    if len(valid) < 2:
        return seq
    for t in range(T):
        if mask[t] > 0:
            continue
        prev = valid[valid < t]
        nxt = valid[valid > t]
        if len(prev) == 0 or len(nxt) == 0:
            continue
        a, b = prev[-1], nxt[0]
        w = (t - a) / (b - a)
        seq[t] = (1 - w) * seq[a] + w * seq[b]
    return seq


def extract_clip(frames_hr, backend, crop_size, margin, normalize):
    T = len(frames_hr)
    raw = np.zeros((T, 33, 3), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        kp = detect_frame_twostage(frames_hr[t], backend, crop_size, margin)
        if kp is not None and kp.shape == (33, 3):
            raw[t] = kp
            mask[t] = 1.0
    raw = interpolate_gaps(raw, mask)
    out = np.zeros_like(raw)
    for t in range(T):
        if raw[t].any():
            out[t] = normalize_pose(raw[t]) if normalize else raw[t]
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    # pose-specific knobs (with safe defaults if absent in config)
    read_size = int(cfg.pose.get("read_size", 256))
    crop_size = int(cfg.pose.get("crop_size", 256))
    margin = float(cfg.pose.get("crop_margin", 0.35))

    index = load_json(cfg.paths.index_file)
    pose_dir = ensure_dir(cfg.paths.pose_dir)

    try:
        import mediapipe  # noqa: F401
    except ImportError:
        log.error("mediapipe not installed. pip install mediapipe")
        return

    backend = make_backend()
    from src.data.extract_frames import read_frames

    log.info("Pose config: read_size=%d crop_size=%d margin=%.2f", read_size, crop_size, margin)
    ok, bad = 0, 0
    det_rates = []
    for cid, meta in tqdm(index.items(), desc="pose"):
        out_kp = pose_dir / f"{cid}.npy"
        out_mask = pose_dir / f"{cid}.mask.npy"
        if out_kp.exists():
            ok += 1
            continue
        # decode the ORIGINAL clip at high resolution (not the 112px cache)
        frames_hr = read_frames(meta["path"], cfg.data.num_frames, read_size)
        if frames_hr is None:
            bad += 1
            continue
        kp, mask = extract_clip(frames_hr, backend, crop_size, margin, cfg.pose.normalize)
        np.save(out_kp, kp)
        np.save(out_mask, mask)
        det_rates.append(mask.mean())
        ok += 1
    backend.close()
    if det_rates:
        log.info("Pose extracted for %d clips (%d failed). Mean per-frame detection: %.2f",
                 ok, bad, float(np.mean(det_rates)))
    else:
        log.info("Pose extracted for %d clips (%d failed).", ok, bad)


if __name__ == "__main__":
    main()
