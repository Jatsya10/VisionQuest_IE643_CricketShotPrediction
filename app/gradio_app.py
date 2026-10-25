"""
Vision Quest — interactive demo interface (Setting A + B).

Upload a cricket shot clip. Choose how much of it the model is allowed to see
(observation ratio). The app:
  * samples frames, runs the VIDEO branch -> Top-3/Top-5 shot prediction + conf
  * runs MediaPipe pose -> POSE branch -> likely/unlikely per shot
  * overlays the batter skeleton on a sampled frame
  * plots the confidence curve across observation ratios (how belief evolves)

This satisfies the "test interface" deliverable and is driveable live in the
viva by uploading a reviewer's clip or moving the observation-ratio slider.

Run
---
python -m app.gradio_app --config configs/default.yaml
# then open the printed local URL (use share=True on Colab)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from src.utils.common import load_config, get_device
from src.data.extract_frames import read_frames
from src.data.extract_pose import normalize_pose
from src.data.dataset import _MEAN, _STD, seen_count
from src.models.video_branch import build_video_model
from src.models.pose_branch import build_pose_model
from src.utils.viz import draw_pose_on_frame, confidence_curve


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.classes = list(cfg.classes)
        self.num_frames = cfg.data.num_frames
        self.size = cfg.data.img_size
        self.video = self._load_video()
        self.pose, self.player2idx = self._load_pose()
        self._mp = None

    def _load_video(self):
        p = Path(self.cfg.paths.ckpt_dir) / "video_best.pt"
        if not p.exists():
            return None
        ck = torch.load(p, map_location=self.device, weights_only=False)
        m = build_video_model(self.cfg, len(self.classes)).to(self.device)
        m.load_state_dict(ck["model"]); m.eval()
        return m

    def _load_pose(self):
        p = Path(self.cfg.paths.ckpt_dir) / "pose_best.pt"
        if not p.exists():
            return None, {"unknown": 0}
        ck = torch.load(p, map_location=self.device, weights_only=False)
        p2i = ck.get("player2idx", {"unknown": 0})
        m = build_pose_model(self.cfg, len(self.classes), num_players=len(p2i)).to(self.device)
        m.load_state_dict(ck["model"]); m.eval()
        return m, p2i

    def _pose_detector(self):
        # Reuse the robust backend from preprocessing (handles both the legacy
        # solutions.pose API and the new Tasks PoseLandmarker API).
        if self._mp is None:
            from src.data.extract_pose import make_backend
            self._mp = make_backend()
        return self._mp

    def _mask_video(self, frames, ratio):
        k = seen_count(self.num_frames, ratio)
        if k < self.num_frames:
            tail = np.repeat(frames[k - 1:k], self.num_frames - k, axis=0)
            frames = np.concatenate([frames[:k], tail], 0)
        return frames, k

    @torch.no_grad()
    def predict(self, video_path, ratio):
        if video_path is None:
            return "Please upload a clip.", None, None, None
        frames = read_frames(video_path, self.num_frames, self.size)
        if frames is None:
            return "Could not read this video.", None, None, None

        # ---- video branch ----
        report_lines = []
        topk_txt = "Video model not loaded."
        if self.video is not None:
            vf, k = self._mask_video(frames.copy(), ratio)
            x = vf.astype(np.float32) / 255.0
            x = (x - _MEAN) / _STD
            x = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)
            probs = torch.softmax(self.video(x), 1)[0].cpu().numpy()
            order = probs.argsort()[::-1]
            topn = min(5, len(self.classes))
            topk_txt = "\n".join(
                f"{r+1}. {self.classes[order[r]]:12s}  {probs[order[r]]*100:5.1f}%"
                for r in range(topn))
            report_lines.append(f"Observed {int(ratio*100)}% of clip ({k}/{self.num_frames} frames).")

        # ---- pose branch ----
        pose_txt = "Pose model not loaded."
        overlay = None
        det = self._pose_detector()
        anchor = min(len(frames) - 1, seen_count(self.num_frames, ratio) - 1)
        kp_anchor = det.detect(frames[anchor])          # (33,3) raw or None
        if kp_anchor is not None:
            kp_xy = kp_anchor[:, :2]
            overlay = draw_pose_on_frame(frames[anchor], kp_xy)
        if self.pose is not None:
            seq = np.zeros((self.num_frames, 33, 3), dtype=np.float32)
            kk = seen_count(self.num_frames, ratio)
            for t in range(kk):
                raw = det.detect(frames[t])
                if raw is not None:
                    seq[t] = normalize_pose(raw)
            st = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            pid = torch.zeros(1, dtype=torch.long, device=self.device)
            pp = torch.softmax(self.pose(st, pid), 1)[0].cpu().numpy()
            thr = self.cfg.eval.pose_likely_threshold
            likely = [self.classes[i] for i in range(len(self.classes)) if pp[i] >= thr]
            ordp = pp.argsort()[::-1]
            pose_txt = "Likely: " + (", ".join(likely) if likely else "(none above threshold)")
            pose_txt += "\n\nPose ranking:\n" + "\n".join(
                f"  {self.classes[ordp[i]]:12s} {pp[ordp[i]]*100:5.1f}%" for i in range(min(3, len(ordp))))

        # ---- confidence curve across ratios (video branch) ----
        curve_path = None
        if self.video is not None:
            per = {}
            for r in self.cfg.data.observation_ratios:
                vf, _ = self._mask_video(frames.copy(), r)
                x = ((vf.astype(np.float32) / 255.0 - _MEAN) / _STD)
                x = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)
                per[r] = torch.softmax(self.video(x), 1)[0].cpu().numpy()
            curve_path = "/tmp/vq_conf_curve.png"
            confidence_curve(per, self.classes, curve_path)

        return topk_txt, pose_txt, overlay, curve_path


def build_ui(engine):
    import gradio as gr
    with gr.Blocks(title="Vision Quest — Early Cricket Shot Prediction") as demo:
        gr.Markdown("# Vision Quest — Early Prediction of Batting Shot-Type\n"
                    "Upload a shot clip and set how much of it the model may see.")
        with gr.Row():
            with gr.Column():
                vid = gr.Video(label="Cricket shot clip")
                ratio = gr.Slider(0.25, 1.0, value=0.5, step=0.25,
                                  label="Observation ratio (fraction of clip seen)")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                topk = gr.Textbox(label="Video branch — Top-k shot prediction", lines=6)
                pose = gr.Textbox(label="Pose branch — likely/unlikely", lines=6)
        with gr.Row():
            overlay = gr.Image(label="Batter pose overlay")
            curve = gr.Image(label="Confidence vs. observed fraction")
        btn.click(engine.predict, [vid, ratio], [topk, pose, overlay, curve])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--share", action="store_true", help="public link (Colab)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    engine = Engine(cfg)
    ui = build_ui(engine)
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
