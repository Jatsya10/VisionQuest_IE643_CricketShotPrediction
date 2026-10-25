#!/usr/bin/env bash
# End-to-end pipeline. Run from the repo root. Edit CONFIG if needed.
set -e
CONFIG=configs/default.yaml

echo "==> [1/6] Download + inspect dataset"
python -m src.data.download --config $CONFIG

echo "==> [2/6] Build index/split + cache frames"
python -m src.data.extract_frames --config $CONFIG

echo "==> [3/6] Extract pose keypoints"
python -m src.data.extract_pose --config $CONFIG

echo "==> [4/6] Train video branch (Setting A)"
python -m src.train.train_video --config $CONFIG

echo "==> [5/6] Train pose branch (Setting B)"
python -m src.train.train_pose --config $CONFIG

echo "==> [6/6] Temporal early-prediction analysis + figures"
python -m src.eval.temporal_analysis --config $CONFIG

echo "Done. See results/ and checkpoints/. Launch demo with:"
echo "   python -m app.gradio_app --config $CONFIG --share"
