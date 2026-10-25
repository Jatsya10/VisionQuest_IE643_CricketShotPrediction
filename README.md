# Vision Quest — Early Prediction of Batting Shot-Type from Cricket Videos

IE 643 Course Project (Deep Learning – Theory and Practice), IIT Bombay, 2026.
**Team Vision Quest.**

Predicts a batter's cricket shot **before the shot is completed**, in two
complementary settings:

- **Setting A (video):** Top-3 / Top-5 shot prediction from only the first
  **50% / 75%** of a clip (early action prediction).
- **Setting B (pose):** player-specific **likely / unlikely** shot scoring from
  the batter's body-pose keypoint sequence.

A temporal analysis measures how accuracy grows with the observed fraction and
how early a reliable call can be made.

---

## 1. Repository layout

```
configs/default.yaml      All paths + hyperparameters (single source of truth)
src/data/                 download, frame extraction, pose extraction, datasets
src/models/               video branch (R3D-18 / CNN-GRU), pose branch (ST-GCN)
src/train/                train_video.py, train_pose.py
src/eval/                 metrics.py, temporal_analysis.py (Subtask 4)
src/utils/                config/seed/device, pose-overlay & confidence-curve viz
app/gradio_app.py         interactive test interface (live demo)
scripts/run_all.sh        end-to-end pipeline
scripts/smoke_test.py     synthetic end-to-end test (no data/GPU needed)
notebooks/colab_runner.ipynb   one-click Colab run
```

## 2. Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended (the video branch is a 3D-CNN). On Apple Silicon the
code runs on MPS/CPU but training is slow; use Google Colab (T4) for real runs.

## 3. Data

Dataset: **Cricket Shots – IPL 2023** (Kaggle, Apache-2.0).
`taarunsridhar/cricket-shots-ipl-2023`. Place your Kaggle API token at
`~/.kaggle/kaggle.json` (`chmod 600`), then the pipeline downloads it.

Verify the class folder names match `classes:` in the config:

```bash
python -m src.data.download --inspect --raw data/raw/cricket-shots-ipl-2023
```

Edit `configs/default.yaml -> classes:` if the dataset's folder names differ.

## 4. Run everything

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
python -m src.data.download          # 1. fetch + inspect
python -m src.data.extract_frames    # 2. index, stratified split, cache frames
python -m src.data.extract_pose      # 3. MediaPipe keypoints per clip
python -m src.train.train_video      # 4. Setting A
python -m src.train.train_pose       # 5. Setting B
python -m src.eval.temporal_analysis # 6. Subtask 4: curves, F1, confusion
```

Outputs:
- `checkpoints/video_best.pt`, `checkpoints/pose_best.pt`
- `results/temporal_results.json`, `results/*_history.json`
- `results/figures/accuracy_vs_ratio.png`, `confusion_video_full.png`, `perclass_f1.png`

## 5. Demo interface

```bash
python -m app.gradio_app --share   # --share for a public Colab link
```

Upload a clip, move the **observation-ratio** slider (25/50/75/100%), and the
app shows Top-k shot prediction, pose-based likely/unlikely, the batter pose
overlay, and the confidence-vs-time curve. Driveable live during the viva.

## 6. Smoke test (verify the code without data)

```bash
python scripts/smoke_test.py    # prints "SMOKE TEST PASSED"
```

## 7. Method summary

| Component | Choice | Why |
|---|---|---|
| Video backbone | R3D-18 (Kinetics-400 pretrained) | strong spatiotemporal features, trainable on one T4 |
| Video alt | ResNet-18 + GRU | CricShotClassify-style baseline |
| Pose extractor | MediaPipe BlazePose (33 kpts) | real-time, robust on broadcast footage |
| Pose model | compact ST-GCN + player embedding | skeleton is a graph; player-specific technique |
| Partial obs. | prefix masking in the Dataset | one model generalises across horizons |
| Loss | CE (label-smooth) + earliness term | rewards correct early predictions (Aliakbarian 2017) |
| Metrics | Top-k, per-class F1, MRR, earliness score | matches the four subtasks |

Full-clip reference: Rajarajeswari et al. (2026, *Scientific Reports*) report
95.6% on this dataset with a 3D-CNN + optical-flow + pose pipeline.

## 8. Config knobs you will likely touch

- `classes:` — must match dataset folder names.
- `data.num_frames`, `data.img_size` — speed/accuracy trade-off.
- `train.epochs`, `train.batch_size`, `train.lr`.
- `model.video.backbone` — `r3d_18` or `cnn_gru`.
- `model.pose.player_embedding_dim` — set `0` to disable player-specific head.

## 9. References

See the project report's References section.
