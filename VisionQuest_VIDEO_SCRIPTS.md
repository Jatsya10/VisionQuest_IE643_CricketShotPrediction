# Vision Quest — Video Scripts (Code Walkthrough + Demo)

Both videos: **OBS**, screen share + webcam on, **both members visible and audible**.
Speak to the beats, don't read verbatim. Keep files as `.mp4`.

====================================================================
## A. CODE-WALKTHROUGH VIDEO  (hard cap 10 min — aim 8–9)
====================================================================
Split roughly in half between the two members. Have the repo open in an editor.

### [0:00–0:30] Intro (Member 1, on cam)
"We're Team Vision Quest. This is the code walkthrough for early prediction of
batting shot-type from cricket videos. The repo is config-driven; everything
runs from `configs/default.yaml`. I'll cover data and models, [Member 2] covers
training, loss, and evaluation."

### [0:30–2:00] Data pipeline (Member 1) — open `src/data/`
- `download.py`: pulls the Kaggle IPL-2023 set and `--inspect` prints class
  folders + counts so we confirm the 7-class taxonomy.
- `extract_frames.py`: builds the clip index, a **stratified** 70/15/15 split,
  and caches **20 uniformly-sampled frames** per clip at 112×112. Show
  `stratified_split()` and the frame-sampling (`np.linspace`).
- `extract_pose.py`: runs **MediaPipe BlazePose**, 33 keypoints/frame, and
  `normalize_pose()` — centre on mid-hip, scale by torso length (point to it).

### [2:00–3:00] Partial observation (Member 1) — `src/data/dataset.py`
"This is the core idea." Show `seen_count()` and, in `VideoClipDataset`, how for
ratio g we keep the first k frames and mask the future by repeating frame k.
Note the training datasets draw a **random ratio per sample** so one model
serves all horizons. Show `build_player_vocab()` for the player-specific ids.

### [3:00–4:30] Models (Member 1) — `src/models/`
- `video_branch.py`: R3D-18 (Kinetics-pretrained), FC replaced by 7-class head;
  mention the CNN-GRU alternative.
- `pose_branch.py`: the compact **ST-GCN** — show `build_adjacency()` (the
  skeleton graph), the learnable `edge_imp` mask, the temporal conv, and the
  optional **player embedding** concatenated before the classifier.

### [4:30–7:00] Training + loss (Member 2) — `src/train/train_video.py`
- Walk through the loop. **Loss:** point to the earliness term:
  `L = CE(x) + λ·CE(x_early)` — explain it masks future frames and adds CE on the
  forced-early view, rewarding correct early predictions (Aliakbarian 2017).
- **Optimiser:** AdamW, lr 3e-4, weight decay 1e-4; **cosine schedule with
  2-epoch warmup** (show `build_optim`); gradient clip 1.0; mixed precision;
  early stopping on val Top-1.
- `train_pose.py`: same shape, higher lr, player ids passed to the model.

### [7:00–8:30] Evaluation (Member 2) — `src/eval/`
- `metrics.py`: `topk_accuracy`, `per_class_f1`, `mean_reciprocal_rank`, and
  `earliness_score` (smallest ratio within 3% of full-clip Top-1) — read that one
  out, it's our headline metric.
- `temporal_analysis.py`: loops every observation ratio for both branches,
  writes `temporal_results.json` and the three figures (accuracy-vs-ratio,
  confusion, per-class F1).

### [8:30–9:00] Wrap (both)
"`scripts/run_all.sh` runs the whole thing; `smoke_test.py` verifies it on
synthetic data. The interface is next in the demo video." Mention the GitHub link.

====================================================================
## B. DEMO VIDEO  (hard cap 5 min — aim 4:00–4:30)
====================================================================
Have `python -m app.gradio_app --share` already running; a few test clips ready.

### [0:00–0:30] Intro (Member 1, on cam)
"This is our test interface. It runs both branches — the video model and the
pose model — on an uploaded clip, and lets you control how much of the clip the
model sees."

### [0:30–2:00] Setting A — early video prediction (Member 1)
- Upload a clip. Set observation ratio to **50%**. Click Predict.
- Read out the **Top-3 / Top-5** with confidences.
- Move slider to **75%**, then **100%**. Narrate how the ranking/confidence
  sharpens — "the model is already [correct / close] at 50%".

### [2:00–3:30] Setting B — pose + confidence curve (Member 2)
- Point to the **pose overlay** (skeleton on the batter).
- Read the **likely / unlikely** list from the pose branch.
- Point to the **confidence-vs-observed-fraction** curve — "this shows how belief
  evolves as the shot unfolds; the true shot's line climbs as more is revealed".

### [3:30–4:30] Reviewer-driven test + close (both)
- "To show it's live, we'll take a clip we haven't used" — upload one, run it.
- "During the viva you can hand us any test clip and we'll run it here."
- Close: one line on the headline result (e.g. "Top-1 reaches X% at full clip and
  holds Y% at 50% observation").

**Tips:** keep the Gradio window large and readable; test your mic levels first;
if a prediction is wrong on a clip, keep it — explaining a failure case honestly
reads better than hiding it, and the rubric rewards understanding over polish.
