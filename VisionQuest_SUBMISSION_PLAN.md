# Vision Quest — Intensive Assessment: Submission Plan & Checklist
**Stage-1 deadline: Mon 26 Oct 2026, 10:00 AM IST.** Work backwards from that.

Folder to create in Drive (IITB SSO), set to **"Anyone with the link"**:
`VisionQuest_IE643_CourseProject_IntensiveAssessment`

Must contain: **Report (PDF)**, **Slides (PDF/PPTX)**, **Code-walkthrough video**, **Demo video**.
Also: push code to **GitHub** and put the link in the report + slides.

---

## What's already built for you (in this repo / outputs)
- ✅ Full codebase (`VisionQuest_IE643_CricketShotPrediction/`) — **smoke-tested end-to-end**
- ✅ Gradio demo interface (`app/gradio_app.py`)
- ✅ Report LaTeX scaffold (`report/IE643_Project_Report.tex`) — **you write the prose**
- ✅ 18-slide deck (`VisionQuest_IE643_IntensiveAssessment_Slides.pptx`) — **you fill results**
- ✅ Colab runner notebook (`notebooks/colab_runner.ipynb`)
- ✅ This plan + video scripts

## What only you can do
- ❌ Record both videos (your faces + voices required by the rubric)
- ❌ Write report prose in your own words (plagiarism check = zero for AI text)
- ❌ Run training on Colab to get the real numbers

---

## Tonight's order of operations (realistic ~12h)

### Phase 1 — Train (start FIRST, it runs while you do everything else) · ~3–5h wall
1. Create a GitHub repo, push `VisionQuest_IE643_CricketShotPrediction/`.
2. Open `notebooks/colab_runner.ipynb` in Colab, set **GPU runtime**, put your repo URL in cell 1.
3. Upload `kaggle.json`, run cells 1→6. **Confirm the class names** printed in cell 3 match
   `configs/default.yaml` — edit `classes:` if they differ, then re-run.
4. If time is tight, reduce `--set train.epochs=15 data.num_frames=16` to finish faster.
5. When cell 6 prints `temporal_results.json`, download it + `results/figures/*`.

> If training is slow: train the **video branch only** first (it carries the main result),
> then pose. The temporal-analysis script runs with whichever checkpoints exist.

### Phase 2 — Fill results (30 min, after Phase 1)
- Open `results/temporal_results.json`. Copy numbers into:
  - Report Table (acc vs ratio), abstract, results inferences.
  - Slides 12–15 (drop in the 3 PNG figures + the table numbers).

### Phase 3 — Report (2–3h, parallel with training)
- Compile `report/IE643_Project_Report.tex` on **Overleaf**.
- **Rewrite every `\TODO` and body paragraph in your own words.** This is graded for
  plagiarism — do not submit the scaffold text as-is.
- Insert figures from `results/figures/`. Export **PDF**.

### Phase 4 — Videos (2–3h, after results exist)
- **Code walkthrough (≤10 min):** OBS, screen + webcam. Follow `VIDEO_SCRIPT` below.
  Both members on camera (split sections).
- **Demo (≤5 min):** run `python -m app.gradio_app --share`, screen-record uploading a
  clip and moving the observation-ratio slider. Both members visible.

### Phase 5 — Upload (30 min)
- Put Report.pdf, Slides.pdf, both videos in the Drive folder.
- Set folder sharing to **Anyone with the link**. Test in an incognito window.
- Submit the folder link in the Google Form.

---

## Risk cuts (if you run out of time, in priority order)
1. Pose branch is the first thing to simplify — if it won't train in time, report the
   **video branch** results fully and mark pose as "implemented, results pending" (the
   code is there and demoable).
2. Player-specific head: set `model.pose.player_embedding_dim=0` (global pose model) —
   simpler, trains faster, still valid.
3. `data.num_frames=16`, `train.epochs=12` to cut training time roughly in half.
4. Demo video can use a handful of test clips rather than a polished UI tour.

## Do NOT cut
- At least the **video branch** must produce real Top-k numbers at 50/75/100%.
- Both videos must exist (penalty / non-evaluation otherwise) and show faces.
- Report must be in the template and in your own words.
