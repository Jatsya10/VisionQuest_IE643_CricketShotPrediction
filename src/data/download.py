"""
Download + inspect the Kaggle 'Cricket Shots - IPL 2023' dataset.

Usage
-----
# 1. Put your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600), then:
python -m src.data.download --config configs/default.yaml

# Already have the data unzipped somewhere? Skip the download:
python -m src.data.download --inspect --raw data/raw/cricket-shots-ipl-2023

Notes
-----
* On Colab:  from google.colab import files; files.upload()  # kaggle.json
             !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
* The exact Kaggle slug is taarunsridhar/cricket-shots-ipl-2023.
* This script does NOT hard-fail if Kaggle is unreachable — it tells you
  how to download manually and point --raw at the folder.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.utils.common import load_config, get_logger

log = get_logger("download")

KAGGLE_SLUG = "taarunsridhar/cricket-shots-ipl-2023"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def download_kaggle(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle  # noqa: F401  (import validates credentials exist)
    except Exception as e:
        log.error("kaggle package/credentials not ready: %s", e)
        log.error("Install: pip install kaggle ; put token at ~/.kaggle/kaggle.json")
        log.error("Or download manually from https://www.kaggle.com/datasets/%s and unzip to %s", KAGGLE_SLUG, dest)
        sys.exit(1)
    log.info("Downloading %s -> %s", KAGGLE_SLUG, dest)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(dest), "--unzip"]
    subprocess.run(cmd, check=True)
    log.info("Download complete.")


def inspect(raw: Path):
    """Print class folders + clip counts so classes in config can be verified."""
    if not raw.exists():
        log.error("Path does not exist: %s", raw)
        sys.exit(1)
    # The dataset may nest one level (e.g. raw/cricket-shots/<class>/*.mp4).
    # Find the directory whose subdirs contain the most video files.
    candidates = [raw] + [p for p in raw.rglob("*") if p.is_dir()]
    best, best_count = None, -1
    for c in candidates:
        subdirs = [d for d in c.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        cnt = sum(1 for d in subdirs for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        if cnt > best_count:
            best, best_count = c, cnt
    if best is None:
        log.error("No class-subfolder structure with videos found under %s", raw)
        sys.exit(1)
    log.info("Detected class root: %s", best)
    counts = Counter()
    for d in sorted(best.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        counts[d.name] = n
    log.info("Classes found (%d): ", len(counts))
    for name, n in counts.most_common():
        log.info("    %-18s %4d clips", name, n)
    log.info("Total clips: %d", sum(counts.values()))
    log.info("-> Copy these folder names into configs/default.yaml 'classes:' if they differ.")
    return best, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw", default=None, help="override raw dataset path")
    ap.add_argument("--inspect", action="store_true", help="only inspect an existing folder")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    raw = Path(args.raw or cfg.paths.raw_dataset)

    if not args.inspect:
        if raw.exists() and any(raw.rglob("*.mp4")):
            log.info("Data already present at %s — skipping download.", raw)
        else:
            download_kaggle(raw)
    inspect(raw)


if __name__ == "__main__":
    main()
