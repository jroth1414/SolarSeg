"""
imgstats_eda.py
================
EDA slice: image-level statistics for the MAGFiLO_1.0 solar-filament dataset.

Covers BOTH train/train_images (707 files) and test/test_images (180 files).

For every image on disk this script:
  - opens it with PIL and records width, height, mode, file size (bytes)
  - computes pixel-intensity mean/std/min/max (via numpy on the decoded array)
  - parses the filename YYYYMMDDHHMMSSII.jpeg into a capture datetime + a
    2-letter instrument/observatory code

It then produces:
  1. imgstats_per_image.csv          - one row per file (train+test), all stats above
  2. imgstats_dimension_check.csv    - summary of any images that are NOT 2048x2048 / mode 'L'
  3. imgstats_instrument_dist.png    - instrument-code bar chart, train vs test side by side
  4. imgstats_instrument_dist.csv    - the underlying counts
  5. imgstats_date_timeline.png      - capture-date histogram/scatter, train vs test
  6. imgstats_intensity_hist.png     - aggregate pixel-intensity histogram (train vs test)
  7. imgstats_intensity_summary.csv  - per-split intensity summary stats
  8. imgstats_sample_panel.png       - small multi-panel of raw sample images, different instrument codes

Run with:  python imgstats_eda.py
(paths are relative to the repo root D:/SolarSeg, but absolute paths are used
 internally so it can be run from any cwd)
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("D:/SolarSeg")
DATA_ROOT = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_DIR = DATA_ROOT / "train" / "train_images"
TEST_DIR = DATA_ROOT / "test" / "test_images"
OUT_DIR = REPO_ROOT / "notebooks" / "eda_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "imgstats"

FNAME_RE = re.compile(r"^(\d{14})([A-Za-z]{2})\.jpe?g$", re.IGNORECASE)


def parse_filename(fname: str):
    """Parse YYYYMMDDHHMMSSII.jpeg -> (datetime or None, instrument_code or None, parse_ok bool)."""
    m = FNAME_RE.match(fname)
    if not m:
        return None, None, False
    ts_str, code = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None, code, False
    return dt, code, True


def scan_dir(dir_path: Path, split_label: str):
    """Walk a directory of jpegs, return list of per-image stat dicts."""
    rows = []
    files = sorted(os.listdir(dir_path))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    print(f"[{split_label}] scanning {len(files)} files in {dir_path} ...")
    for i, fname in enumerate(files):
        fpath = dir_path / fname
        file_size_bytes = fpath.stat().st_size

        with Image.open(fpath) as im:
            width, height = im.size
            mode = im.mode
            # Load into a numpy array ONCE and reuse for all stats (per-image, not per-annotation,
            # but still: avoid decoding twice by converting right after open while the decoder
            # buffer is warm).
            arr = np.asarray(im)

        # If mode is unexpectedly RGB/RGBA, arr will be (H,W,3) or (H,W,4); flatten across
        # channels for the intensity stats but flag it separately in the dimension check.
        px_mean = float(arr.mean())
        px_std = float(arr.std())
        px_min = float(arr.min())
        px_max = float(arr.max())

        dt, instrument_code, parse_ok = parse_filename(fname)

        rows.append(dict(
            file_name=fname,
            split=split_label,
            is_train=int(split_label == "train"),
            is_test=int(split_label == "test"),
            width=width,
            height=height,
            mode=mode,
            n_channels=(1 if arr.ndim == 2 else arr.shape[2]),
            file_size_bytes=file_size_bytes,
            px_mean=px_mean,
            px_std=px_std,
            px_min=px_min,
            px_max=px_max,
            filename_parse_ok=parse_ok,
            capture_datetime=dt.isoformat() if dt is not None else None,
            capture_date=dt.date().isoformat() if dt is not None else None,
            instrument_code=instrument_code,
        ))

        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(files)}")

    return rows


def main():
    train_rows = scan_dir(TRAIN_DIR, "train")
    test_rows = scan_dir(TEST_DIR, "test")

    df = pd.DataFrame(train_rows + test_rows)
    df["capture_datetime"] = pd.to_datetime(df["capture_datetime"])

    # -----------------------------------------------------------------
    # 1. Per-image CSV
    # -----------------------------------------------------------------
    per_image_csv = OUT_DIR / f"{SLUG}_per_image.csv"
    df.to_csv(per_image_csv, index=False)
    print(f"wrote {per_image_csv} ({len(df)} rows)")

    # -----------------------------------------------------------------
    # 2. Dimension / mode consistency check
    # -----------------------------------------------------------------
    expected_w, expected_h, expected_mode = 2048, 2048, "L"
    df["is_expected_dim"] = (df["width"] == expected_w) & (df["height"] == expected_h)
    df["is_expected_mode"] = df["mode"] == expected_mode

    n_total = len(df)
    n_bad_dim = int((~df["is_expected_dim"]).sum())
    n_bad_mode = int((~df["is_expected_mode"]).sum())
    n_bad_parse = int((~df["filename_parse_ok"]).sum())

    exceptions = df[(~df["is_expected_dim"]) | (~df["is_expected_mode"]) | (~df["filename_parse_ok"])]
    dim_check_csv = OUT_DIR / f"{SLUG}_dimension_check.csv"
    exceptions.to_csv(dim_check_csv, index=False)

    print(f"Dimension check: {n_total} images total; "
          f"{n_bad_dim} not {expected_w}x{expected_h}; "
          f"{n_bad_mode} not mode '{expected_mode}'; "
          f"{n_bad_parse} filename parse failures.")
    print(f"wrote {dim_check_csv} ({len(exceptions)} exception rows)")

    # Split-level dimension/mode summary (for the text summary / sanity)
    dim_summary = df.groupby("split").agg(
        n=("file_name", "count"),
        n_2048x2048=("is_expected_dim", "sum"),
        n_mode_L=("is_expected_mode", "sum"),
        n_filename_parsed=("filename_parse_ok", "sum"),
        unique_widths=("width", lambda s: sorted(s.unique().tolist())),
        unique_heights=("height", lambda s: sorted(s.unique().tolist())),
        unique_modes=("mode", lambda s: sorted(s.unique().tolist())),
    )
    dim_summary_csv = OUT_DIR / f"{SLUG}_dimension_summary_by_split.csv"
    dim_summary.to_csv(dim_summary_csv)
    print(f"wrote {dim_summary_csv}")

    # -----------------------------------------------------------------
    # 3. Instrument-code distribution: train vs test bar chart
    # -----------------------------------------------------------------
    inst_counts = (
        df.groupby(["split", "instrument_code"])
        .size()
        .rename("count")
        .reset_index()
    )
    inst_pivot = inst_counts.pivot(index="instrument_code", columns="split", values="count").fillna(0)
    for col in ("train", "test"):
        if col not in inst_pivot.columns:
            inst_pivot[col] = 0
    inst_pivot["train_frac"] = inst_pivot["train"] / inst_pivot["train"].sum()
    inst_pivot["test_frac"] = inst_pivot["test"] / inst_pivot["test"].sum()
    inst_pivot = inst_pivot.sort_values("train", ascending=False)

    inst_csv = OUT_DIR / f"{SLUG}_instrument_dist.csv"
    inst_pivot.to_csv(inst_csv)
    print(f"wrote {inst_csv}")
    print("Instrument code distribution (train vs test):")
    print(inst_pivot)

    fig, ax = plt.subplots(figsize=(10, 5))
    codes = inst_pivot.index.tolist()
    x = np.arange(len(codes))
    width_bar = 0.38
    ax.bar(x - width_bar / 2, inst_pivot["train_frac"] * 100, width_bar, label=f"train (n={len(train_rows)})", color="#3b6fa0")
    ax.bar(x + width_bar / 2, inst_pivot["test_frac"] * 100, width_bar, label=f"test (n={len(test_rows)})", color="#d97c3b")
    ax.set_xticks(x)
    ax.set_xticklabels(codes)
    ax.set_xlabel("Instrument / observatory code")
    ax.set_ylabel("% of split")
    ax.set_title("Instrument-code distribution: train vs test (normalized within split)")
    ax.legend()
    for i, c in enumerate(codes):
        ax.text(i - width_bar / 2, inst_pivot["train_frac"].iloc[i] * 100 + 0.5,
                 str(int(inst_pivot["train"].iloc[i])), ha="center", fontsize=8)
        ax.text(i + width_bar / 2, inst_pivot["test_frac"].iloc[i] * 100 + 0.5,
                 str(int(inst_pivot["test"].iloc[i])), ha="center", fontsize=8)
    fig.tight_layout()
    inst_png = OUT_DIR / f"{SLUG}_instrument_dist.png"
    fig.savefig(inst_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {inst_png}")

    # -----------------------------------------------------------------
    # 4. Capture-date timeline: train vs test
    # -----------------------------------------------------------------
    df_dt = df.dropna(subset=["capture_datetime"]).copy()
    train_dt = df_dt[df_dt.split == "train"]["capture_datetime"]
    test_dt = df_dt[df_dt.split == "test"]["capture_datetime"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw=dict(height_ratios=[2, 1]))

    ax0 = axes[0]
    # Common date bins across both splits
    all_dates_num = matplotlib.dates.date2num(df_dt["capture_datetime"])
    bins = np.linspace(all_dates_num.min(), all_dates_num.max(), 60)
    ax0.hist(matplotlib.dates.date2num(train_dt), bins=bins, alpha=0.6, label=f"train (n={len(train_dt)})", color="#3b6fa0")
    ax0.hist(matplotlib.dates.date2num(test_dt), bins=bins, alpha=0.6, label=f"test (n={len(test_dt)})", color="#d97c3b")
    ax0.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax0.set_ylabel("count")
    ax0.set_title("Capture-date histogram: train vs test")
    ax0.legend()

    ax1 = axes[1]
    ax1.scatter(train_dt, np.zeros(len(train_dt)) + 1, s=8, alpha=0.5, color="#3b6fa0", label="train")
    ax1.scatter(test_dt, np.zeros(len(test_dt)) + 0, s=8, alpha=0.5, color="#d97c3b", label="test")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["test", "train"])
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlabel("capture date")
    ax1.set_title("Per-image capture date scatter (interleaving check)")
    fig.autofmt_xdate()
    fig.tight_layout()
    date_png = OUT_DIR / f"{SLUG}_date_timeline.png"
    fig.savefig(date_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {date_png}")

    date_summary = df_dt.groupby("split")["capture_datetime"].agg(["min", "max", "count"])
    date_summary_csv = OUT_DIR / f"{SLUG}_date_summary.csv"
    date_summary.to_csv(date_summary_csv)
    print(f"wrote {date_summary_csv}")
    print("Date range by split:")
    print(date_summary)

    # -----------------------------------------------------------------
    # 5. Pixel-intensity distribution (aggregate histogram, train vs test)
    # -----------------------------------------------------------------
    intensity_summary = df.groupby("split")[["px_mean", "px_std", "px_min", "px_max"]].agg(
        ["mean", "std", "min", "max"]
    )
    intensity_summary_csv = OUT_DIR / f"{SLUG}_intensity_summary.csv"
    intensity_summary.to_csv(intensity_summary_csv)
    print(f"wrote {intensity_summary_csv}")
    print("Per-image intensity summary by split:")
    print(intensity_summary)

    # Build a TRUE aggregate pixel-value histogram by re-reading a random
    # subsample of images per split and pooling raw pixel values (reading all
    # 887 full-res images' full pixel arrays into one histogram is wasteful;
    # a stratified random subsample is enough to characterize the aggregate
    # distribution shape).
    rng = np.random.default_rng(42)
    N_SAMPLE_PER_SPLIT = 60

    def sample_pixel_values(split_df, n_sample):
        chosen = split_df.sample(n=min(n_sample, len(split_df)), random_state=42)
        vals = []
        for _, row in chosen.iterrows():
            split_dir = TRAIN_DIR if row["split"] == "train" else TEST_DIR
            with Image.open(split_dir / row["file_name"]) as im:
                arr = np.asarray(im)
            # subsample pixels within the image too, to keep memory bounded
            flat = arr.reshape(-1)
            if flat.size > 200_000:
                idx = rng.choice(flat.size, size=200_000, replace=False)
                flat = flat[idx]
            vals.append(flat)
        return np.concatenate(vals)

    train_px_sample = sample_pixel_values(df[df.split == "train"], N_SAMPLE_PER_SPLIT)
    test_px_sample = sample_pixel_values(df[df.split == "test"], N_SAMPLE_PER_SPLIT)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax0, ax1 = axes
    bins = np.linspace(0, 255, 128)
    ax0.hist(train_px_sample, bins=bins, alpha=0.6, density=True, label="train", color="#3b6fa0")
    ax0.hist(test_px_sample, bins=bins, alpha=0.6, density=True, label="test", color="#d97c3b")
    ax0.set_xlabel("pixel intensity (0-255)")
    ax0.set_ylabel("density")
    ax0.set_title(f"Aggregate pixel-intensity histogram\n(subsample: {N_SAMPLE_PER_SPLIT} imgs/split, up to 200k px/img)")
    ax0.legend()

    ax1.hist(train_px_sample, bins=bins, alpha=0.6, density=True, label="train", color="#3b6fa0")
    ax1.hist(test_px_sample, bins=bins, alpha=0.6, density=True, label="test", color="#d97c3b")
    ax1.set_yscale("log")
    ax1.set_xlabel("pixel intensity (0-255)")
    ax1.set_ylabel("density (log scale)")
    ax1.set_title("Same histogram, log-y (reveals dark-tail / bimodal structure)")
    ax1.legend()

    fig.tight_layout()
    intensity_png = OUT_DIR / f"{SLUG}_intensity_hist.png"
    fig.savefig(intensity_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {intensity_png}")

    # -----------------------------------------------------------------
    # 6. Sample panel of raw images across different instrument codes
    # -----------------------------------------------------------------
    # Pick up to 6 train images spanning distinct instrument codes (by count desc),
    # plus 2 test images, unmodified display.
    codes_by_count = inst_pivot.sort_values("train", ascending=False).index.tolist()
    sample_rows = []
    for code in codes_by_count:
        cand = df[(df.split == "train") & (df.instrument_code == code)]
        if len(cand) > 0:
            sample_rows.append(cand.iloc[0])
        if len(sample_rows) >= 6:
            break
    # add 2 test samples (different codes if possible)
    test_codes_used = set()
    for _, r in df[df.split == "test"].iterrows():
        if r["instrument_code"] not in test_codes_used:
            sample_rows.append(r)
            test_codes_used.add(r["instrument_code"])
        if len(test_codes_used) >= 2:
            break

    n_panels = len(sample_rows)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, row in zip(axes, sample_rows):
        split_dir = TRAIN_DIR if row["split"] == "train" else TEST_DIR
        with Image.open(split_dir / row["file_name"]) as im:
            arr = np.asarray(im)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{row['file_name']}\n[{row['split']}] code={row['instrument_code']}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Raw sample images by instrument code (unmodified, no overlays)", fontsize=13)
    fig.tight_layout()
    sample_png = OUT_DIR / f"{SLUG}_sample_panel.png"
    fig.savefig(sample_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {sample_png}")

    # -----------------------------------------------------------------
    # Console summary of headline numbers (also useful when re-running)
    # -----------------------------------------------------------------
    print("\n==== HEADLINE NUMBERS ====")
    print(f"train images: {len(train_rows)}, test images: {len(test_rows)}")
    print(f"images NOT 2048x2048: {n_bad_dim}")
    print(f"images NOT mode 'L': {n_bad_mode}")
    print(f"filename parse failures: {n_bad_parse}")
    print(f"train instrument codes: {sorted(df[df.split=='train'].instrument_code.dropna().unique().tolist())}")
    print(f"test instrument codes: {sorted(df[df.split=='test'].instrument_code.dropna().unique().tolist())}")
    codes_train = set(df[df.split == "train"].instrument_code.dropna().unique())
    codes_test = set(df[df.split == "test"].instrument_code.dropna().unique())
    print(f"codes in test but not train: {codes_test - codes_train}")
    print(f"codes in train but not test: {codes_train - codes_test}")
    print(f"train date range: {train_dt.min()} to {train_dt.max()}")
    print(f"test date range: {test_dt.min()} to {test_dt.max()}")
    print(f"train px_mean overall: {df[df.split=='train'].px_mean.mean():.2f} +/- {df[df.split=='train'].px_mean.std():.2f}")
    print(f"test  px_mean overall: {df[df.split=='test'].px_mean.mean():.2f} +/- {df[df.split=='test'].px_mean.std():.2f}")


if __name__ == "__main__":
    main()
