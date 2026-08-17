"""Post-processing sweep on cached U-Net validation probability maps (CPU-only).

Grid-searches threshold x min_area (x optional closing x per-CC mean-prob
filter) against val PQ, using probability maps cached by scripts/train_unet.py
under checkpoints/<run>_valprobs/. Model inference is NOT re-run, so a full
grid costs minutes of CPU, no GPU -- re-run this after every model change
(the optimum drifts).

Usage:
    python scripts/sweep_postproc.py [--probs checkpoints/unet_v1_valprobs]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.semantic_dataset import load_entries
from src.eval.metrics import compute_panoptic_quality
from src.eval.postproc import probs_to_instances
from scripts.train_unet import entry_instance_masks

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = (REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train"
             / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

# Trimmed grid informed by the first full sweep on unet_v1 (294 configs, 3.9h):
# min_mean_prob had ZERO effect at any value (surviving components are all
# high-confidence) so that axis is dropped; the optimum sat at thr 0.45-0.50 /
# min_area 300-400 with closing=True marginally ahead, so the grid centers there
# and extends the min_area direction that was still improving at the old edge.
THRESHOLDS = [0.40, 0.45, 0.50, 0.55]
MIN_AREAS = [200, 300, 400, 500, 600]
CLOSINGS = [False, True]
MIN_MEAN_PROBS = [0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=str(REPO_ROOT / "checkpoints" / "unet_v1_valprobs"))
    args = ap.parse_args()
    probs_dir = Path(args.probs)

    prob_files = sorted(probs_dir.glob("*.npy"))
    if not prob_files:
        sys.exit(f"no cached probability maps in {probs_dir} -- run train_unet.py first")
    print(f"{len(prob_files)} cached probability maps from {probs_dir}")

    entries_by_id = {e["id"]: e for e in load_entries(JSON_PATH)}

    configs = [
        {"threshold": thr, "min_area": min_area, "closing": closing, "min_mean_prob": mmp}
        for closing in CLOSINGS
        for thr in THRESHOLDS
        for min_area in MIN_AREAS
        for mmp in MIN_MEAN_PROBS
    ]
    print(f"sweeping {len(configs)} configs (streaming one image at a time)")

    # Stream images (outer loop) rather than preloading all probs + GT masks:
    # a full preload costs ~8GB RAM at 175 x 2048^2 and OOMs alongside a
    # concurrent training run. Per-image, all configs are evaluated while its
    # prob map and GT rasterization are in memory once.
    per_config_pqs = [[] for _ in configs]
    t0 = time.time()
    for n_done, f in enumerate(prob_files, 1):
        prob = np.load(f).astype(np.float32)
        gt = entry_instance_masks(entries_by_id[f.stem])
        for ci, cfg in enumerate(configs):
            pq = compute_panoptic_quality(
                gt, probs_to_instances(prob, **cfg))["pq"]
            per_config_pqs[ci].append(pq)
        if n_done % 25 == 0 or n_done == len(prob_files):
            print(f"  {n_done}/{len(prob_files)} images ({time.time() - t0:.0f}s elapsed)")

    results = [
        {**cfg, "pq": float(np.mean(pqs))} for cfg, pqs in zip(configs, per_config_pqs)
    ]

    results.sort(key=lambda r: -r["pq"])
    print("\n=== TOP 10 CONFIGS ===")
    for r in results[:10]:
        print(f"  PQ={r['pq']:.4f}  thr={r['threshold']:.2f} min_area={r['min_area']} "
              f"closing={r['closing']} min_mean_prob={r['min_mean_prob']}")
    best = results[0]
    print(f"\nBEST: {best}")


if __name__ == "__main__":
    main()
