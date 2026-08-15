"""5-fold ensemble submission: average flip-TTA probability maps across all
fold checkpoints, then apply the tuned post-processing once to the mean map.

All fold models share one recipe (equal strength), which is the regime where
probability-map averaging measured positively -- unlike the refuted
unequal-strength 2-model ensemble (see reports/experiments.md #8c).

Usage:
    python scripts/predict_ensemble.py --out submissions/unet_5fold_tta.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.eval.postproc import probs_to_instances
from src.eval.submission import build_submission_csv, rle_string_to_mask
from src.models.unet import build_unet
from scripts.eval_tta import tta_prob

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+",
                    default=[str(REPO_ROOT / "checkpoints" / f"unet_f{i}.pt") for i in range(5)])
    ap.add_argument("--out", default=str(REPO_ROOT / "submissions" / "unet_5fold_tta.csv"))
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=400)
    ap.add_argument("--closing", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | {len(args.checkpoints)} checkpoints | "
          f"pp: thr={args.threshold} min_area={args.min_area} closing={args.closing}")

    models = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = build_unet(encoder_name=ckpt["encoder"], encoder_weights=None).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  loaded {path} (epoch {ckpt['epoch']}, val_pq {ckpt['val_pq']:.4f})")

    test_paths = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
    print(f"{len(test_paths)} test images")

    predictions = {}
    n_zero = 0
    t0 = time.time()
    with torch.no_grad():
        for i, path in enumerate(test_paths, 1):
            with Image.open(path) as im:
                arr = np.array(im.convert("L"), dtype=np.uint8)
            x = torch.from_numpy(arr).float().div_(255.0)[None, None].to(device)
            prob = None
            for m in models:
                p = tta_prob(m, x, device)
                prob = p if prob is None else prob + p
            prob = (prob / len(models)).cpu().numpy()

            instances = probs_to_instances(
                prob, threshold=args.threshold, min_area=args.min_area,
                closing=args.closing, non_empty_guard=True,
            )
            predictions[path.stem] = instances
            if not instances:
                n_zero += 1
            if i % 30 == 0 or i == len(test_paths):
                print(f"  {i}/{len(test_paths)} ({time.time() - t0:.0f}s)")

    total_rows = sum(len(v) for v in predictions.values())
    print(f"\n{len(models)}-model x 4-flip inference: {time.time() - t0:.0f}s | rows: {total_rows} "
          f"| zero-pred images: {n_zero}/{len(test_paths)}")

    out_path = Path(args.out)
    build_submission_csv(predictions, out_path)
    print(f"wrote {out_path}")

    import csv
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    idx = np.linspace(0, len(rows) - 1, num=min(5, len(rows)), dtype=int)
    ok = True
    for j in idx:
        r = rows[int(j)]
        m = rle_string_to_mask(r["segmentation_rle"], 2048, 2048)
        good = m.shape == (2048, 2048) and set(np.unique(m)).issubset({0, 1})
        print(f"  [{'PASS' if good else 'FAIL'}] {r['filament_id']}: fg={int(m.sum())}")
        ok &= good
    print("SPOT-CHECK:", "ALL PASS" if ok else "FAILED")


if __name__ == "__main__":
    main()
