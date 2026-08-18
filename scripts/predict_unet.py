"""Generate a competition submission from a trained U-Net checkpoint.

Runs full-resolution 2048^2 inference on all test images, converts each
probability map to instance masks via the SAME probs_to_instances step used
for validation scoring (threshold / min_area / closing / min_mean_prob /
non-empty guard), and writes the RLE submission CSV.

Post-processing defaults come from the checkpoint's stored values; override
with CLI flags after running scripts/sweep_postproc.py, e.g.:

    python scripts/predict_unet.py --threshold 0.45 --min-area 150 --out submissions/unet_v1.csv
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

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "unet_v1.pt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "submissions" / "unet_v1.csv"))
    ap.add_argument("--threshold", type=float, default=None, help="override checkpoint pp_threshold")
    ap.add_argument("--min-area", type=int, default=None, help="override checkpoint pp_min_area")
    ap.add_argument("--min-mean-prob", type=float, default=0.0)
    ap.add_argument("--closing", action="store_true")
    ap.add_argument("--tta", action="store_true",
                    help="average probabilities over 4 flip variants (measured +0.010 val PQ)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    threshold = args.threshold if args.threshold is not None else ckpt["pp_threshold"]
    min_area = args.min_area if args.min_area is not None else ckpt["pp_min_area"]
    print(f"checkpoint: {args.checkpoint} (encoder={ckpt['encoder']}, epoch={ckpt['epoch']}, "
          f"val_pq={ckpt['val_pq']:.4f})")
    print(f"post-processing: threshold={threshold} min_area={min_area} "
          f"min_mean_prob={args.min_mean_prob} closing={args.closing}")

    model = build_unet(encoder_name=ckpt["encoder"], encoder_weights=None,
                       classes=ckpt.get("classes", 1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
            if args.tta:
                from scripts.eval_tta import tta_prob
                prob = tta_prob(model, x, device).cpu().numpy()
            else:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = model(x)
                prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

            instances = probs_to_instances(
                prob, threshold=threshold, min_area=min_area,
                min_mean_prob=args.min_mean_prob, closing=args.closing,
                non_empty_guard=True,
            )
            predictions[path.stem] = instances
            if not instances:
                n_zero += 1
            if i % 30 == 0 or i == len(test_paths):
                print(f"  {i}/{len(test_paths)} ({time.time() - t0:.1f}s)")

    total_rows = sum(len(v) for v in predictions.values())
    print(f"\ninference+postproc: {time.time() - t0:.1f}s | rows: {total_rows} | "
          f"images with 0 predictions: {n_zero}/{len(test_paths)}")

    out_path = Path(args.out)
    build_submission_csv(predictions, out_path)
    print(f"wrote {out_path}")

    # spot-check decode
    import csv
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    idx = np.linspace(0, len(rows) - 1, num=min(5, len(rows)), dtype=int)
    ok = True
    for i in idx:
        r = rows[int(i)]
        m = rle_string_to_mask(r["segmentation_rle"], 2048, 2048)
        good = m.shape == (2048, 2048) and set(np.unique(m)).issubset({0, 1})
        print(f"  [{'PASS' if good else 'FAIL'}] {r['filament_id']}: fg={int(m.sum())}")
        ok &= good
    print("SPOT-CHECK:", "ALL PASS" if ok else "FAILED")


if __name__ == "__main__":
    main()
