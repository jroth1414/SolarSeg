"""Run the trained baseline_v1 checkpoint on the full (unlabeled) test set and write
the competition submission CSV.

Loads checkpoints/baseline_v1.pt (produced by scripts/train_baseline.py), which stores
both the model weights and the architecture config (num_classes/min_size/max_size) plus
the SCORE_THRESH/MASK_THRESH used at training-time evaluation -- reused here so
inference is scored identically between the reported val metrics and the submission.

Inference resolution note: exactly like scripts/train_baseline.py's evaluate(), test
images are passed to the model at their NATIVE 2048x2048 resolution; torchvision's
internal GeneralizedRCNNTransform downsamples for the backbone and automatically
upsamples predicted boxes/masks back to 2048x2048 during postprocessing, so no manual
resize step is needed here -- predicted masks come out already at the submission's
required 2048x2048 size.

Test images have NO ground-truth annotation file (data/.../test/test_images/ only,
no COCO json) -- this script lists the directory directly rather than going through
FilamentDataset/get_grouped_split (those are built around the train COCO json).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.eval.submission import build_submission_csv, rle_string_to_mask
from src.models.baseline import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "baseline_v1.pt"
OUT_CSV_PATH = REPO_ROOT / "submissions" / "baseline_v1.csv"

BATCH_SIZE = 2


def load_image(path: Path) -> torch.Tensor:
    """Load one test JPEG as a [1,H,W] float32 tensor in [0,1], matching
    FilamentDataset's image convention exactly (grayscale, not RGB-expanded)."""
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.array(im, dtype=np.uint8)
    return torch.from_numpy(arr).to(torch.float32).div(255.0).unsqueeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    min_size = ckpt["min_size"]
    max_size = ckpt["max_size"]
    score_thresh = ckpt["score_thresh"]
    mask_thresh = ckpt["mask_thresh"]
    print(
        f"  checkpoint config: num_classes={num_classes} min_size={min_size} "
        f"max_size={max_size} score_thresh={score_thresh} mask_thresh={mask_thresh}"
    )

    model = build_model(num_classes=num_classes, min_size=min_size, max_size=max_size, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_paths = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
    print(f"found {len(test_paths)} test images in {TEST_IMAGES_DIR}")

    predictions = {}  # image_id (stem) -> list of HxW uint8 binary masks
    n_zero_pred_images = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_start in range(0, len(test_paths), BATCH_SIZE):
            batch_paths = test_paths[batch_start : batch_start + BATCH_SIZE]
            images = [load_image(p).to(device) for p in batch_paths]
            preds = model(images)

            for path, pred in zip(batch_paths, preds):
                image_id = path.stem  # base file_name without extension, per submission format
                keep = pred["scores"] > score_thresh
                kept_masks = pred["masks"][keep]  # (K, 1, H, W) soft masks at native 2048x2048
                masks = [(m[0].cpu().numpy() > mask_thresh).astype(np.uint8) for m in kept_masks]
                predictions[image_id] = masks
                if len(masks) == 0:
                    n_zero_pred_images += 1

            done = batch_start + len(batch_paths)
            if done % 20 == 0 or done == len(test_paths):
                elapsed = time.time() - t0
                print(f"  processed {done}/{len(test_paths)} images ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t0
    print(f"\ninference wall-clock: {total_time:.1f}s ({total_time / 60:.1f} min)")

    total_rows = sum(len(m) for m in predictions.values())
    print(f"total predicted filament instances (rows to write): {total_rows}")
    print(f"test images with 0 predicted filaments: {n_zero_pred_images}/{len(test_paths)}")

    build_submission_csv(predictions, OUT_CSV_PATH)
    print(f"wrote submission CSV: {OUT_CSV_PATH}")

    # ---- spot-check a handful of written rows decode cleanly ----
    print("\n=== spot-check: decode a few written rows back to 2048x2048 binary masks ===")
    import csv

    with open(OUT_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"CSV has {len(rows)} data rows (header not counted)")

    if len(rows) == 0:
        print("  (no rows written -- nothing to spot-check)")
    else:
        sample_idx = np.linspace(0, len(rows) - 1, num=min(5, len(rows)), dtype=int)
        all_ok = True
        for i in sample_idx:
            row = rows[int(i)]
            fid = row["filament_id"]
            rle_str = row["segmentation_rle"]
            try:
                mask = rle_string_to_mask(rle_str, 2048, 2048)
                ok = mask.shape == (2048, 2048) and mask.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 1})
                pixels = int(mask.sum())
            except Exception as e:  # noqa: BLE001
                ok = False
                pixels = None
                print(f"  [FAIL] {fid}: decode raised {e!r}")
            if ok:
                print(f"  [PASS] {fid}: decoded shape=(2048,2048) fg_pixels={pixels}")
            all_ok &= ok
        print("SPOT-CHECK:", "ALL PASS" if all_ok else "SOME FAILED")


if __name__ == "__main__":
    main()
