"""Baseline Mask R-CNN training run for the MAGFiLO solar-filament task.

This is a deliberately BOUNDED baseline-v1 run, not a competitive training run:
  - Train split is subsampled down to TRAIN_SUBSET_SIZE images (deterministic,
    seed=SEED) out of the full grouped train split, so the whole pipeline
    (data -> model -> loss -> checkpoint -> eval -> submission) can be smoke-tested
    end-to-end in ~10-20 minutes of GPU wall-clock instead of a multi-hour run.
    TO REMOVE THE SUBSAMPLING FOR A REAL RUN: set TRAIN_SUBSET_SIZE = None below
    (or just delete the `_subsample` call and use `train_ids` directly) -- every
    other part of this script (model, optimizer, eval) is unaffected by that change.
  - EPOCHS is small (see constant below) for the same reason.
  - Validation is NOT subsampled: it always runs on the FULL grouped-out val split
    (get_grouped_split's val_ids, unmodified) so the reported PQ/Dice numbers are
    not cherry-picked on a tiny sample, per the task brief.

Pipeline:
  1. get_grouped_split(...) for a file_name-grouped train/val split (val_fraction=0.15,
     seed=0) -- see src/data/coco_dataset.py for why grouping matters (41.9% of
     images[] entries are duplicate-file_name / multi-annotator-batch rows).
  2. Deterministically subsample the train split to TRAIN_SUBSET_SIZE images.
  3. Train build_model(...) (src/models/baseline.py) for EPOCHS epochs with AdamW,
     logging per-epoch mean total loss + per-component means.
  4. Run inference (model.eval()) on the full val split, threshold predictions
     (score > SCORE_THRESH, per-pixel mask prob > MASK_THRESH), and score against
     GT with compute_panoptic_quality_dataset / compute_dice_dataset
     (src/eval/metrics.py).
  5. Save a checkpoint to checkpoints/baseline_v1.pt containing the model
     state_dict plus the architecture config needed to rebuild it
     (predict_and_submit.py reads this).

Design note on inference resolution: images are passed into the model at their
NATIVE 2048x2048 resolution (not pre-resized by this script). torchvision's
GeneralizedRCNNTransform internally downsamples to (min_size, max_size) for the
backbone/RPN/heads, then automatically upsamples predicted boxes/masks back to
the size of the images that were actually passed in (its recorded
"original_image_size") during postprocessing. So predicted masks come back
already at 2048x2048, directly comparable to GT masks with no manual resize step.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Allow `python scripts/train_baseline.py` to resolve `src.*` imports regardless
# of the invoking cwd (script-directory-relative sys.path[0] doesn't include the
# repo root by default).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.coco_dataset import (
    Compose,
    FilamentDataset,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    collate_fn,
    get_grouped_split,
)
from src.eval.metrics import compute_dice_dataset, compute_panoptic_quality_dataset
from src.models.baseline import build_model

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
JSON_PATH = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGES_DIR = DATA_ROOT / "train" / "train_images"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "baseline_v1.pt"

SEED = 0
VAL_FRACTION = 0.15

# --- deliberate baseline-v1 scope limits (see module docstring) ---
TRAIN_SUBSET_SIZE = 250  # None -> use the full grouped train split for a real run
EPOCHS = 8  # measured ~0.6s/step steady-state on the target GPU at batch_size=2 ->
            # ~125 steps/epoch * 0.6s ~= 75s/epoch -> ~10min total train time for 8
            # epochs, leaving room under the 10-20min budget for full-val-split eval

BATCH_SIZE = 2
NUM_WORKERS = 0  # Windows-safe default; pycocotools polygon rasterization is the
                  # per-item cost and is fast enough at this dataset scale that
                  # worker-process overhead isn't worth it here.
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 10.0  # basic stability guard, not tuned

MIN_SIZE = 800
MAX_SIZE = 1024

SCORE_THRESH = 0.5  # per-instance confidence filter (documented choice, per task brief)
MASK_THRESH = 0.5   # per-pixel probability -> binary mask threshold (standard Mask R-CNN convention)

LOG_EVERY = 20  # print a running-loss line every N training steps


def _subsample(ids, k, seed):
    """Deterministic subsample of `ids` down to `k` elements (or all of them if
    len(ids) <= k). Uses a seeded RNG separate from the split's own RNG."""
    if k is None or k >= len(ids):
        return list(ids)
    rng = random.Random(seed)
    return rng.sample(list(ids), k)


def _to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, loader, optimizer, device, epoch_idx):
    model.train()
    component_sums = {}
    total_sum = 0.0
    n_steps = 0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader):
        images, targets = _to_device(images, targets, device)

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        loss_val = float(total_loss.item())
        total_sum += loss_val
        n_steps += 1
        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + float(v.item())

        if (step + 1) % LOG_EVERY == 0 or (step + 1) == len(loader):
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch_idx} step {step + 1}/{len(loader)} "
                f"loss={loss_val:.4f} running_mean={total_sum / n_steps:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

    epoch_time = time.time() - t0
    mean_total = total_sum / max(n_steps, 1)
    mean_components = {k: v / max(n_steps, 1) for k, v in component_sums.items()}
    return mean_total, mean_components, epoch_time


@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference on every image in `loader`, threshold predictions, and
    return (gt_masks_per_image, pred_masks_per_image) in matching order, ready
    to hand to compute_panoptic_quality_dataset / compute_dice_dataset."""
    model.eval()
    gt_masks_per_image = []
    pred_masks_per_image = []
    n_images = 0
    n_pred_instances = 0

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        preds = model(images_dev)

        for target, pred in zip(targets, preds):
            gt_masks = [m.numpy().astype(np.uint8) for m in target["masks"]]

            keep = pred["scores"] > SCORE_THRESH
            kept_masks = pred["masks"][keep]  # (K, 1, H, W) soft masks, already at native res
            pred_masks = [
                (m[0].cpu().numpy() > MASK_THRESH).astype(np.uint8) for m in kept_masks
            ]

            gt_masks_per_image.append(gt_masks)
            pred_masks_per_image.append(pred_masks)
            n_images += 1
            n_pred_instances += len(pred_masks)

    print(f"  evaluated {n_images} val images, {n_pred_instances} total predicted instances "
          f"(score>{SCORE_THRESH}) kept")
    return gt_masks_per_image, pred_masks_per_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # all images are 2048x2048 -> consistent
                                                 # resized shape, benchmark mode pays off

    torch.manual_seed(SEED)

    # ---- split ----
    train_ids, val_ids = get_grouped_split(JSON_PATH, val_fraction=VAL_FRACTION, seed=SEED)
    print(f"full grouped split: train={len(train_ids)} val={len(val_ids)}")

    train_ids_subset = _subsample(train_ids, TRAIN_SUBSET_SIZE, seed=SEED)
    print(
        f"baseline-v1 scope limit: training on {len(train_ids_subset)}/{len(train_ids)} "
        f"train images (TRAIN_SUBSET_SIZE={TRAIN_SUBSET_SIZE}); "
        f"validating on the FULL {len(val_ids)}-image val split (not subsampled)"
    )

    train_transforms = Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
    train_ds = FilamentDataset(JSON_PATH, IMAGES_DIR, image_ids=train_ids_subset, transforms=train_transforms)
    val_ds = FilamentDataset(JSON_PATH, IMAGES_DIR, image_ids=val_ids, transforms=None)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn, drop_last=False,
    )

    # ---- model / optimizer ----
    model = build_model(num_classes=2, min_size=MIN_SIZE, max_size=MAX_SIZE, pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ---- train ----
    print(f"\n=== training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"{len(train_loader)} steps/epoch, AdamW lr={LR} wd={WEIGHT_DECAY} ===")
    history = []
    train_t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        mean_total, mean_components, epoch_time = train_one_epoch(model, train_loader, optimizer, device, epoch)
        history.append({"epoch": epoch, "mean_total_loss": mean_total, **mean_components, "epoch_time_s": epoch_time})
        comp_str = " ".join(f"{k}={v:.4f}" for k, v in mean_components.items())
        print(f"[epoch {epoch}/{EPOCHS}] mean_total_loss={mean_total:.4f} ({comp_str}) time={epoch_time:.1f}s")
    train_total_time = time.time() - train_t0
    print(f"\ntotal training wall-clock: {train_total_time:.1f}s ({train_total_time / 60:.1f} min)")

    # ---- checkpoint ----
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 2,
            "min_size": MIN_SIZE,
            "max_size": MAX_SIZE,
            "score_thresh": SCORE_THRESH,
            "mask_thresh": MASK_THRESH,
            "train_subset_size": TRAIN_SUBSET_SIZE,
            "epochs": EPOCHS,
            "history": history,
        },
        CHECKPOINT_PATH,
    )
    print(f"saved checkpoint: {CHECKPOINT_PATH}")

    # ---- evaluate on full val split ----
    print(f"\n=== evaluating on full val split ({len(val_ds)} images) ===")
    eval_t0 = time.time()
    gt_masks_per_image, pred_masks_per_image = evaluate(model, val_loader, device)
    eval_time = time.time() - eval_t0
    print(f"eval wall-clock: {eval_time:.1f}s ({eval_time / 60:.1f} min)")

    pq_result = compute_panoptic_quality_dataset(gt_masks_per_image, pred_masks_per_image)
    dice_result = compute_dice_dataset(gt_masks_per_image, pred_masks_per_image)

    print("\n=== FINAL VALIDATION METRICS (baseline_v1) ===")
    print(f"  PQ   = {pq_result['pq']:.4f}")
    print(f"  SQ   = {pq_result['sq']:.4f}")
    print(f"  RQ   = {pq_result['rq']:.4f}")
    print(f"  Dice = {dice_result:.4f}")

    total_tp = sum(r["tp"] for r in pq_result["per_image"])
    total_fp = sum(r["fp"] for r in pq_result["per_image"])
    total_fn = sum(r["fn"] for r in pq_result["per_image"])
    print(f"  (aggregate over val set: TP={total_tp} FP={total_fp} FN={total_fn})")


if __name__ == "__main__":
    main()
