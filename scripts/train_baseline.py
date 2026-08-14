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
     GT with compute_panoptic_quality / compute_dice_per_image at each threshold
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
from src.eval.metrics import compute_dice_per_image, compute_panoptic_quality
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

# --- Phase 0.2: full-data run (baseline-v1 used TRAIN_SUBSET_SIZE=250, EPOCHS=8) ---
TRAIN_SUBSET_SIZE = None  # full grouped train split (979 entries)
EPOCHS = 16  # ~490 steps/epoch at bs2 on full data, ~0.5s/step -> ~65-75 min total

BATCH_SIZE = 2
NUM_WORKERS = 0  # Windows-safe default; pycocotools polygon rasterization is the
                  # per-item cost and is fast enough at this dataset scale that
                  # worker-process overhead isn't worth it here.
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 10.0  # basic stability guard, not tuned

MIN_SIZE = 800
MAX_SIZE = 1024

# Eval sweeps the score threshold and the checkpoint stores the best one by val PQ
# (baseline-v1 measured 0.75 as optimal: PQ 0.319 @ 0.5 -> 0.343 @ 0.75).
SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
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


def train_one_epoch(model, loader, optimizer, device, epoch_idx, scheduler=None):
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

        if scheduler is not None:
            scheduler.step()

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
def evaluate_multi_threshold(model, loader, device, score_thresholds):
    """Run inference once, then score PQ/Dice at every score threshold in
    `score_thresholds` (one sweep for the price of one inference pass).

    Per-image metrics are computed incrementally so full-resolution masks for
    the whole val set never need to be held in memory at once. Returns
    {threshold: {"pq_per_image": [...], "dice_per_image": [...], "n_kept": int}}.
    """
    model.eval()
    results = {t: {"pq_per_image": [], "dice_per_image": [], "n_kept": 0} for t in score_thresholds}
    n_images = 0

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        preds = model(images_dev)

        for target, pred in zip(targets, preds):
            gt_masks = [m.numpy().astype(np.uint8) for m in target["masks"]]
            scores = pred["scores"].cpu().numpy()
            bin_masks = [
                (m[0].cpu().numpy() > MASK_THRESH).astype(np.uint8) for m in pred["masks"]
            ]

            for t in score_thresholds:
                kept = [m for m, s in zip(bin_masks, scores) if s > t]
                results[t]["n_kept"] += len(kept)
                results[t]["pq_per_image"].append(compute_panoptic_quality(gt_masks, kept))
                results[t]["dice_per_image"].append(compute_dice_per_image(gt_masks, kept))
            n_images += 1

    print(f"  evaluated {n_images} val images at {len(score_thresholds)} score thresholds")
    return results


def summarize_threshold_results(results):
    """Aggregate evaluate_multi_threshold output into per-threshold means and
    pick the best threshold by mean PQ."""
    summary = {}
    for t, r in results.items():
        pq_list = r["pq_per_image"]
        summary[t] = {
            "pq": float(np.mean([x["pq"] for x in pq_list])),
            "sq": float(np.mean([x["sq"] for x in pq_list])),
            "rq": float(np.mean([x["rq"] for x in pq_list])),
            "dice": float(np.mean(r["dice_per_image"])),
            "tp": int(sum(x["tp"] for x in pq_list)),
            "fp": int(sum(x["fp"] for x in pq_list)),
            "fn": int(sum(x["fn"] for x in pq_list)),
            "n_kept": r["n_kept"],
        }
    best_t = max(summary, key=lambda t: summary[t]["pq"])
    return summary, best_t


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
    # Cosine decay LR 1e-4 -> 1e-6 over the whole run, stepped per iteration.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6
    )

    # ---- train ----
    print(f"\n=== training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"{len(train_loader)} steps/epoch, AdamW lr={LR} wd={WEIGHT_DECAY} ===")
    history = []
    train_t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        mean_total, mean_components, epoch_time = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scheduler=scheduler
        )
        history.append({"epoch": epoch, "mean_total_loss": mean_total, **mean_components, "epoch_time_s": epoch_time})
        comp_str = " ".join(f"{k}={v:.4f}" for k, v in mean_components.items())
        print(f"[epoch {epoch}/{EPOCHS}] mean_total_loss={mean_total:.4f} ({comp_str}) time={epoch_time:.1f}s")
    train_total_time = time.time() - train_t0
    print(f"\ntotal training wall-clock: {train_total_time:.1f}s ({train_total_time / 60:.1f} min)")

    # ---- checkpoint (saved before eval for crash-safety; best threshold added after) ----
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_classes": 2,
        "min_size": MIN_SIZE,
        "max_size": MAX_SIZE,
        "score_thresh": 0.75,  # provisional; replaced by the sweep's best below
        "mask_thresh": MASK_THRESH,
        "train_subset_size": TRAIN_SUBSET_SIZE,
        "epochs": EPOCHS,
        "history": history,
    }
    torch.save(ckpt, CHECKPOINT_PATH)
    print(f"saved checkpoint: {CHECKPOINT_PATH}")

    # ---- evaluate on full val split, sweeping the score threshold ----
    print(f"\n=== evaluating on full val split ({len(val_ds)} images), "
          f"thresholds={SCORE_THRESHOLDS} ===")
    eval_t0 = time.time()
    results = evaluate_multi_threshold(model, val_loader, device, SCORE_THRESHOLDS)
    eval_time = time.time() - eval_t0
    print(f"eval wall-clock: {eval_time:.1f}s ({eval_time / 60:.1f} min)")

    summary, best_t = summarize_threshold_results(results)
    print("\n=== THRESHOLD SWEEP (full val split) ===")
    for t in SCORE_THRESHOLDS:
        s = summary[t]
        marker = "  <-- best" if t == best_t else ""
        print(f"  thr={t:.2f}: PQ={s['pq']:.4f} SQ={s['sq']:.4f} RQ={s['rq']:.4f} "
              f"Dice={s['dice']:.4f} TP={s['tp']} FP={s['fp']} FN={s['fn']}{marker}")

    best = summary[best_t]
    print(f"\n=== FINAL VALIDATION METRICS (score_thresh={best_t}) ===")
    print(f"  PQ   = {best['pq']:.4f}")
    print(f"  SQ   = {best['sq']:.4f}")
    print(f"  RQ   = {best['rq']:.4f}")
    print(f"  Dice = {best['dice']:.4f}")
    print(f"  (aggregate: TP={best['tp']} FP={best['fp']} FN={best['fn']})")

    # bake the winning threshold + sweep results into the checkpoint
    ckpt["score_thresh"] = float(best_t)
    ckpt["threshold_sweep"] = {str(t): summary[t] for t in SCORE_THRESHOLDS}
    torch.save(ckpt, CHECKPOINT_PATH)
    print(f"updated checkpoint with best score_thresh={best_t}: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
