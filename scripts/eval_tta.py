"""Flip-TTA evaluation: average probability maps over {identity, hflip, vflip,
hvflip} and cache the averaged maps for the post-processing sweep.

Reads a U-Net checkpoint, runs 4x test-time-augmented inference on the val
split, scores PQ/Dice with the checkpoint's stored post-processing config, and
caches averaged prob maps to checkpoints/<run>_tta_valprobs/ so
scripts/sweep_postproc.py can re-tune thresholds on the smoother maps.

Usage:
    python scripts/eval_tta.py --checkpoint checkpoints/unet_v2.pt --run-name unet_v2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.coco_dataset import get_grouped_split
from src.data.semantic_dataset import SemanticFilamentDataset, semantic_collate
from src.eval.metrics import compute_dice_per_image, compute_panoptic_quality
from src.eval.postproc import probs_to_instances
from src.models.unet import build_unet
from scripts.train_unet import entry_instance_masks

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
JSON_PATH = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGES_DIR = DATA_ROOT / "train" / "train_images"


def tta_prob(model, x, device):
    """Mean sigmoid probability over the 4 flip variants of x ([1,1,H,W])."""
    probs = None
    for flip_h in (False, True):
        for flip_v in (False, True):
            xi = x
            if flip_h:
                xi = torch.flip(xi, dims=[-1])
            if flip_v:
                xi = torch.flip(xi, dims=[-2])
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(xi)
            p = torch.sigmoid(logits.float())
            if flip_h:
                p = torch.flip(p, dims=[-1])
            if flip_v:
                p = torch.flip(p, dims=[-2])
            probs = p if probs is None else probs + p
    return (probs / 4.0)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "unet_v2.pt"))
    ap.add_argument("--run-name", default="unet_v2")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--closing", action="store_true")
    ap.add_argument("--kfold", type=int, default=None,
                    help="evaluate on get_grouped_kfold(k, fold) val split instead of the fixed split")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    threshold = args.threshold if args.threshold is not None else ckpt["pp_threshold"]
    min_area = args.min_area if args.min_area is not None else ckpt["pp_min_area"]
    print(f"device: {device} | checkpoint: {args.checkpoint} (val_pq {ckpt['val_pq']:.4f}) | "
          f"pp: thr={threshold} min_area={min_area} closing={args.closing}")

    model = build_unet(encoder_name=ckpt["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.kfold:
        from src.data.coco_dataset import get_grouped_kfold
        _train_ids, val_ids = get_grouped_kfold(JSON_PATH, k=args.kfold, fold=args.fold, seed=0)
    else:
        _train_ids, val_ids = get_grouped_split(JSON_PATH, val_fraction=0.15, seed=0)
    val_ds = SemanticFilamentDataset(JSON_PATH, IMAGES_DIR, image_ids=val_ids, train=False)
    entries_by_id = {e["id"]: e for e in val_ds.entries}
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=semantic_collate)

    out_dir = REPO_ROOT / "checkpoints" / f"{args.run_name}_tta_valprobs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pq_list, dice_list = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (images, targets, ids) in enumerate(val_loader, 1):
            x = images.to(device)
            prob = tta_prob(model, x, device).cpu().numpy()
            gt_instances = entry_instance_masks(entries_by_id[ids[0]])
            gt_union = targets[0, 0].numpy().astype(np.uint8)

            preds = probs_to_instances(prob, threshold=threshold, min_area=min_area,
                                       closing=args.closing, non_empty_guard=True)
            pq_list.append(compute_panoptic_quality(gt_instances, preds))
            dice_list.append(compute_dice_per_image([gt_union], preds))
            np.save(out_dir / f"{ids[0]}.npy", prob.astype(np.float16))
            if i % 50 == 0 or i == len(val_loader):
                print(f"  {i}/{len(val_loader)} ({time.time() - t0:.0f}s)")

    pq = float(np.mean([x["pq"] for x in pq_list]))
    sq = float(np.mean([x["sq"] for x in pq_list]))
    rq = float(np.mean([x["rq"] for x in pq_list]))
    tp = sum(x["tp"] for x in pq_list); fp = sum(x["fp"] for x in pq_list); fn = sum(x["fn"] for x in pq_list)
    print(f"\nTTA val: PQ={pq:.4f} SQ={sq:.4f} RQ={rq:.4f} Dice={float(np.mean(dice_list)):.4f} "
          f"(TP={tp} FP={fp} FN={fn})")
    print(f"cached {len(list(out_dir.glob('*.npy')))} averaged prob maps -> {out_dir}")


if __name__ == "__main__":
    main()
