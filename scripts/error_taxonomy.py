"""Error taxonomy for a set of cached validation probability maps (Phase 3).

Classifies every prediction/GT-instance error on the val split so Phase-3
post-processing effort targets the dominant bucket (also directly reportable:
the competition rubric scores fragmentation/merge statistics).

Buckets (per images-entry, aggregated):
- TP:           pred matched to GT at IoU>0.5 (PQ definition)
- near_miss:    unmatched GT whose best pred IoU is in (0.4, 0.5] -- threshold
                dilation/growing could recover these
- fragmented:   unmatched GT covered by >=2 preds each at IoU>0.1 -- CC-merge
                repair could recover these
- merged_pred:  pred overlapping >=2 GT instances at IoU>0.1 each -- watershed
                splitting could recover these
- pure_miss:    GT with no pred overlap at IoU>0.1 (model never saw it)
- hallucinated: pred with no GT overlap at IoU>0.1 (model invented it)

Usage:
    python scripts/error_taxonomy.py --probs checkpoints/unet_v2_tta_valprobs \
        --threshold 0.55 --min-area 400 --closing
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pycocotools import mask as maskUtils

from src.data.semantic_dataset import load_entries
from src.eval.postproc import probs_to_instances
from scripts.train_unet import entry_instance_masks

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = (REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train"
             / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")


def iou_matrix(pred_masks, gt_masks):
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)))
    p_rles = [maskUtils.encode(np.asfortranarray(m)) for m in pred_masks]
    g_rles = [maskUtils.encode(np.asfortranarray(m)) for m in gt_masks]
    return np.array(maskUtils.iou(p_rles, g_rles, [0] * len(g_rles))).reshape(
        len(pred_masks), len(gt_masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=str(REPO_ROOT / "checkpoints" / "unet_v2_tta_valprobs"))
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=400)
    ap.add_argument("--closing", action="store_true")
    args = ap.parse_args()

    entries_by_id = {e["id"]: e for e in load_entries(JSON_PATH)}
    prob_files = sorted(Path(args.probs).glob("*.npy"))
    print(f"{len(prob_files)} prob maps | pp: thr={args.threshold} "
          f"min_area={args.min_area} closing={args.closing}\n")

    counts = Counter()
    near_miss_ious = []
    for f in prob_files:
        entry = entries_by_id[f.stem]
        gt = entry_instance_masks(entry)
        prob = np.load(f).astype(np.float32)
        preds = probs_to_instances(prob, threshold=args.threshold, min_area=args.min_area,
                                   closing=args.closing, non_empty_guard=True)
        iou = iou_matrix(preds, gt)  # (P, G)

        matched_gt, matched_pred = set(), set()
        # PQ-style matching: IoU>0.5 pairs are unique
        for p in range(len(preds)):
            for g in range(len(gt)):
                if iou[p, g] > 0.5:
                    matched_gt.add(g)
                    matched_pred.add(p)
                    counts["TP"] += 1

        for g in range(len(gt)):
            if g in matched_gt:
                continue
            overlaps = iou[:, g] if len(preds) else np.array([])
            best = float(overlaps.max()) if overlaps.size else 0.0
            n_overlapping = int((overlaps > 0.1).sum()) if overlaps.size else 0
            if 0.4 < best <= 0.5:
                counts["near_miss"] += 1
                near_miss_ious.append(best)
            elif n_overlapping >= 2:
                counts["fragmented"] += 1
            elif n_overlapping == 0:
                counts["pure_miss"] += 1
            else:
                counts["partial_undercover"] += 1  # 1 pred overlaps but IoU<=0.4

        for p in range(len(preds)):
            if p in matched_pred:
                continue
            overlaps = iou[p, :] if len(gt) else np.array([])
            n_gt_overlapping = int((overlaps > 0.1).sum()) if overlaps.size else 0
            if n_gt_overlapping >= 2:
                counts["merged_pred"] += 1
            elif n_gt_overlapping == 0:
                counts["hallucinated"] += 1
            else:
                counts["partial_overcover"] += 1

    total_gt_errors = sum(counts[k] for k in
                          ["near_miss", "fragmented", "pure_miss", "partial_undercover"])
    total_pred_errors = sum(counts[k] for k in
                            ["merged_pred", "hallucinated", "partial_overcover"])
    print("=== GT-side (drives FN) ===")
    for k in ["near_miss", "fragmented", "pure_miss", "partial_undercover"]:
        print(f"  {k:>20}: {counts[k]:4d}  ({100 * counts[k] / max(total_gt_errors, 1):.0f}% of FN)")
    print("=== pred-side (drives FP) ===")
    for k in ["merged_pred", "hallucinated", "partial_overcover"]:
        print(f"  {k:>20}: {counts[k]:4d}  ({100 * counts[k] / max(total_pred_errors, 1):.0f}% of FP)")
    print(f"\n  TP={counts['TP']}  total-FN={total_gt_errors}  total-FP={total_pred_errors}")
    if near_miss_ious:
        print(f"  near-miss best-IoU mean: {np.mean(near_miss_ious):.3f}")


if __name__ == "__main__":
    main()
