"""
Local evaluation metrics for the Solar Filament Segmentation Challenge 2026.

Implements the two metrics named on the competition page:
  - Panoptic Quality (PQ), per Kirillov et al. 2019, single "thing" class
    (filament vs background -- category_id in the source COCO json is not
    part of scoring and is deliberately ignored here; see project notes).
  - Mean Dice score.

Both metrics are computed PER IMAGE and then averaged across images, which
matches how the competition describes its leaderboard aggregation.

--------------------------------------------------------------------------
Panoptic Quality -- exact algorithm implemented (mirrors the competition's
cited formulation):
  1. A (pred, gt) pair is a valid match iff IoU(pred, gt) > 0.5 (strict).
     Because the threshold is > 0.5, each gt can match at most one pred and
     vice versa (the standard panoptic-quality uniqueness property), so in
     the well-behaved case (non-overlapping instances within each of the
     gt set and the pred set) matching reduces to "does some pred have
     IoU > 0.5 with this gt". We still implement matching as an explicit
     greedy-by-descending-IoU assignment rather than assuming the property
     holds, so the function stays correct even if it is ever handed
     pathological (self-overlapping) prediction sets.
  2. TP = matched pairs, FN = unmatched gt, FP = unmatched pred.
  3. SQ = mean IoU over TP pairs (0.0 if TP == 0).
  4. RQ = TP / (TP + 0.5*FP + 0.5*FN) (0.0 if the denominator is 0, i.e.
     TP == FP == FN == 0, which only happens when both gt and pred are
     empty for that image).
  5. PQ = SQ * RQ.

IoU computation uses pycocotools.mask.iou on RLE-encoded masks to get the
whole NxM IoU matrix in one (compiled) call, rather than a naive Python
double loop -- some images in this dataset have up to ~26 instances, and
mask IoU on 2048x2048 masks is not cheap to redo per-pair in Python.

--------------------------------------------------------------------------
Dice -- implementation choice (documented per task instructions):
We implement per-image UNION-MASK binary Dice directly with numpy/bitwise
ops rather than routing through torchmetrics.segmentation.DiceScore:
  - All predicted instance masks for an image are OR-ed into one binary
    HxW mask; all ground-truth instance masks are OR-ed into another.
    Dice = 2*|A ∩ B| / (|A| + |B|) between those two binary masks.
  - This is the simple, well-defined, matching-free reading of "mean Dice
    score" for an instance-segmentation task with a single foreground
    class, and it complements PQ (which already covers the instance-aware
    view). It is NOT guaranteed to bit-exactly match the hidden
    competition scorer's internal convention, since that convention is not
    fully documented publicly.
  - We use a direct numpy implementation instead of
    torchmetrics.segmentation.DiceScore because that metric's
    input_format/num_classes configuration is designed around per-pixel
    class-probability or one-hot tensors accumulated over batches; for a
    single pair of binary HxW masks the direct set-overlap formula is
    equivalent, has no ambiguity in how to configure it for the binary
    case, and avoids taking a torch/torchmetrics dependency in a pure
    numpy-mask evaluation path. (torchmetrics is still available in the
    project if a training loop wants a differentiable/batched variant.)
  - Special case: if both the gt union mask and the pred union mask are
    empty (no instances on either side), Dice is defined as 1.0 (a
    trivially perfect match on an empty case), rather than 0/0 -> NaN.

Author's note on mask rasterization: this module operates purely on
already-rasterized boolean/uint8 HxW masks. It intentionally does NOT
rasterize COCO polygons itself -- see src/eval/submission.py and the
dataset-loading code for that step (and its documented choice about
implicit polygon-ring closing via pycocotools).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from pycocotools import mask as maskUtils


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _to_uint8(mask: np.ndarray) -> np.ndarray:
    """Coerce a bool/uint8/int HxW mask to a strictly-0/1 uint8 array."""
    return (np.asarray(mask) > 0).astype(np.uint8)


def _encode_rle(mask: np.ndarray):
    """RLE-encode one HxW binary mask via pycocotools (Fortran order required)."""
    return maskUtils.encode(np.asfortranarray(_to_uint8(mask)))


def _iou_matrix(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> np.ndarray:
    """
    Full pairwise IoU matrix, shape (len(pred_masks), len(gt_masks)), computed
    via pycocotools.mask.iou on RLE-encoded masks (compiled C, not a Python
    double loop).
    """
    pred_rles = [_encode_rle(m) for m in pred_masks]
    gt_rles = [_encode_rle(m) for m in gt_masks]
    iscrowd = [0] * len(gt_rles)
    iou = maskUtils.iou(pred_rles, gt_rles, iscrowd)
    return np.asarray(iou, dtype=np.float64).reshape(len(pred_masks), len(gt_masks))


# --------------------------------------------------------------------------
# Panoptic Quality
# --------------------------------------------------------------------------


def compute_panoptic_quality(
    gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]
) -> Dict[str, float]:
    """
    Panoptic Quality for a SINGLE image, single foreground class.

    Args:
        gt_masks: list of HxW boolean/uint8 ground-truth instance masks.
        pred_masks: list of HxW boolean/uint8 predicted instance masks.

    Returns:
        dict with keys 'pq', 'sq', 'rq' (floats) and 'tp', 'fp', 'fn' (ints).
    """
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    if n_gt == 0 and n_pred == 0:
        # Nothing to match, nothing predicted, nothing missed. Denominator
        # of RQ is 0 -> RQ defined as 0 per the spec, so PQ = 0 as well.
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 0}

    if n_gt == 0:
        # Every prediction is a false positive; no TP possible.
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": n_pred, "fn": 0}

    if n_pred == 0:
        # Every gt instance is missed.
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": n_gt}

    iou = _iou_matrix(pred_masks, gt_masks)  # shape (n_pred, n_gt)

    # Greedy one-to-one matching over all candidate pairs with IoU > 0.5,
    # processed in descending IoU order. Under the standard panoptic-quality
    # property (IoU > 0.5 threshold => matches are automatically unique)
    # this is equivalent to "does each gt have a >0.5-IoU pred", but doing
    # it as explicit greedy assignment keeps the function correct even if
    # that property doesn't hold for some malformed input.
    pred_idx, gt_idx = np.where(iou > 0.5)
    if pred_idx.size > 0:
        order = np.argsort(-iou[pred_idx, gt_idx], kind="stable")
        pred_idx = pred_idx[order]
        gt_idx = gt_idx[order]

    used_pred = set()
    used_gt = set()
    tp_ious: List[float] = []
    for p, g in zip(pred_idx.tolist(), gt_idx.tolist()):
        if p in used_pred or g in used_gt:
            continue
        used_pred.add(p)
        used_gt.add(g)
        tp_ious.append(float(iou[p, g]))

    tp = len(tp_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    sq = float(np.mean(tp_ious)) if tp > 0 else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = (tp / denom) if denom > 0 else 0.0
    pq = sq * rq

    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def compute_panoptic_quality_dataset(
    gt_masks_per_image: List[List[np.ndarray]],
    pred_masks_per_image: List[List[np.ndarray]],
) -> Dict[str, object]:
    """
    Dataset-level Panoptic Quality: calls compute_panoptic_quality per image
    and averages pq/sq/rq across images (simple mean, one image = one unit).

    Args:
        gt_masks_per_image: list (length = num images) of lists of HxW gt masks.
        pred_masks_per_image: list (length = num images) of lists of HxW pred masks.
            Must be the same length and same image order as gt_masks_per_image.

    Returns:
        dict with keys 'pq', 'sq', 'rq' (mean across images, floats) and
        'per_image' (list of the raw per-image dicts from compute_panoptic_quality,
        same order as the input).
    """
    if len(gt_masks_per_image) != len(pred_masks_per_image):
        raise ValueError(
            f"gt_masks_per_image and pred_masks_per_image must have the same "
            f"length (got {len(gt_masks_per_image)} vs {len(pred_masks_per_image)})"
        )

    per_image = [
        compute_panoptic_quality(gt_masks, pred_masks)
        for gt_masks, pred_masks in zip(gt_masks_per_image, pred_masks_per_image)
    ]

    n = len(per_image)
    pq_mean = float(np.mean([r["pq"] for r in per_image])) if n > 0 else 0.0
    sq_mean = float(np.mean([r["sq"] for r in per_image])) if n > 0 else 0.0
    rq_mean = float(np.mean([r["rq"] for r in per_image])) if n > 0 else 0.0

    return {"pq": pq_mean, "sq": sq_mean, "rq": rq_mean, "per_image": per_image}


# --------------------------------------------------------------------------
# Dice
# --------------------------------------------------------------------------


def _union_mask(masks: List[np.ndarray], like: np.ndarray = None) -> np.ndarray:
    """OR all masks in the list into one boolean HxW mask."""
    if len(masks) == 0:
        if like is None:
            raise ValueError("Cannot infer mask shape: no masks given and no 'like' reference.")
        return np.zeros_like(like, dtype=bool)
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= np.asarray(m).astype(bool)
    return out


def compute_dice_per_image(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]) -> float:
    """
    Per-image union-mask binary Dice (see module docstring for the
    implementation-choice rationale).

    Unions all gt_masks into one binary HxW mask A, all pred_masks into one
    binary HxW mask B, and returns Dice(A, B) = 2*|A∩B| / (|A|+|B|).
    If both A and B are empty (no gt instances and no pred instances), returns
    1.0 by definition (perfect match on an empty case).
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0

    gt_union = _union_mask(gt_masks, like=pred_masks[0] if pred_masks else None)
    pred_union = _union_mask(pred_masks, like=gt_masks[0] if gt_masks else None)

    inter = np.logical_and(gt_union, pred_union).sum(dtype=np.int64)
    total = gt_union.sum(dtype=np.int64) + pred_union.sum(dtype=np.int64)

    if total == 0:
        return 1.0
    return float(2.0 * inter / total)


def compute_dice_dataset(
    gt_masks_per_image: List[List[np.ndarray]],
    pred_masks_per_image: List[List[np.ndarray]],
) -> float:
    """Mean of compute_dice_per_image across all images (simple mean, one image = one unit)."""
    if len(gt_masks_per_image) != len(pred_masks_per_image):
        raise ValueError(
            f"gt_masks_per_image and pred_masks_per_image must have the same "
            f"length (got {len(gt_masks_per_image)} vs {len(pred_masks_per_image)})"
        )
    if len(gt_masks_per_image) == 0:
        return 0.0
    dices = [
        compute_dice_per_image(gt_masks, pred_masks)
        for gt_masks, pred_masks in zip(gt_masks_per_image, pred_masks_per_image)
    ]
    return float(np.mean(dices))


# --------------------------------------------------------------------------
# Smoke test with hand-verifiable toy cases
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    CANVAS = 30

    def _rect_mask(r0, r1, c0, c1, size=CANVAS):
        """HxW uint8 mask with a filled rectangle over rows[r0,r1) cols[c0,c1)."""
        m = np.zeros((size, size), dtype=np.uint8)
        m[r0:r1, c0:c1] = 1
        return m

    all_pass = True

    def _check(name, got, expected, tol=1e-9):
        if isinstance(expected, float):
            ok = abs(got - expected) <= tol
        else:
            ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: got={got!r} expected={expected!r}")
        return ok

    # ---- Case 1: perfect match, 2 non-overlapping squares -> PQ=1, Dice=1
    print("Case 1: perfect match (2 non-overlapping 5x5 squares)")
    gt1 = _rect_mask(0, 5, 0, 5)
    gt2 = _rect_mask(10, 15, 10, 15)
    pred1 = gt1.copy()
    pred2 = gt2.copy()
    res = compute_panoptic_quality([gt1, gt2], [pred1, pred2])
    ok = True
    ok &= _check("tp", res["tp"], 2)
    ok &= _check("fp", res["fp"], 0)
    ok &= _check("fn", res["fn"], 0)
    ok &= _check("sq", res["sq"], 1.0)
    ok &= _check("rq", res["rq"], 1.0)
    ok &= _check("pq", res["pq"], 1.0)
    dice1 = compute_dice_per_image([gt1, gt2], [pred1, pred2])
    ok &= _check("dice", dice1, 1.0)
    all_pass &= ok

    # ---- Case 2: same as case 1, plus one extra FP prediction elsewhere.
    #      Hand math: TP=2 (unchanged), FP=1, FN=0.
    #      SQ = mean IoU over TP = (1.0+1.0)/2 = 1.0
    #      RQ = 2 / (2 + 0.5*1 + 0.5*0) = 2/2.5 = 0.8
    #      PQ = 1.0 * 0.8 = 0.8
    #      Dice: gt union area=25+25=50; pred union area=25+25+25=75;
    #            intersection=50 (extra FP doesn't overlap any gt) ->
    #            Dice = 2*50/(50+75) = 100/125 = 0.8
    print("Case 2: case 1 + one extra false-positive prediction")
    pred_fp = _rect_mask(20, 25, 20, 25)
    res2 = compute_panoptic_quality([gt1, gt2], [pred1, pred2, pred_fp])
    ok = True
    ok &= _check("tp", res2["tp"], 2)
    ok &= _check("fp", res2["fp"], 1)
    ok &= _check("fn", res2["fn"], 0)
    ok &= _check("sq", res2["sq"], 1.0)
    ok &= _check("rq", res2["rq"], 0.8)
    ok &= _check("pq", res2["pq"], 0.8)
    dice2 = compute_dice_per_image([gt1, gt2], [pred1, pred2, pred_fp])
    ok &= _check("dice", dice2, 0.8)
    all_pass &= ok

    # ---- Case 3: IoU matching-boundary checks (single gt vs single pred).
    # 3a: IoU exactly 0.5 -> gt rows[0,4) cols[0,3) (4x3=12), pred rows[0,4)
    #     cols[1,4) (4x3=12). overlap = rows(4) * cols_overlap([1,3)=2) = 8.
    #     union = 12+12-8 = 16. IoU = 8/16 = 0.5 exactly.
    #     Since matching requires IoU > 0.5 STRICTLY, this must NOT match.
    print("Case 3a: IoU exactly 0.5 (boundary) -> must NOT match")
    gt_a = _rect_mask(0, 4, 0, 3)
    pred_a = _rect_mask(0, 4, 1, 4)
    # sanity-check the hand-derived IoU independently via raw pixel counts
    inter_a = np.logical_and(gt_a, pred_a).sum()
    union_a = np.logical_or(gt_a, pred_a).sum()
    hand_iou_a = inter_a / union_a
    res3a = compute_panoptic_quality([gt_a], [pred_a])
    ok = True
    ok &= _check("hand-computed IoU", float(hand_iou_a), 0.5)
    ok &= _check("tp", res3a["tp"], 0)
    ok &= _check("fp", res3a["fp"], 1)
    ok &= _check("fn", res3a["fn"], 1)
    ok &= _check("pq", res3a["pq"], 0.0)
    all_pass &= ok

    # 3b: IoU = 0.6 (clearly above 0.5) -> must match.
    #     gt rows[0,3) cols[0,4) (3x4=12), pred rows[0,3) cols[1,5) (3x4=12).
    #     overlap = rows(3) * cols_overlap([1,4)=3) = 9. union=12+12-9=15.
    #     IoU = 9/15 = 0.6
    print("Case 3b: IoU = 0.6 (clearly above 0.5) -> must match")
    gt_b = _rect_mask(0, 3, 0, 4)
    pred_b = _rect_mask(0, 3, 1, 5)
    inter_b = np.logical_and(gt_b, pred_b).sum()
    union_b = np.logical_or(gt_b, pred_b).sum()
    hand_iou_b = inter_b / union_b
    res3b = compute_panoptic_quality([gt_b], [pred_b])
    ok = True
    ok &= _check("hand-computed IoU", float(hand_iou_b), 0.6)
    ok &= _check("tp", res3b["tp"], 1)
    ok &= _check("fp", res3b["fp"], 0)
    ok &= _check("fn", res3b["fn"], 0)
    ok &= _check("sq", res3b["sq"], 0.6)
    ok &= _check("rq", res3b["rq"], 1.0)
    ok &= _check("pq", res3b["pq"], 0.6)
    all_pass &= ok

    # 3c: IoU = 1/3 (clearly below 0.5) -> must NOT match.
    #     gt rows[0,3) cols[0,4) (3x4=12), pred rows[0,3) cols[2,6) (3x4=12).
    #     overlap = rows(3) * cols_overlap([2,4)=2) = 6. union=12+12-6=18.
    #     IoU = 6/18 = 0.3333...
    print("Case 3c: IoU = 1/3 (clearly below 0.5) -> must NOT match")
    gt_c = _rect_mask(0, 3, 0, 4)
    pred_c = _rect_mask(0, 3, 2, 6)
    inter_c = np.logical_and(gt_c, pred_c).sum()
    union_c = np.logical_or(gt_c, pred_c).sum()
    hand_iou_c = inter_c / union_c
    res3c = compute_panoptic_quality([gt_c], [pred_c])
    ok = True
    ok &= _check("hand-computed IoU", float(hand_iou_c), 1.0 / 3.0)
    ok &= _check("tp", res3c["tp"], 0)
    ok &= _check("fp", res3c["fp"], 1)
    ok &= _check("fn", res3c["fn"], 1)
    ok &= _check("pq", res3c["pq"], 0.0)
    all_pass &= ok

    # ---- Case 4: completely disjoint pred vs gt -> PQ=0, Dice=0
    print("Case 4: completely disjoint pred vs gt (2 gt, 2 pred, zero overlap anywhere)")
    gt4a = _rect_mask(0, 5, 0, 5)
    gt4b = _rect_mask(10, 15, 10, 15)
    pred4a = _rect_mask(20, 25, 20, 25)
    pred4b = _rect_mask(0, 5, 20, 25)
    res4 = compute_panoptic_quality([gt4a, gt4b], [pred4a, pred4b])
    ok = True
    ok &= _check("tp", res4["tp"], 0)
    ok &= _check("fp", res4["fp"], 2)
    ok &= _check("fn", res4["fn"], 2)
    ok &= _check("sq", res4["sq"], 0.0)
    ok &= _check("rq", res4["rq"], 0.0)
    ok &= _check("pq", res4["pq"], 0.0)
    dice4 = compute_dice_per_image([gt4a, gt4b], [pred4a, pred4b])
    ok &= _check("dice", dice4, 0.0)
    all_pass &= ok

    # ---- Case 5: both gt and pred empty for an image -> Dice=1.0 by definition,
    #      PQ=0.0 per the literal "0 if denominator is 0" spec (documented).
    print("Case 5: both gt and pred empty for an image")
    res5 = compute_panoptic_quality([], [])
    ok = True
    ok &= _check("tp", res5["tp"], 0)
    ok &= _check("fp", res5["fp"], 0)
    ok &= _check("fn", res5["fn"], 0)
    ok &= _check("pq", res5["pq"], 0.0)
    dice5 = compute_dice_per_image([], [])
    ok &= _check("dice", dice5, 1.0)
    all_pass &= ok

    # ---- Dataset-level aggregation sanity check: average of cases 1 and 4's PQ
    print("Case 6: dataset-level aggregation (mean of case-1 image and case-4 image)")
    ds = compute_panoptic_quality_dataset(
        [[gt1, gt2], [gt4a, gt4b]], [[pred1, pred2], [pred4a, pred4b]]
    )
    expected_pq_mean = (1.0 + 0.0) / 2.0
    ok = True
    ok &= _check("dataset pq (mean)", ds["pq"], expected_pq_mean)
    ok &= _check("len(per_image)", len(ds["per_image"]), 2)
    dice_ds = compute_dice_dataset([[gt1, gt2], [gt4a, gt4b]], [[pred1, pred2], [pred4a, pred4b]])
    ok &= _check("dataset dice (mean)", dice_ds, (1.0 + 0.0) / 2.0)
    all_pass &= ok

    print()
    if all_pass:
        print("ALL CASES: PASS")
    else:
        print("ALL CASES: FAIL (see [FAIL] lines above)")
        sys.exit(1)
