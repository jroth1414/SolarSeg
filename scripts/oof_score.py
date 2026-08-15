"""Out-of-fold PQ/Dice: stitch every fold's cached TTA val probability maps
into one full-coverage evaluation over all 1154 train entries.

Each entry is scored using the probability map from the ONE fold that held it
out, so no entry is ever scored by a model that trained on it. This is the
robust single-model estimate of the recipe (the test-time fold-ensemble is
expected to score at or above it).

Usage:
    python scripts/oof_score.py --threshold 0.55 --min-area 400 --closing
"""

from __future__ import annotations

import argparse
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-pattern", default="unet_f{fold}_tta_valprobs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=400)
    ap.add_argument("--closing", action="store_true")
    args = ap.parse_args()

    entries_by_id = {e["id"]: e for e in load_entries(JSON_PATH)}

    prob_files = {}
    for fold in range(args.folds):
        d = REPO_ROOT / "checkpoints" / args.probs_pattern.format(fold=fold)
        for f in d.glob("*.npy"):
            assert f.stem not in prob_files, f"entry {f.stem} in two folds!"
            prob_files[f.stem] = f
    missing = set(entries_by_id) - set(prob_files)
    print(f"OOF coverage: {len(prob_files)}/{len(entries_by_id)} entries"
          + (f" -- MISSING {len(missing)}" if missing else " (complete)"))

    pqs = []
    for entry_id, f in sorted(prob_files.items()):
        prob = np.load(f).astype(np.float32)
        gt = entry_instance_masks(entries_by_id[entry_id])
        preds = probs_to_instances(prob, threshold=args.threshold, min_area=args.min_area,
                                   closing=args.closing, non_empty_guard=True)
        pqs.append(compute_panoptic_quality(gt, preds))

    pq = float(np.mean([x["pq"] for x in pqs]))
    sq = float(np.mean([x["sq"] for x in pqs]))
    rq = float(np.mean([x["rq"] for x in pqs]))
    tp = sum(x["tp"] for x in pqs); fp = sum(x["fp"] for x in pqs); fn = sum(x["fn"] for x in pqs)
    print(f"\nOOF ({len(pqs)} entries): PQ={pq:.4f} SQ={sq:.4f} RQ={rq:.4f} "
          f"(TP={tp} FP={fp} FN={fn})")


if __name__ == "__main__":
    main()
