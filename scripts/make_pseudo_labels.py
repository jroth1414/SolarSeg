"""Phase A of the pseudo-labeling pipeline: ensemble-averaged probability maps
for the unlabeled test images.

For each test image, averages the sigmoid probability map over ALL given
checkpoints x 4 flip-TTA variants (10 models x 4 flips by default: the five
1024-crop folds unet_f{0..4} plus the five 1536-crop folds unet1536_f{0..4})
and saves the result as float16 .npy under data/pseudo_labels/probs/<stem>.npy.
These maps are the soft targets consumed by src/data/pseudo_dataset.py /
`scripts/train_unet.py --pseudo-dir`.

The full run is 180 images x 10 models x 4 flips; use --limit (and a single
--checkpoints entry) for quick smoke tests.

Usage:
    python scripts/make_pseudo_labels.py                # full 10-model run
    python scripts/make_pseudo_labels.py --limit 3 --checkpoints checkpoints/unet_f0.pt
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

from src.models.unet import build_unet
from scripts.eval_tta import tta_prob

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
PSEUDO_ROOT = REPO_ROOT / "data" / "pseudo_labels"

DEFAULT_CHECKPOINTS = [
    str(REPO_ROOT / "checkpoints" / f"unet_f{i}.pt") for i in range(5)
] + [
    str(REPO_ROOT / "checkpoints" / f"unet1536_f{i}.pt") for i in range(5)
]

README_TEXT = (
    "Model-generated pseudo-labels for the unlabeled test images "
    "(mean sigmoid probability over an ensemble of fold checkpoints x 4-flip TTA, "
    "see scripts/make_pseudo_labels.py) -- no ground truth involved.\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N test images (smoke testing)")
    ap.add_argument("--out-dir", default=str(PSEUDO_ROOT / "probs"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | {len(args.checkpoints)} checkpoints x 4 flips")

    models = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = build_unet(encoder_name=ckpt["encoder"], encoder_weights=None,
                       classes=ckpt.get("classes", 1)).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  loaded {path} (epoch {ckpt['epoch']}, val_pq {ckpt['val_pq']:.4f})")

    test_paths = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
    if args.limit is not None:
        test_paths = test_paths[: args.limit]
    print(f"{len(test_paths)} test images -> {args.out_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = PSEUDO_ROOT / "README.md"
    if not readme.exists():
        readme.write_text(README_TEXT, encoding="utf-8")

    fg_fracs = []
    global_min, global_max = 1.0, 0.0
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

            np.save(out_dir / f"{path.stem}.npy", prob.astype(np.float16))
            fg_fracs.append(float((prob > 0.5).mean()))
            global_min = min(global_min, float(prob.min()))
            global_max = max(global_max, float(prob.max()))
            if i % 10 == 0 or i == len(test_paths):
                print(f"  {i}/{len(test_paths)} ({time.time() - t0:.0f}s) "
                      f"last: {path.stem} fg@0.5={fg_fracs[-1]:.4%}")

    print(f"\nwrote {len(fg_fracs)} prob maps to {out_dir} in {time.time() - t0:.0f}s")
    print(f"fg fraction @0.5: mean={np.mean(fg_fracs):.4%} "
          f"min={np.min(fg_fracs):.4%} max={np.max(fg_fracs):.4%}")
    print(f"prob value range: [{global_min:.4f}, {global_max:.4f}]")


if __name__ == "__main__":
    main()
