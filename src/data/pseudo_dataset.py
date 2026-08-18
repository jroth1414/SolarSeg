"""Pseudo-label dataset for semi-supervised U-Net training.

One sample per unlabeled test image whose target is the ensemble-averaged
sigmoid probability map produced by scripts/make_pseudo_labels.py (float16
.npy, same 2048x2048 grid as the image). With soft=True the float probabilities
are kept as-is and act as soft BCE targets -- the same "converge to the
pixel-wise consensus" mechanism the duplicate-annotator training targets
exploit (see src/data/semantic_dataset.py); soft=False binarizes at 0.5.

Training behavior mirrors SemanticFilamentDataset train mode exactly: random
fg-biased square crops (here the bias picks a random pixel with prob > 0.5 as
the crop center, since there are no annotation bboxes) plus the same
hflip/vflip/rot90 d4-subset augmentation. Samples are (image [1,c,c] float,
target [1,c,c] float, id_str) tuples, collate-compatible with
semantic_collate; ids are prefixed "pseudo-" so pseudo samples remain
identifiable downstream.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PseudoLabelDataset(Dataset):
    """See module docstring.

    Args:
        images_dir: directory with the unlabeled <stem>.jpeg images.
        probs_dir: directory with matching <stem>.npy probability maps.
        crop_size: training crop side length.
        fg_bias: probability that a crop is centered on a random pixel with
                 prob > 0.5 (when any exist) instead of uniformly placed.
        soft: True keeps float probabilities as soft targets;
              False binarizes the target at 0.5.
    """

    def __init__(self, images_dir, probs_dir, crop_size=1024, fg_bias=0.7, soft=True):
        self.images_dir = Path(images_dir)
        self.probs_dir = Path(probs_dir)
        self.crop_size = crop_size
        self.fg_bias = fg_bias
        self.soft = soft

        self.stems = sorted(
            p.stem for p in self.probs_dir.glob("*.npy")
            if (self.images_dir / f"{p.stem}.jpeg").exists()
        )
        if not self.stems:
            raise FileNotFoundError(
                f"no <stem>.npy prob maps in {self.probs_dir} with matching "
                f"<stem>.jpeg images in {self.images_dir} "
                f"(run scripts/make_pseudo_labels.py first)"
            )

    def __len__(self):
        return len(self.stems)

    def _crop_coords(self, prob, h, w):
        """Top-left corner of a crop_size crop, biased toward prob>0.5 pixels."""
        c = self.crop_size
        if random.random() < self.fg_bias:
            fg = np.argwhere(prob > 0.5)
            if len(fg):
                cy, cx = fg[random.randrange(len(fg))]
                y0 = int(round(cy - c / 2))
                x0 = int(round(cx - c / 2))
                return int(np.clip(y0, 0, h - c)), int(np.clip(x0, 0, w - c))
        return random.randint(0, max(h - c, 0)), random.randint(0, max(w - c, 0))

    def __getitem__(self, idx):
        stem = self.stems[idx]
        with Image.open(self.images_dir / f"{stem}.jpeg") as im:
            image = np.array(im.convert("L"), dtype=np.uint8)
        prob = np.load(self.probs_dir / f"{stem}.npy").astype(np.float32)
        target = prob if self.soft else (prob > 0.5).astype(np.float32)

        h, w = image.shape
        y0, x0 = self._crop_coords(prob, h, w)
        c = self.crop_size
        image = image[y0:y0 + c, x0:x0 + c]
        target = target[y0:y0 + c, x0:x0 + c]

        # d4-subset augmentation, identical to SemanticFilamentDataset train mode
        if random.random() < 0.5:
            image, target = image[:, ::-1], target[:, ::-1]
        if random.random() < 0.5:
            image, target = image[::-1, :], target[::-1, :]
        k = random.randint(0, 3)
        if k:
            image, target = np.rot90(image, k), np.rot90(target, k)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0).unsqueeze(0)
        target_t = torch.from_numpy(np.ascontiguousarray(target)).float().unsqueeze(0)
        return image_t, target_t, f"pseudo-{stem}"


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.semantic_dataset import semantic_collate

    root = Path(__file__).resolve().parents[2]
    images_dir = root / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
    probs_dir = root / "data" / "pseudo_labels" / "probs"

    failures = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    ds = PseudoLabelDataset(images_dir, probs_dir, crop_size=1024, fg_bias=0.7, soft=True)
    print(f"{len(ds)} pseudo-label samples from {probs_dir}")

    img, tgt, sid = ds[0]
    check("image shape/dtype", img.shape == (1, 1024, 1024) and img.dtype == torch.float32,
          f"{tuple(img.shape)} {img.dtype}")
    check("target shape/dtype", tgt.shape == (1, 1024, 1024) and tgt.dtype == torch.float32,
          f"{tuple(tgt.shape)} {tgt.dtype}")
    check("image range [0,1]", 0.0 <= img.min() and img.max() <= 1.0,
          f"[{img.min():.3f}, {img.max():.3f}]")
    check("target range [0,1]", 0.0 <= tgt.min() and tgt.max() <= 1.0,
          f"[{tgt.min():.3f}, {tgt.max():.3f}]")
    check("id has pseudo- prefix", sid.startswith("pseudo-"), sid)

    # soft targets should contain non-binary values somewhere in the dataset
    any_soft = any(
        bool(((ds[i][1] > 0.0) & (ds[i][1] < 1.0)).any()) for i in range(len(ds))
    )
    check("soft targets contain fractional probs", any_soft)

    # fg bias: most crops should contain fg (prob > 0.5) pixels
    n_fg = sum(bool((ds[i % len(ds)][1] > 0.5).any()) for i in range(40))
    check("fg-biased crops mostly contain fg", n_fg >= 25, f"{n_fg}/40 crops had fg")

    ds_hard = PseudoLabelDataset(images_dir, probs_dir, crop_size=1024, soft=False)
    _, tgt_h, _ = ds_hard[0]
    check("soft=False binarizes target", set(torch.unique(tgt_h).tolist()) <= {0.0, 1.0})

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                        collate_fn=semantic_collate)
    images, targets, ids = next(iter(loader))
    check("semantic_collate batch shapes",
          images.shape == (2, 1, 1024, 1024) and targets.shape == (2, 1, 1024, 1024)
          and len(ids) == 2, f"{tuple(images.shape)} {tuple(targets.shape)} ids={ids}")

    print("\nSMOKE TEST:", "FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)
