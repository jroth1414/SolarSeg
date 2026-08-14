"""Semantic-segmentation dataset for the U-Net + connected-components pipeline.

Each COCO images[] ENTRY (not unique file) is one sample whose target is the
binary UNION of all its filament masks. Multi-annotator duplicate entries are
kept deliberately: the same pixels appear 2-3 times with different annotators'
targets, so a sigmoid net trained with BCE converges toward the pixel-wise mean
across annotators -- i.e. the majority-vote consensus, which scores ~2x better
against any single annotator than imitating one annotator does (PQ 0.654 vs
0.346, measured on the real multi-annotated images).

Training mode returns random square crops biased toward filament-containing
regions (filaments cover <1% of pixels; unbiased crops would be mostly empty).
Eval mode returns the full 2048x2048 image so the fully-convolutional net can
be scored at native resolution.

Augmentation is manual (hflip/vflip/rot90) rather than albumentations -- its
current release needs a C build step on Windows, and d4 symmetries are the
only v1 augmentations anyway (the Sun has no preferred orientation in these
frames beyond limb geometry, which the crops mostly ignore).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset


def load_entries(json_path):
    """Parse the COCO json into a list of per-entry dicts:
    {id, file_name, height, width, anns: [annotation dicts]}."""
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    entries = []
    for im in coco["images"]:
        entries.append(
            {
                "id": im["id"],
                "file_name": im["file_name"],
                "height": im["height"],
                "width": im["width"],
                "anns": anns_by_image.get(im["id"], []),
            }
        )
    return entries


def rasterize_union_mask(entry) -> np.ndarray:
    """Union of all the entry's filament polygons as one HxW uint8 {0,1} mask."""
    h, w = entry["height"], entry["width"]
    polys = [ann["segmentation"][0] for ann in entry["anns"] if ann.get("segmentation")]
    if not polys:
        return np.zeros((h, w), dtype=np.uint8)
    rles = maskUtils.frPyObjects(polys, h, w)
    return maskUtils.decode(maskUtils.merge(rles)).astype(np.uint8)


class SemanticFilamentDataset(Dataset):
    """See module docstring.

    Args:
        json_path / images_dir: the train COCO json + train_images dir.
        image_ids: restrict to these images[].id values (train/val split).
        crop_size: training crop side length. Ignored when train=False.
        train: True -> random biased crops + d4 augmentation;
               False -> full-resolution image, no augmentation.
        fg_bias: probability that a training crop is centered near a randomly
                 chosen filament (with jitter) instead of uniformly placed.
    """

    def __init__(self, json_path, images_dir, image_ids=None, crop_size=1024,
                 train=True, fg_bias=0.7):
        self.images_dir = Path(images_dir)
        self.crop_size = crop_size
        self.train = train
        self.fg_bias = fg_bias

        entries = load_entries(json_path)
        if image_ids is not None:
            wanted = set(image_ids)
            entries = [e for e in entries if e["id"] in wanted]
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _load_image(self, entry) -> np.ndarray:
        with Image.open(self.images_dir / entry["file_name"]) as im:
            return np.array(im.convert("L"), dtype=np.uint8)

    def _crop_coords(self, entry, h, w):
        """Top-left corner of a crop_size crop, biased toward filaments."""
        c = self.crop_size
        if entry["anns"] and random.random() < self.fg_bias:
            ann = random.choice(entry["anns"])
            bx, by, bw, bh = ann["bbox"]
            cx = bx + bw / 2 + random.uniform(-c / 4, c / 4)
            cy = by + bh / 2 + random.uniform(-c / 4, c / 4)
            y0 = int(round(cy - c / 2))
            x0 = int(round(cx - c / 2))
        else:
            y0 = random.randint(0, max(h - c, 0))
            x0 = random.randint(0, max(w - c, 0))
        return int(np.clip(y0, 0, h - c)), int(np.clip(x0, 0, w - c))

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = self._load_image(entry)
        target = rasterize_union_mask(entry)

        if self.train:
            h, w = image.shape
            y0, x0 = self._crop_coords(entry, h, w)
            c = self.crop_size
            image = image[y0:y0 + c, x0:x0 + c]
            target = target[y0:y0 + c, x0:x0 + c]

            # d4-subset augmentation: flips + 90-degree rotations
            if random.random() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                image, target = image[::-1, :], target[::-1, :]
            k = random.randint(0, 3)
            if k:
                image, target = np.rot90(image, k), np.rot90(target, k)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0).unsqueeze(0)
        target_t = torch.from_numpy(np.ascontiguousarray(target)).float().unsqueeze(0)
        return image_t, target_t, entry["id"]


def semantic_collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    ids = [b[2] for b in batch]
    return images, targets, ids


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.coco_dataset import get_grouped_split

    root = Path(__file__).resolve().parents[2] / "data" / "MAGFiLO_1.0_Kaggle_2026"
    jp = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    imdir = root / "train" / "train_images"

    train_ids, val_ids = get_grouped_split(jp, val_fraction=0.15, seed=0)

    failures = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    ds = SemanticFilamentDataset(jp, imdir, image_ids=train_ids[:20], crop_size=1024, train=True)
    check("train dataset length", len(ds) == 20, f"len={len(ds)}")
    img, tgt, eid = ds[0]
    check("train crop image shape/dtype", img.shape == (1, 1024, 1024) and img.dtype == torch.float32,
          f"{tuple(img.shape)} {img.dtype}")
    check("train crop target shape/binary", tgt.shape == (1, 1024, 1024)
          and set(torch.unique(tgt).tolist()) <= {0.0, 1.0})
    check("image range [0,1]", 0.0 <= img.min() and img.max() <= 1.0)

    # fg bias smoke check: most crops should contain some filament pixels
    n_fg = sum((ds[i % len(ds)][1].sum() > 0).item() for i in range(40))
    check("fg-biased crops mostly contain filaments", n_fg >= 25, f"{n_fg}/40 crops had fg")

    ds_eval = SemanticFilamentDataset(jp, imdir, image_ids=val_ids[:4], train=False)
    img, tgt, eid = ds_eval[0]
    check("eval full-res shapes", img.shape == (1, 2048, 2048) and tgt.shape == (1, 2048, 2048))

    # union-mask consistency vs per-annotation area sums (union <= sum, close when no overlap)
    e = ds_eval.entries[0]
    um = rasterize_union_mask(e)
    area_sum = sum(a["area"] for a in e["anns"])
    check("union mask area <= sum of stored areas", um.sum() <= area_sum + 1,
          f"union={int(um.sum())} sum={area_sum:.0f}")

    print("\nSMOKE TEST:", "FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)
