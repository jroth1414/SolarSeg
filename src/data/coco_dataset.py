"""COCO-format dataset pipeline for the MAGFiLO solar-filament instance-segmentation task.

This module is the data-loading contract for the rest of the pipeline (model +
training code depends on the exact class/function names and signatures below,
without necessarily reading this file):

    FilamentDataset(json_path, images_dir, image_ids=None, transforms=None)
    get_grouped_split(json_path, val_fraction=0.15, seed=0) -> (train_ids, val_ids)
    collate_fn(batch)

Key modeling decision (see repo-level task context / CLAUDE.md): category_id
(1=Left, 2=Right, 3=Unidentifiable, 4=Ambiguous) is NOT used for modeling. The
competition submission format and PQ/Dice scoring are single-class (foreground
"filament" vs background) instance segmentation, so every annotation becomes a
label=1 instance here regardless of its source category_id.

Ring-closure choice (documented per task instructions): segmentation polygons
in this dataset are frequently NOT closed rings (42.6% of the 8199 polygons in
the train JSON have a first-vertex-to-last-vertex gap > 1px, up to ~31px; see
notebooks/eda_assets/xref_summary.txt). This module rasterizes polygons via
pycocotools.mask.frPyObjects(...) -> merge(...) -> decode(...), which silently
closes each ring with a straight line from the last vertex back to the first.
This is a deliberate baseline-v1 choice (accept pycocotools' implicit closing),
not an assumption that the polygons were already closed -- it matches how the
EDA notebook validated `area`/`bbox` against rasterized masks (relerr ~ 0), so
it is also the most "reference-consistent" choice available. A future version
could instead explicitly densify/close each ring before rasterizing if thin
barb tips near the gap turn out to matter for the metric.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils

Transform = Callable[[torch.Tensor, dict], Tuple[torch.Tensor, dict]]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _stable_int_id(str_id: str) -> int:
    """Deterministic string -> int64-safe id, stable across processes/runs/dataset
    instances (unlike Python's salted built-in hash()). Used for target['image_id']
    so that different FilamentDataset "views" (e.g. a train subset and a val subset
    built from the same JSON) assign the *same* int id to the *same* COCO image id.
    """
    digest = hashlib.md5(str_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # 32-bit, fits comfortably in int64


def _polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Rasterize a COCO polygon segmentation (list containing one flat [x0,y0,x1,y1,...]
    polygon, per this dataset's schema) to a (height, width) uint8 0/1 mask.

    See module docstring for the ring-closure handling decision.
    """
    rles = maskUtils.frPyObjects(segmentation, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask.astype(np.uint8)


# --------------------------------------------------------------------------- #
# simple augmentations (optional; wired through `transforms`)
# --------------------------------------------------------------------------- #


def hflip(image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
    """Flip image + target left-right. image: [1,H,W], target['boxes']: xyxy."""
    width = image.shape[-1]
    image = image.flip(-1)
    target = dict(target)
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    target["boxes"] = boxes
    target["masks"] = target["masks"].flip(-1)
    return image, target


def vflip(image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
    """Flip image + target top-bottom. image: [1,H,W], target['boxes']: xyxy."""
    height = image.shape[-2]
    image = image.flip(-2)
    target = dict(target)
    boxes = target["boxes"].clone()
    boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
    target["boxes"] = boxes
    target["masks"] = target["masks"].flip(-2)
    return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        if random.random() < self.p:
            image, target = hflip(image, target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        if random.random() < self.p:
            image, target = vflip(image, target)
        return image, target


class Compose:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


class FilamentDataset(torch.utils.data.Dataset):
    """Instance-segmentation dataset over the MAGFiLO COCO-format annotations.

    Parameters
    ----------
    json_path : path to the COCO-style annotation JSON (train file).
    images_dir : directory containing the JPEGs referenced by images[].file_name.
    image_ids : optional subset of images[].id string values to restrict to
        (e.g. output of get_grouped_split, or a small subset for a bounded
        baseline run). If None, uses every image in the JSON.
    transforms : optional callable(image, target) -> (image, target).

    __getitem__ returns (image, target):
        image : torch.FloatTensor [1, H, W], values in [0, 1], single-channel
            (NOT expanded to 3 channels -- that's left to model-building code).
        target : dict with
            'boxes'    : torch.FloatTensor (N, 4), xyxy absolute pixel coords.
            'labels'   : torch.int64 (N,), all ones (single foreground class;
                         background=0 implicit, per torchvision detection convention).
            'masks'    : torch.uint8 (N, H, W), 0/1.
            'image_id' : torch.int64 scalar, stable hash of the COCO string id
                         (see _stable_int_id). Map back to the string id via
                         dataset.int_to_id[int(target['image_id'])], and from
                         there to file_name via dataset.images_by_id[str_id]['file_name'].
            'area'     : torch.FloatTensor (N,), COCO-stored polygon area.
            'iscrowd'  : torch.int64 (N,), all zeros (iscrowd is always 0 in this data).
    """

    def __init__(
        self,
        json_path,
        images_dir,
        image_ids: Optional[Sequence[str]] = None,
        transforms: Optional[Transform] = None,
    ):
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(self.json_path, "r") as f:
            coco = json.load(f)

        self.categories = coco.get("categories", [])
        self.images_by_id: Dict[str, dict] = {img["id"]: img for img in coco["images"]}

        if image_ids is None:
            self.ids: List[str] = [img["id"] for img in coco["images"]]
        else:
            wanted = set(image_ids)
            # preserve the JSON's own ordering for determinism, rather than the
            # (possibly arbitrary) order of the incoming image_ids sequence
            self.ids = [img["id"] for img in coco["images"] if img["id"] in wanted]
            missing = wanted - set(self.ids)
            if missing:
                sample = sorted(missing)[:5]
                raise ValueError(
                    f"{len(missing)} requested image_ids not found in {self.json_path} "
                    f"(e.g. {sample})"
                )

        anns_by_image: Dict[str, list] = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)
        self.anns_by_image = anns_by_image

        # stable string<->int image-id mapping; independent of subsetting/order so
        # different FilamentDataset views of the same JSON agree on the same ids.
        self.id_to_int: Dict[str, int] = {sid: _stable_int_id(sid) for sid in self.images_by_id}
        self.int_to_id: Dict[int, str] = {v: k for k, v in self.id_to_int.items()}
        # parallel list of the (string) COCO image ids, indexable by dataset position,
        # for downstream code that wants "the id this __getitem__(idx) came from".
        self.image_ids: List[str] = list(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        str_id = self.ids[idx]
        img_info = self.images_by_id[str_id]
        height, width = img_info["height"], img_info["width"]

        img_path = self.images_dir / img_info["file_name"]
        with Image.open(img_path) as im:
            im = im.convert("L")
            arr = np.array(im, dtype=np.uint8)
        if arr.shape != (height, width):
            # Defensive: trust the actual file dims over stored JSON dims if they
            # ever disagree (EDA found 0 mismatches across the real dataset, but
            # rasterizing at the wrong size would silently corrupt masks).
            height, width = arr.shape

        image = torch.from_numpy(arr).to(torch.float32).div(255.0).unsqueeze(0)  # [1,H,W]

        anns = self.anns_by_image.get(str_id, [])
        n = len(anns)
        boxes = np.zeros((n, 4), dtype=np.float32)
        masks = np.zeros((n, height, width), dtype=np.uint8)
        areas = np.zeros((n,), dtype=np.float32)
        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            boxes[i] = (x, y, x + w, y + h)
            masks[i] = _polygon_to_mask(ann["segmentation"], height, width)
            areas[i] = ann.get("area", float(masks[i].sum()))

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones((n,), dtype=torch.int64),
            "masks": torch.from_numpy(masks),
            "image_id": torch.tensor(self.id_to_int[str_id], dtype=torch.int64),
            "area": torch.from_numpy(areas),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# --------------------------------------------------------------------------- #
# grouped train/val split
# --------------------------------------------------------------------------- #


def get_grouped_split(
    json_path, val_fraction: float = 0.15, seed: int = 0
) -> Tuple[List[str], List[str]]:
    """Split images[].id values into (train_ids, val_ids), grouped by file_name so
    that every images[] entry sharing the same source file_name (i.e. every
    independent-annotator-batch duplicate of the same base image) lands in the
    SAME split. Splits by a deterministic seed-based shuffle of the unique
    file_name groups (not of individual images[] rows/ids).
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    file_name_to_ids: Dict[str, List[str]] = defaultdict(list)
    for img in coco["images"]:
        file_name_to_ids[img["file_name"]].append(img["id"])

    file_names = sorted(file_name_to_ids.keys())  # deterministic pre-shuffle order
    rng = random.Random(seed)
    rng.shuffle(file_names)

    n_groups = len(file_names)
    if val_fraction <= 0:
        n_val_groups = 0
    elif val_fraction >= 1:
        n_val_groups = n_groups
    else:
        n_val_groups = max(1, round(n_groups * val_fraction))

    val_file_names = set(file_names[:n_val_groups])

    train_ids: List[str] = []
    val_ids: List[str] = []
    for img in coco["images"]:
        if img["file_name"] in val_file_names:
            val_ids.append(img["id"])
        else:
            train_ids.append(img["id"])

    return train_ids, val_ids


# --------------------------------------------------------------------------- #
# collate
# --------------------------------------------------------------------------- #


def collate_fn(batch):
    """Standard torchvision-detection-style collate: tuple of images, tuple of
    targets. Deliberately does NOT stack into a single tensor -- images in a
    batch can have differing annotation counts (and, in general, differing
    sizes), which is exactly what torchvision detection models expect.
    """
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    JSON_PATH = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    IMAGES_DIR = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"

    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print(f"Loading JSON from {JSON_PATH}")
    print(f"Images dir: {IMAGES_DIR}")

    # ---- grouped split ----
    train_ids, val_ids = get_grouped_split(JSON_PATH, val_fraction=0.15, seed=0)
    print(f"split sizes: train={len(train_ids)} val={len(val_ids)} (total={len(train_ids) + len(val_ids)})")

    with open(JSON_PATH, "r") as f:
        coco_raw = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in coco_raw["images"]}
    check("split covers every images[] id exactly once",
          sorted(train_ids + val_ids) == sorted(id_to_fname.keys()))

    train_fnames = {id_to_fname[i] for i in train_ids}
    val_fnames = {id_to_fname[i] for i in val_ids}
    overlap = train_fnames & val_fnames
    check("zero file_name overlap between train/val splits", len(overlap) == 0,
          f"overlap={len(overlap)} file_names" if overlap else f"train_fnames={len(train_fnames)} val_fnames={len(val_fnames)}")

    # determinism check: same seed -> same split
    train_ids2, val_ids2 = get_grouped_split(JSON_PATH, val_fraction=0.15, seed=0)
    check("split is deterministic given the same seed",
          train_ids == train_ids2 and val_ids == val_ids2)

    # ---- dataset on a handful of ids (mix of train + val so both paths get exercised) ----
    sample_ids = train_ids[:8] + val_ids[:4]
    ds = FilamentDataset(JSON_PATH, IMAGES_DIR, image_ids=sample_ids, transforms=None)
    check("dataset length matches requested id subset", len(ds) == len(sample_ids),
          f"len(ds)={len(ds)} expected={len(sample_ids)}")

    total_instances = 0
    max_area_relerr = 0.0
    seen_int_ids = set()

    for idx in range(len(ds)):
        image, target = ds[idx]
        str_id = ds.image_ids[idx]
        img_info = ds.images_by_id[str_id]
        h, w = img_info["height"], img_info["width"]

        check(f"[{idx}] image dtype float32", image.dtype == torch.float32)
        check(f"[{idx}] image shape (1,H,W)", tuple(image.shape) == (1, h, w),
              f"got {tuple(image.shape)} expected (1,{h},{w})")
        check(f"[{idx}] image value range in [0,1]",
              float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
              f"min={float(image.min()):.4f} max={float(image.max()):.4f}")

        n = target["boxes"].shape[0]
        total_instances += n

        check(f"[{idx}] boxes dtype/shape", target["boxes"].dtype == torch.float32 and target["boxes"].shape == (n, 4))
        check(f"[{idx}] labels all ones, int64",
              target["labels"].dtype == torch.int64 and (n == 0 or bool((target["labels"] == 1).all())))
        check(f"[{idx}] masks dtype/shape", target["masks"].dtype == torch.uint8 and tuple(target["masks"].shape) == (n, h, w))
        check(f"[{idx}] masks are 0/1 only", bool(((target["masks"] == 0) | (target["masks"] == 1)).all()))
        check(f"[{idx}] area dtype/shape", target["area"].dtype == torch.float32 and target["area"].shape == (n,))
        check(f"[{idx}] iscrowd all zero, int64", target["iscrowd"].dtype == torch.int64 and bool((target["iscrowd"] == 0).all()))
        check(f"[{idx}] image_id dtype", target["image_id"].dtype == torch.int64)

        int_id = int(target["image_id"])
        check(f"[{idx}] image_id maps back to the correct string id via dataset.int_to_id",
              ds.int_to_id.get(int_id) == str_id)
        seen_int_ids.add(int_id)

        # mask-vs-stored-area sanity check (this is also, by construction, a check
        # on the rasterization pipeline itself: frPyObjects -> merge -> decode)
        for i in range(n):
            mask_px = float(target["masks"][i].sum())
            stored_area = float(target["area"][i])
            denom = max(stored_area, 1.0)
            relerr = abs(mask_px - stored_area) / denom
            max_area_relerr = max(max_area_relerr, relerr)
            check(f"[{idx}] instance {i} mask pixel count ~= stored area", relerr < 0.02,
                  f"mask_px={mask_px} stored_area={stored_area} relerr={relerr:.4f}")

        # boxes should be within image bounds and well-formed (x1>x0, y1>y0)
        if n > 0:
            b = target["boxes"]
            ok_bounds = bool((b[:, 0] >= 0).all() and (b[:, 1] >= 0).all() and (b[:, 2] <= w).all() and (b[:, 3] <= h).all())
            ok_order = bool((b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all())
            check(f"[{idx}] boxes within image bounds", ok_bounds)
            check(f"[{idx}] boxes well-formed (x1>x0, y1>y0)", ok_order)

    check("image_id ints are unique across the sampled subset", len(seen_int_ids) == len(sample_ids),
          f"{len(seen_int_ids)} unique vs {len(sample_ids)} images")
    print(f"total instances seen: {total_instances}, max mask-vs-area relerr: {max_area_relerr:.5f}")

    # ---- collate_fn ----
    batch = [ds[i] for i in range(min(4, len(ds)))]
    images_batch, targets_batch = collate_fn(batch)
    check("collate_fn returns tuple of images", isinstance(images_batch, tuple) and len(images_batch) == len(batch))
    check("collate_fn returns tuple of targets", isinstance(targets_batch, tuple) and len(targets_batch) == len(batch))
    check("collate_fn preserves per-image dict targets", all(isinstance(t, dict) for t in targets_batch))

    # ---- transforms plumbing (hflip / vflip correctness) ----
    ds_t = FilamentDataset(JSON_PATH, IMAGES_DIR, image_ids=sample_ids[:1], transforms=None)
    image0, target0 = ds_t[0]
    image_h, target_h = hflip(image0, target0)
    image_v, target_v = vflip(image0, target0)
    check("hflip preserves image shape", image_h.shape == image0.shape)
    check("hflip preserves mask pixel counts", bool((target_h["masks"].sum(dim=(1, 2)) == target0["masks"].sum(dim=(1, 2))).all()))
    check("hflip is self-inverse on the image", bool(torch.equal(hflip(image_h, target_h)[0], image0)))
    check("vflip preserves image shape", image_v.shape == image0.shape)
    check("vflip preserves mask pixel counts", bool((target_v["masks"].sum(dim=(1, 2)) == target0["masks"].sum(dim=(1, 2))).all()))
    n0 = target0["boxes"].shape[0]
    if n0 > 0:
        w0 = image0.shape[-1]
        check("hflip box x-coords transform correctly",
              bool(torch.allclose(target_h["boxes"][:, [0, 2]], w0 - target0["boxes"][:, [2, 0]])))

    print()
    if failures:
        print(f"SMOKE TEST: FAIL ({len(failures)} check(s) failed)")
        for name in failures:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST: PASS (all checks passed; {len(sample_ids)} images / {total_instances} instances exercised)")
