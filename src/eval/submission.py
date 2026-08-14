"""
Submission CSV encoder for the Solar Filament Segmentation Challenge 2026.

Submission format (validated end-to-end against real data, see project notes):
    Single CSV, one row per predicted filament instance:
        filament_id,segmentation_rle
        20150125172714Mh_1,<rle counts string>
        20150125172714Mh_2,<rle counts string>

    - filament_id = f"{image_id}_{n}" where image_id is the base file_name
      WITHOUT extension, and n just needs to make rows unique per image.
    - segmentation_rle = RLE COUNTS ONLY (no size field -- size is fixed
      2048x2048 for every image), produced via
      pycocotools.mask.encode(np.asfortranarray(binary_mask))['counts'].decode('ascii').
    - An image with 0 predicted filaments contributes 0 rows (no placeholder row).

Empirical note on RLE character content (checked against 500 real rasterized
train-set masks before deciding on CSV quoting, see mask_to_rle_string):
pycocotools' compressed-RLE 'counts' strings only ever used ASCII characters
in the range ord 48-111 in that sample (no commas, no quote characters).
We still write the CSV via Python's csv module with its default QUOTE_MINIMAL
policy rather than hand-formatting rows, so correctness does not depend on
that empirical observation continuing to hold -- if a comma or quote ever
did show up in a counts string, csv.writer would quote/escape it correctly
per RFC 4180 automatically.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np
from pycocotools import mask as maskUtils


def mask_to_rle_string(binary_mask: np.ndarray) -> str:
    """
    Encode a single HxW binary mask (bool or uint8, foreground=truthy) to the
    submission's RLE counts-only string (no 'size' field -- image size is
    fixed 2048x2048 for every image in this competition and is not encoded
    here; callers/downstream readers must already know it).

    Reference implementation (verified to round-trip pixel-identically
    against pycocotools.mask.decode in an earlier EDA pass):
        pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('ascii')
    """
    mask_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask_u8))
    return rle["counts"].decode("ascii")


def rle_string_to_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """
    Inverse of mask_to_rle_string: given the counts-only RLE string plus the
    (fixed, externally-known) image height/width, reconstruct the binary
    mask. Returns an HxW uint8 array with values in {0, 1}.
    """
    rle = {"size": [height, width], "counts": rle_str.encode("ascii")}
    mask = maskUtils.decode(rle)
    return mask.astype(np.uint8)


def build_submission_csv(predictions: Dict[str, List[np.ndarray]], out_path) -> None:
    """
    Write the competition submission CSV.

    Args:
        predictions: dict mapping image_id (base file_name WITHOUT extension,
            e.g. '20150125172714Mh') -> list of predicted binary HxW masks for
            that image. An image with an empty list contributes 0 rows.
        out_path: path (str or os.PathLike) to write the CSV to. Parent
            directory is created if it doesn't already exist.

    Writes header row 'filament_id,segmentation_rle' followed by one row per
    predicted instance, filament_id = f'{image_id}_{n}' with n starting at 1
    per image. No index column. Quoting is left to csv.writer's default
    QUOTE_MINIMAL policy (see module docstring for why that's safe/correct
    here).
    """
    out_path = os.fspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(out_path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)  # default QUOTE_MINIMAL, comma delimiter
        writer.writerow(["filament_id", "segmentation_rle"])
        for image_id, masks in predictions.items():
            for n, mask in enumerate(masks, start=1):
                filament_id = f"{image_id}_{n}"
                rle_str = mask_to_rle_string(mask)
                writer.writerow([filament_id, rle_str])


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import tempfile

    import pandas as pd

    all_pass = True

    def _check(name, ok):
        global all_pass
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_pass &= ok

    # ---- Part 1: round-trip 10 REAL masks rasterized from the actual train COCO json
    print("Part 1: round-trip 10 real annotations from the train COCO json")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    coco_path = os.path.join(
        repo_root,
        "data",
        "MAGFiLO_1.0_Kaggle_2026",
        "train",
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    print(f"  loading {coco_path}")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    sample_anns = coco["annotations"][:10]
    print(f"  sampled {len(sample_anns)} annotations")

    n_identical = 0
    for i, ann in enumerate(sample_anns):
        img = images_by_id[ann["image_id"]]
        h, w = img["height"], img["width"]
        seg = ann["segmentation"]  # list containing exactly one flat polygon
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles) if len(rles) > 1 else rles[0]
        original_mask = maskUtils.decode(rle).astype(np.uint8)  # HxW, {0,1}

        rle_str = mask_to_rle_string(original_mask)
        recovered_mask = rle_string_to_mask(rle_str, h, w)

        identical = np.array_equal(original_mask, recovered_mask)
        n_pixels_orig = int(original_mask.sum())
        n_pixels_rec = int(recovered_mask.sum())
        print(
            f"    ann[{i}] id={ann['id']} shape={original_mask.shape} "
            f"pixels(orig)={n_pixels_orig} pixels(recovered)={n_pixels_rec} "
            f"identical={identical}"
        )
        if identical:
            n_identical += 1

    _check(
        f"all {len(sample_anns)} real masks round-trip pixel-identical "
        f"({n_identical}/{len(sample_anns)} identical)",
        n_identical == len(sample_anns),
    )

    # ---- Part 2: build a tiny fake submission CSV, read it back, validate RLE parses
    print("Part 2: build a tiny fake submission CSV and read it back")

    def _rasterize(ann):
        img = images_by_id[ann["image_id"]]
        h, w = img["height"], img["width"]
        rles = maskUtils.frPyObjects(ann["segmentation"], h, w)
        rle = maskUtils.merge(rles) if len(rles) > 1 else rles[0]
        return maskUtils.decode(rle).astype(np.uint8), h, w

    m0, h0, w0 = _rasterize(sample_anns[0])
    m1, h1, w1 = _rasterize(sample_anns[1])
    m2, h2, w2 = _rasterize(sample_anns[2])

    fake_predictions = {
        "20150125172714Mh": [m0, m1],  # image with 2 predicted filaments
        "20140609195854Bh": [m2],  # image with 1 predicted filament
        "20140101000000Xy": [],  # image with 0 predicted filaments -> 0 rows
    }

    tmp_dir = tempfile.mkdtemp(prefix="solarseg_submission_smoketest_")
    csv_path = os.path.join(tmp_dir, "fake_submission.csv")
    build_submission_csv(fake_predictions, csv_path)
    print(f"  wrote {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  read back {len(df)} rows, columns={list(df.columns)}")

    _check(
        "row count == number of non-empty predicted masks (2+1+0=3)",
        len(df) == 3,
    )
    _check(
        "columns are exactly ['filament_id', 'segmentation_rle']",
        list(df.columns) == ["filament_id", "segmentation_rle"],
    )
    _check(
        "filament_ids are as expected",
        list(df["filament_id"]) == [
            "20150125172714Mh_1",
            "20150125172714Mh_2",
            "20140609195854Bh_1",
        ],
    )
    _check("no placeholder row for the 0-prediction image", "20140101000000Xy" not in "".join(df["filament_id"]))

    # decode each row's RLE back into a mask and confirm it matches the source mask
    expected_masks = {
        "20150125172714Mh_1": (m0, h0, w0),
        "20150125172714Mh_2": (m1, h1, w1),
        "20140609195854Bh_1": (m2, h2, w2),
    }
    all_rows_valid = True
    for _, row in df.iterrows():
        fid = row["filament_id"]
        rle_str = str(row["segmentation_rle"])
        exp_mask, h, w = expected_masks[fid]
        parsed_mask = rle_string_to_mask(rle_str, h, w)
        valid = parsed_mask.shape == (h, w) and np.array_equal(parsed_mask, exp_mask)
        print(f"    row {fid}: parsed shape={parsed_mask.shape} pixel-identical-to-source={valid}")
        all_rows_valid &= valid
    _check("every CSV row's RLE parses back into the exact source mask", all_rows_valid)

    print()
    if all_pass:
        print("ALL CHECKS: PASS")
    else:
        print("ALL CHECKS: FAIL (see [FAIL] lines above)")
        sys.exit(1)
