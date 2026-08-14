"""
maskrle: mask/RLE round-trip integrity EDA for MAGFiLO_1.0 (SolarSeg competition).

Covers, for ALL 8199 annotations (full pass, no sampling -- runtime was
acceptable, see summary):
  1. segmentation -> RLE (pycocotools.mask.frPyObjects + merge) -> decode ->
     area via pycocotools.mask.area vs stored 'area' field (relative error dist).
  2. bbox via pycocotools.mask.toBbox vs stored 'bbox' (mismatch flagging).
  3. Degenerate-polygon flags: <3 distinct (x,y) points, zero decoded pixels,
     self-intersecting ring (via shapely -- installed, so this check IS run).
  4. Polygon coordinate bounds check against [0,width] x [0,height].
  5. Source-JPEG pixel intensity: mean-inside-mask vs mean-in-local-annulus
     (dilated bbox minus mask), effect size, aggregated.
  6. Concrete submission-format round trip demo (counts-only RLE string) on a
     handful of examples, verified pixel-identical against the original mask.

Outputs (all under D:/SolarSeg/notebooks/eda_assets/):
  maskrle_per_annotation_validation.csv   -- full per-annotation table
  maskrle_area_relerr_hist.png
  maskrle_bbox_mismatch_hist.png
  maskrle_intensity_inside_vs_outside.png
  maskrle_submission_roundtrip_examples.csv
  (this script itself lives under scripts/)

Run: python maskrle_mask_roundtrip_validation.py
"""

import json
import os
import time
import numpy as np
import pandas as pd
import cv2
from pycocotools import mask as maskUtils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from shapely.geometry import LinearRing
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False

# ---------------------------------------------------------------------------
DATA_ROOT = "D:/SolarSeg/data/MAGFiLO_1.0_Kaggle_2026"
ANN_PATH = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
IMG_DIR = os.path.join(DATA_ROOT, "train", "train_images")
OUT_DIR = "D:/SolarSeg/notebooks/eda_assets"
SLUG = "maskrle"

os.makedirs(OUT_DIR, exist_ok=True)

AREA_RELERR_FLAG_THRESH = 0.01     # 1%
BBOX_MISMATCH_PX_THRESH = 1.0      # px, generous rounding allowance
NEIGHBORHOOD_DILATE_PX = 25        # local annulus width around bbox

N_ROUNDTRIP_EXAMPLES = 5

# ---------------------------------------------------------------------------


def load_annotations():
    with open(ANN_PATH, "r") as f:
        data = json.load(f)
    return data


def polygon_to_rle(seg_flat, h, w):
    """segmentation is [[x0,y0,x1,y1,...]] -- exactly one ring per our schema check."""
    rles = maskUtils.frPyObjects([seg_flat], h, w)
    rle = maskUtils.merge(rles)
    return rle


def distinct_points(seg_flat):
    pts = np.array(seg_flat, dtype=np.float64).reshape(-1, 2)
    # polygon is closed (first==last) per spec -- drop the closing duplicate
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    uniq = np.unique(pts, axis=0)
    return pts, uniq


def check_self_intersecting(pts):
    """pts: Nx2 array, open ring (no duplicated closing vertex)."""
    if len(pts) < 3:
        return False  # not a real ring; degenerate check catches this separately
    try:
        ring = LinearRing(pts)
        return not ring.is_simple
    except Exception:
        # e.g. duplicate consecutive points collapsing the ring
        return True


def main():
    t0 = time.time()
    data = load_annotations()
    images = {im["id"]: im for im in data["images"]}
    anns = data["annotations"]
    n_total = len(anns)
    print(f"Loaded {len(images)} image entries, {n_total} annotations.")

    # sanity: iscrowd always 0?
    iscrowd_vals = set(a.get("iscrowd", None) for a in anns)
    print("Distinct iscrowd values across all annotations:", iscrowd_vals)

    # sanity: image_id suffix matches file_name stem?
    id_mismatch = 0
    for im in data["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        if not im["id"].endswith(stem):
            id_mismatch += 1
    print(f"images[] entries where id-suffix != file_name stem: {id_mismatch} / {len(images)}")

    # image cache: decode each source JPEG at most once
    img_cache = {}

    def get_image(file_name):
        if file_name not in img_cache:
            path = os.path.join(IMG_DIR, file_name)
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_cache[file_name] = im
        return img_cache[file_name]

    rows = []
    roundtrip_examples = []
    missing_image_files = set()

    for i, ann in enumerate(anns):
        if i % 1000 == 0:
            print(f"  ...{i}/{n_total} ({time.time()-t0:.1f}s elapsed)")

        aid = ann["id"]
        image_id = ann["image_id"]
        im_meta = images.get(image_id)
        if im_meta is None:
            # shouldn't happen per schema, but guard anyway
            rows.append(dict(
                id=aid, image_id=image_id, area_stored=ann.get("area"),
                area_recomputed=np.nan, area_relerr=np.nan,
                bbox_stored=str(ann.get("bbox")), bbox_recomputed=None,
                bbox_max_abs_diff=np.nan,
                degenerate_flag="missing_image_meta",
                out_of_bounds_flag=False, self_intersecting_flag=False,
                mean_inside=np.nan, mean_outside=np.nan,
            ))
            continue

        h, w = im_meta["height"], im_meta["width"]
        seg_list = ann["segmentation"]
        assert len(seg_list) == 1, f"ann {aid} has {len(seg_list)} rings, expected 1"
        seg_flat = seg_list[0]

        pts, uniq = distinct_points(seg_flat)
        n_distinct = len(uniq)

        # bounds check
        out_of_bounds = bool(((pts[:, 0] < 0) | (pts[:, 0] > w) |
                               (pts[:, 1] < 0) | (pts[:, 1] > h)).any()) if len(pts) else False

        # self-intersection check (shapely available)
        self_intersecting = check_self_intersecting(pts) if HAVE_SHAPELY else False

        degenerate_reasons = []
        if n_distinct < 3:
            degenerate_reasons.append("lt3_points")

        # RLE round trip
        try:
            rle = polygon_to_rle(seg_flat, h, w)
            area_recomputed = float(maskUtils.area(rle))
            bbox_recomputed = maskUtils.toBbox(rle).tolist()
            decoded = maskUtils.decode(rle)  # HxW uint8 {0,1}
            n_mask_px = int(decoded.sum())
        except Exception as e:
            degenerate_reasons.append(f"rle_error:{e}")
            area_recomputed = np.nan
            bbox_recomputed = [np.nan] * 4
            decoded = None
            n_mask_px = 0

        if n_mask_px == 0:
            degenerate_reasons.append("zero_pixels")
        if self_intersecting:
            degenerate_reasons.append("self_intersecting")

        area_stored = ann.get("area", np.nan)
        if area_stored and not np.isnan(area_recomputed):
            area_relerr = abs(area_recomputed - area_stored) / area_stored if area_stored != 0 else np.nan
        else:
            area_relerr = np.nan

        bbox_stored = ann.get("bbox", [np.nan] * 4)
        if bbox_recomputed and not any(np.isnan(bbox_recomputed)):
            bbox_max_abs_diff = float(np.max(np.abs(np.array(bbox_stored) - np.array(bbox_recomputed))))
        else:
            bbox_max_abs_diff = np.nan

        # pixel intensity: inside mask vs local outside neighborhood
        mean_inside = np.nan
        mean_outside = np.nan
        file_name = im_meta["file_name"]
        img = get_image(file_name)
        if img is None:
            missing_image_files.add(file_name)
        elif decoded is not None and n_mask_px > 0:
            x, y, bw, bh = ann["bbox"]
            x0 = max(0, int(np.floor(x)) - NEIGHBORHOOD_DILATE_PX)
            y0 = max(0, int(np.floor(y)) - NEIGHBORHOOD_DILATE_PX)
            x1 = min(w, int(np.ceil(x + bw)) + NEIGHBORHOOD_DILATE_PX)
            y1 = min(h, int(np.ceil(y + bh)) + NEIGHBORHOOD_DILATE_PX)

            mask_bool = decoded.astype(bool)
            local_region = np.zeros_like(mask_bool)
            local_region[y0:y1, x0:x1] = True
            outside_region = local_region & (~mask_bool)

            inside_vals = img[mask_bool]
            outside_vals = img[outside_region]
            if inside_vals.size > 0:
                mean_inside = float(inside_vals.mean())
            if outside_vals.size > 0:
                mean_outside = float(outside_vals.mean())

        degenerate_flag = ";".join(degenerate_reasons) if degenerate_reasons else ""

        rows.append(dict(
            id=aid,
            image_id=image_id,
            category_id=ann.get("category_id"),
            area_stored=area_stored,
            area_recomputed=area_recomputed,
            area_relerr=area_relerr,
            bbox_stored=str([round(v, 4) for v in bbox_stored]),
            bbox_recomputed=str([round(v, 4) for v in bbox_recomputed]) if bbox_recomputed else None,
            bbox_max_abs_diff=bbox_max_abs_diff,
            n_distinct_points=n_distinct,
            n_mask_px=n_mask_px,
            degenerate_flag=degenerate_flag,
            out_of_bounds_flag=out_of_bounds,
            self_intersecting_flag=bool(self_intersecting),
            mean_inside=mean_inside,
            mean_outside=mean_outside,
        ))

        # stash a few round-trip demo examples (well-formed, reasonably sized)
        if len(roundtrip_examples) < N_ROUNDTRIP_EXAMPLES and not degenerate_reasons and n_mask_px > 50:
            roundtrip_examples.append(dict(
                id=aid, image_id=image_id, seg_flat=seg_flat, h=h, w=w,
                decoded_from_frpyobjects=decoded,
            ))

    print(f"Finished per-annotation pass in {time.time()-t0:.1f}s. "
          f"{len(missing_image_files)} distinct missing/unreadable image files: "
          f"{sorted(missing_image_files)[:10]}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"{SLUG}_per_annotation_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    # -----------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------
    n = len(df)
    area_relerr = df["area_relerr"].dropna()
    n_area_over_thresh = int((area_relerr > AREA_RELERR_FLAG_THRESH).sum())
    print(f"\n--- AREA RELATIVE ERROR (recomputed via pycocotools.mask.area vs stored) ---")
    print(area_relerr.describe())
    print(f"Annotations with area_relerr > {AREA_RELERR_FLAG_THRESH*100:.0f}%: "
          f"{n_area_over_thresh} / {len(area_relerr)} "
          f"({100*n_area_over_thresh/max(len(area_relerr),1):.2f}%)")

    bbox_diff = df["bbox_max_abs_diff"].dropna()
    n_bbox_mismatch = int((bbox_diff > BBOX_MISMATCH_PX_THRESH).sum())
    print(f"\n--- BBOX MAX ABS COORD DIFF (toBbox(rle) vs stored bbox) ---")
    print(bbox_diff.describe())
    print(f"Annotations with bbox max-abs-diff > {BBOX_MISMATCH_PX_THRESH}px: "
          f"{n_bbox_mismatch} / {len(bbox_diff)} "
          f"({100*n_bbox_mismatch/max(len(bbox_diff),1):.2f}%)")

    n_degenerate = int((df["degenerate_flag"] != "").sum())
    n_lt3 = int(df["degenerate_flag"].str.contains("lt3_points", na=False).sum())
    n_zero_px = int(df["degenerate_flag"].str.contains("zero_pixels", na=False).sum())
    n_self_int = int(df["self_intersecting_flag"].sum())
    n_oob = int(df["out_of_bounds_flag"].sum())
    print(f"\n--- DEGENERACY / VALIDITY FLAGS ---")
    print(f"Any degenerate flag set: {n_degenerate} / {n}")
    print(f"  <3 distinct points: {n_lt3}")
    print(f"  zero decoded pixels: {n_zero_px}")
    print(f"  self-intersecting ring (shapely): {n_self_int}")
    print(f"Out-of-[0,W]x[0,H] polygon points: {n_oob} / {n}")

    # -----------------------------------------------------------------
    # Plot: area relative error histogram (log-scale x, clipped view + zoomed)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    finite_err = area_relerr[np.isfinite(area_relerr)]
    axes[0].hist(finite_err.clip(upper=0.05), bins=80, color="#3b6ea5", edgecolor="none")
    axes[0].axvline(AREA_RELERR_FLAG_THRESH, color="red", linestyle="--",
                     label=f"{AREA_RELERR_FLAG_THRESH*100:.0f}% flag threshold")
    axes[0].set_xlabel("area relative error (|recomputed-stored|/stored), clipped at 5%")
    axes[0].set_ylabel("annotation count")
    axes[0].set_title(f"Area relerr distribution (n={len(finite_err)})")
    axes[0].legend()

    axes[1].hist(finite_err.clip(upper=0.002), bins=80, color="#3b6ea5", edgecolor="none")
    axes[1].set_xlabel("area relative error, clipped at 0.2% (zoomed)")
    axes[1].set_ylabel("annotation count")
    axes[1].set_title("Zoomed view near zero")
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, f"{SLUG}_area_relerr_hist.png")
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {hist_path}")

    # -----------------------------------------------------------------
    # Plot: bbox mismatch histogram
    # -----------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.hist(bbox_diff.clip(upper=5), bins=60, color="#a53b3b", edgecolor="none")
    plt.axvline(BBOX_MISMATCH_PX_THRESH, color="black", linestyle="--",
                label=f"{BBOX_MISMATCH_PX_THRESH}px flag threshold")
    plt.xlabel("max abs coordinate diff, stored bbox vs toBbox(rle) [px], clipped at 5px")
    plt.ylabel("annotation count")
    plt.title(f"bbox(stored) vs pycocotools.mask.toBbox(rle) mismatch (n={len(bbox_diff)})")
    plt.legend()
    plt.tight_layout()
    bbox_path = os.path.join(OUT_DIR, f"{SLUG}_bbox_mismatch_hist.png")
    plt.savefig(bbox_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {bbox_path}")

    # -----------------------------------------------------------------
    # Plot: intensity inside vs outside
    # -----------------------------------------------------------------
    inten = df[["mean_inside", "mean_outside"]].dropna()
    diff = inten["mean_outside"] - inten["mean_inside"]  # positive => inside darker
    frac_darker = float((diff > 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(inten["mean_inside"], bins=60, alpha=0.6, label="mean inside mask", color="#204060")
    axes[0].hist(inten["mean_outside"], bins=60, alpha=0.6, label="mean outside (local annulus)", color="#c08020")
    axes[0].set_xlabel("mean grayscale pixel intensity (0-255)")
    axes[0].set_ylabel("annotation count")
    axes[0].set_title(f"Inside-mask vs local-outside intensity (n={len(inten)})")
    axes[0].legend()

    axes[1].hist(diff, bins=60, color="#3b8a5a", edgecolor="none")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("mean_outside - mean_inside  (positive = filament darker than surround)")
    axes[1].set_ylabel("annotation count")
    axes[1].set_title(f"{frac_darker*100:.1f}% of annotations: filament darker than local surround")
    plt.tight_layout()
    inten_path = os.path.join(OUT_DIR, f"{SLUG}_intensity_inside_vs_outside.png")
    plt.savefig(inten_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {inten_path}")

    mean_inside_all = inten["mean_inside"].mean()
    mean_outside_all = inten["mean_outside"].mean()
    pooled_std = np.sqrt((inten["mean_inside"].std()**2 + inten["mean_outside"].std()**2) / 2)
    cohens_d = (mean_outside_all - mean_inside_all) / pooled_std if pooled_std > 0 else np.nan
    print(f"\n--- INTENSITY: inside-mask vs local-outside ---")
    print(f"mean(mean_inside) across annotations = {mean_inside_all:.2f}")
    print(f"mean(mean_outside) across annotations = {mean_outside_all:.2f}")
    print(f"mean(outside-inside) = {diff.mean():.2f}, median = {diff.median():.2f}")
    print(f"Cohen's d (outside vs inside, pooled across-annotation-mean std) = {cohens_d:.3f}")
    print(f"Fraction of annotations where inside is strictly darker than outside: {frac_darker*100:.2f}%")

    # -----------------------------------------------------------------
    # Submission-format round trip demo
    # -----------------------------------------------------------------
    demo_rows = []
    print(f"\n--- SUBMISSION-FORMAT ROUND TRIP DEMO ({len(roundtrip_examples)} examples) ---")
    for ex in roundtrip_examples:
        seg_flat, h, w = ex["seg_flat"], ex["h"], ex["w"]
        rle = polygon_to_rle(seg_flat, h, w)
        counts = rle["counts"]  # bytes, from merge() -- this IS the compressed RLE
        # decode straight from RAW counts-only dict (no 'size' passed to decode call
        # itself; decode() requires the size key be present in the dict we hand it,
        # mirroring exactly what a submission-format {'size':[h,w],'counts':...}
        # object would look like -- 'counts' alone is what goes in the CSV column)
        counts_str = counts.decode("ascii") if isinstance(counts, bytes) else counts
        rle_for_decode = {"size": [h, w], "counts": counts_str.encode("ascii") if isinstance(counts_str, str) else counts_str}
        decoded_roundtrip = maskUtils.decode(rle_for_decode)
        original_decoded = ex["decoded_from_frpyobjects"]
        pixel_identical = bool(np.array_equal(decoded_roundtrip, original_decoded))
        print(f"  ann {ex['id']} (image {ex['image_id']}): counts_str[:80]={counts_str[:80]!r}..., "
              f"len={len(counts_str)}, pixel_identical_after_roundtrip={pixel_identical}, "
              f"mask_px={int(original_decoded.sum())}")
        demo_rows.append(dict(
            id=ex["id"], image_id=ex["image_id"], h=h, w=w,
            counts_str=counts_str, counts_str_len=len(counts_str),
            mask_px=int(original_decoded.sum()),
            pixel_identical_after_roundtrip=pixel_identical,
        ))
    demo_df = pd.DataFrame(demo_rows)
    demo_path = os.path.join(OUT_DIR, f"{SLUG}_submission_roundtrip_examples.csv")
    demo_df.to_csv(demo_path, index=False)
    print(f"Wrote {demo_path}")

    # -----------------------------------------------------------------
    # Supplementary check: does the stored bbox match the polygon's OWN
    # coordinate extent (min/max x,y of its vertices)? This isolates whether
    # bbox mismatches (section 2 above) are a rasterization/rounding artifact
    # of the RLE round trip, or a genuine stored-bbox/stored-polygon mismatch
    # that predates any pycocotools involvement.
    # -----------------------------------------------------------------
    geom_rows = []
    for ann in anns:
        seg = np.array(ann["segmentation"][0], dtype=np.float64).reshape(-1, 2)
        gx0, gy0 = seg[:, 0].min(), seg[:, 1].min()
        gx1, gy1 = seg[:, 0].max(), seg[:, 1].max()
        bx, by, bw, bh = ann["bbox"]
        max_diff = max(abs(gx0 - bx), abs(gy0 - by), abs(gx1 - (bx + bw)), abs(gy1 - (by + bh)))
        geom_rows.append(dict(id=ann["id"], image_id=ann["image_id"], geom_vs_bbox_max_diff=max_diff))
    geom_df = pd.DataFrame(geom_rows)
    geom_path = os.path.join(OUT_DIR, f"{SLUG}_polygon_geom_vs_stored_bbox.csv")
    geom_df.to_csv(geom_path, index=False)
    n_geom_over5 = int((geom_df["geom_vs_bbox_max_diff"] > 5).sum())
    n_geom_over1 = int((geom_df["geom_vs_bbox_max_diff"] > 1).sum())
    print(f"\n--- POLYGON-OWN-EXTENT vs STORED BBOX (independent of RLE/rasterization) ---")
    print(geom_df["geom_vs_bbox_max_diff"].describe())
    print(f">1px: {n_geom_over1} ({100*n_geom_over1/len(geom_df):.2f}%), "
          f">5px: {n_geom_over5} ({100*n_geom_over5/len(geom_df):.2f}%)")
    print(f"Wrote {geom_path}")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")

    return dict(
        n_total=n,
        n_area_over_thresh=n_area_over_thresh,
        area_relerr_median=float(area_relerr.median()) if len(area_relerr) else None,
        area_relerr_max=float(area_relerr.max()) if len(area_relerr) else None,
        n_bbox_mismatch=n_bbox_mismatch,
        bbox_diff_median=float(bbox_diff.median()) if len(bbox_diff) else None,
        bbox_diff_max=float(bbox_diff.max()) if len(bbox_diff) else None,
        n_degenerate=n_degenerate,
        n_lt3=n_lt3,
        n_zero_px=n_zero_px,
        n_self_int=n_self_int,
        n_oob=n_oob,
        mean_inside_all=float(mean_inside_all),
        mean_outside_all=float(mean_outside_all),
        cohens_d=float(cohens_d),
        frac_darker=float(frac_darker),
        iscrowd_vals=list(iscrowd_vals),
        id_mismatch=id_mismatch,
        missing_image_files=sorted(missing_image_files),
        n_geom_over1=n_geom_over1,
        n_geom_over5=n_geom_over5,
        geom_vs_bbox_max_diff_max=float(geom_df["geom_vs_bbox_max_diff"].max()),
    )


if __name__ == "__main__":
    summary = main()
    print("\n=== SUMMARY DICT ===")
    print(json.dumps(summary, indent=2))
