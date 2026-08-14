"""
Annotation-level statistics EDA for MAGFiLO 1.0 (Kaggle 2026) solar filament dataset.

Slice owner: annotation-level statistics from the COCO JSON.

Self-contained / re-runnable: reads the train annotation JSON directly from disk,
computes all summary numbers reported in the EDA writeup, and re-saves every
plot (PNG) and table (CSV) into D:/SolarSeg/notebooks/eda_assets/, prefixed
with "anno_".

Run:
    python anno_stats_eda.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = "D:/SolarSeg"
DATA_ROOT = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
ANNO_JSON = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT_DIR = os.path.join(ROOT, "notebooks", "eda_assets")
SCRIPTS_DIR = os.path.join(OUT_DIR, "scripts")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

PREFIX = "anno"


def outpath(name):
    return os.path.join(OUT_DIR, f"{PREFIX}_{name}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_coco(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    print(f"Loading {ANNO_JSON} ...")
    data = load_coco(ANNO_JSON)

    info = data.get("info", {})
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    licenses = data["licenses"]

    print(f"info keys: {list(info.keys())}")
    print(f"#images entries: {len(images)}")
    print(f"#annotations: {len(annotations)}")
    print(f"#categories: {len(categories)}")
    print(f"licenses type: {type(licenses)}")

    # -----------------------------------------------------------------
    # WARNING check: licenses should be a list per Kaggle docs prose,
    # but organizer data actually has it as a single dict.
    # -----------------------------------------------------------------
    licenses_is_list = isinstance(licenses, list)
    print(f"licenses is list? {licenses_is_list} -- actual type: {type(licenses).__name__}")

    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    # -----------------------------------------------------------------
    # Cross-check images[].id suffix against file_name stem (QC signal
    # mentioned in the task -- id looks like '040301-20140609195854Bh'
    # i.e. 6-digit batch prefix + '-' + file stem).
    # -----------------------------------------------------------------
    id_mismatch_examples = []
    for im in images:
        stem = os.path.splitext(im["file_name"])[0]
        img_id = im["id"]
        # expected pattern: "<6digits>-<stem>"
        parts = img_id.split("-", 1)
        if len(parts) != 2 or parts[1] != stem:
            id_mismatch_examples.append((img_id, im["file_name"]))
    print(f"images[].id / file_name stem mismatches: {len(id_mismatch_examples)}")
    if id_mismatch_examples:
        print("  examples:", id_mismatch_examples[:5])

    # ===================================================================
    # 1. Filaments-per-image distribution (across all 1154 images[] entries)
    # ===================================================================
    image_id_list = [im["id"] for im in images]
    anno_count_by_image = Counter(a["image_id"] for a in annotations)
    filaments_per_image = np.array([anno_count_by_image.get(iid, 0) for iid in image_id_list], dtype=int)

    n_zero = int((filaments_per_image == 0).sum())
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pct_vals = np.percentile(filaments_per_image, percentiles)

    fpi_summary = pd.DataFrame({
        "percentile": percentiles,
        "filaments_per_image": pct_vals,
    })
    fpi_summary.loc[len(fpi_summary)] = ["mean", filaments_per_image.mean()]
    fpi_summary.loc[len(fpi_summary)] = ["std", filaments_per_image.std()]
    fpi_summary.loc[len(fpi_summary)] = ["n_images_zero_annotations", n_zero]
    fpi_summary.loc[len(fpi_summary)] = ["n_images_total", len(image_id_list)]
    fpi_summary.to_csv(outpath("filaments_per_image_percentiles.csv"), index=False)

    # images with unusually many annotations (top 15)
    top_images = sorted(zip(image_id_list, filaments_per_image), key=lambda x: -x[1])[:15]
    top_images_df = pd.DataFrame(top_images, columns=["image_id", "n_filaments"])
    top_images_df.to_csv(outpath("top_images_by_filament_count.csv"), index=False)

    plt.figure(figsize=(8, 5))
    max_show = int(filaments_per_image.max())
    bins = np.arange(0, max_show + 2) - 0.5
    plt.hist(filaments_per_image, bins=bins, color="#3b6ea5", edgecolor="black", linewidth=0.3)
    plt.xlabel("Filament annotations per image entry")
    plt.ylabel("Number of image entries")
    plt.title(f"Filaments-per-image distribution (n={len(image_id_list)} image entries)\n"
              f"median={np.median(filaments_per_image):.0f}, "
              f"p90={np.percentile(filaments_per_image,90):.0f}, "
              f"p99={np.percentile(filaments_per_image,99):.0f}, "
              f"max={filaments_per_image.max():.0f}, zero-annot images={n_zero}")
    plt.tight_layout()
    plt.savefig(outpath("filaments_per_image_hist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print("\n--- Filaments per image ---")
    print(fpi_summary.to_string(index=False))

    # ===================================================================
    # 2. category_id distribution
    # ===================================================================
    cat_counts = Counter(a["category_id"] for a in annotations)
    cat_rows = []
    total_annos = len(annotations)
    for c in categories:
        cid = c["id"]
        cnt = cat_counts.get(cid, 0)
        cat_rows.append({
            "category_id": cid,
            "category_name": c["name"],
            "supercategory": c.get("supercategory", ""),
            "count": cnt,
            "pct": 100.0 * cnt / total_annos if total_annos else 0.0,
        })
    cat_df = pd.DataFrame(cat_rows).sort_values("category_id")
    cat_df.to_csv(outpath("category_distribution.csv"), index=False)
    print("\n--- Category distribution ---")
    print(cat_df.to_string(index=False))

    plt.figure(figsize=(7, 5))
    plt.bar(cat_df["category_name"], cat_df["count"], color=["#3b6ea5", "#5aa469", "#c98a3c", "#a05a9c"])
    for i, (cnt, pct) in enumerate(zip(cat_df["count"], cat_df["pct"])):
        plt.text(i, cnt, f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Annotation count")
    plt.title(f"category_id distribution across {total_annos} annotations")
    plt.tight_layout()
    plt.savefig(outpath("category_distribution_bar.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ===================================================================
    # 3. iscrowd check
    # ===================================================================
    iscrowd_vals = Counter(a.get("iscrowd", None) for a in annotations)
    print("\n--- iscrowd value counts ---")
    print(iscrowd_vals)
    nonzero_crowd_examples = [a["id"] for a in annotations if a.get("iscrowd", 0) != 0][:10]
    iscrowd_df = pd.DataFrame(
        [{"iscrowd_value": k, "count": v} for k, v in sorted(iscrowd_vals.items(), key=lambda x: str(x[0]))]
    )
    iscrowd_df.to_csv(outpath("iscrowd_value_counts.csv"), index=False)

    # ===================================================================
    # 4. Segmentation polygon point-count distribution
    # ===================================================================
    def poly_npoints(anno):
        seg = anno.get("segmentation", None)
        if not seg:
            return 0
        # segmentation: list containing exactly one flat polygon
        if isinstance(seg, list) and len(seg) >= 1 and isinstance(seg[0], (list, tuple)):
            flat = seg[0]
        elif isinstance(seg, list) and all(isinstance(v, (int, float)) for v in seg):
            flat = seg
        else:
            flat = seg[0] if seg else []
        return len(flat) // 2

    n_seg_lists = Counter(len(a.get("segmentation", []) or []) for a in annotations)
    print(f"\nsegmentation outer-list length distribution (should always be 1): {n_seg_lists}")

    poly_pts = np.array([poly_npoints(a) for a in annotations], dtype=int)
    poly_pct_vals = np.percentile(poly_pts, percentiles)
    poly_summary = pd.DataFrame({"percentile": percentiles, "num_polygon_points": poly_pct_vals})
    poly_summary.loc[len(poly_summary)] = ["mean", poly_pts.mean()]
    poly_summary.loc[len(poly_summary)] = ["std", poly_pts.std()]
    poly_summary.loc[len(poly_summary)] = ["min", poly_pts.min()]
    poly_summary.loc[len(poly_summary)] = ["max", poly_pts.max()]
    poly_summary.to_csv(outpath("polygon_point_count_percentiles.csv"), index=False)
    print("\n--- Polygon point-count percentiles ---")
    print(poly_summary.to_string(index=False))

    plt.figure(figsize=(8, 5))
    clip_val = np.percentile(poly_pts, 99.5)
    plot_vals = np.clip(poly_pts, 0, clip_val)
    plt.hist(plot_vals, bins=60, color="#c9622f", edgecolor="black", linewidth=0.3)
    plt.xlabel(f"Polygon vertex count (clipped at p99.5={clip_val:.0f} for display)")
    plt.ylabel("Number of annotations")
    plt.title(f"Segmentation polygon point-count distribution (n={len(poly_pts)})\n"
              f"median={np.median(poly_pts):.0f}, p90={np.percentile(poly_pts,90):.0f}, "
              f"p99={np.percentile(poly_pts,99):.0f}, max={poly_pts.max()}")
    plt.tight_layout()
    plt.savefig(outpath("polygon_point_count_hist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ===================================================================
    # 5. spine point-count / presence distribution
    # ===================================================================
    def spine_npoints(anno):
        sp = anno.get("spine", None)
        if sp is None:
            return -1  # sentinel: key missing entirely
        if isinstance(sp, list) and len(sp) >= 1 and isinstance(sp[0], (list, tuple)):
            flat = sp[0]
        else:
            flat = sp
        return len(flat) // 2

    has_spine_key = np.array(["spine" in a for a in annotations])
    spine_pts_raw = [spine_npoints(a) for a in annotations]
    spine_pts = np.array(spine_pts_raw, dtype=int)

    n_missing_key = int((~has_spine_key).sum())
    n_present_key_empty = int(((spine_pts == 0) & has_spine_key).sum())
    n_populated = int((spine_pts > 0).sum())

    print(f"\nspine: key missing entirely = {n_missing_key}, "
          f"key present but empty/0-point = {n_present_key_empty}, "
          f"populated (>0 pts) = {n_populated}, total = {len(annotations)}")

    missing_spine_examples = [a["id"] for a in annotations if "spine" not in a][:10]
    empty_spine_examples = [a["id"] for a in annotations if "spine" in a and spine_npoints(a) == 0][:10]

    spine_present_pts = spine_pts[spine_pts > 0]
    if len(spine_present_pts) > 0:
        spine_pct_vals = np.percentile(spine_present_pts, percentiles)
    else:
        spine_pct_vals = [np.nan] * len(percentiles)
    spine_summary = pd.DataFrame({"percentile": percentiles, "num_spine_points_when_present": spine_pct_vals})
    spine_summary.loc[len(spine_summary)] = ["n_missing_key", n_missing_key]
    spine_summary.loc[len(spine_summary)] = ["n_present_key_but_empty", n_present_key_empty]
    spine_summary.loc[len(spine_summary)] = ["n_populated_gt0", n_populated]
    spine_summary.loc[len(spine_summary)] = ["n_total_annotations", len(annotations)]
    spine_summary.to_csv(outpath("spine_point_count_percentiles.csv"), index=False)
    print("\n--- Spine point-count summary ---")
    print(spine_summary.to_string(index=False))

    plt.figure(figsize=(8, 5))
    if len(spine_present_pts) > 0:
        clip_val_sp = np.percentile(spine_present_pts, 99.5)
        plot_vals_sp = np.clip(spine_present_pts, 0, clip_val_sp)
        plt.hist(plot_vals_sp, bins=60, color="#4c8c4a", edgecolor="black", linewidth=0.3)
        plt.title(f"Spine point-count distribution, populated spines only (n={len(spine_present_pts)}/{len(annotations)})\n"
                  f"median={np.median(spine_present_pts):.0f}, p90={np.percentile(spine_present_pts,90):.0f}, "
                  f"missing/empty={n_missing_key + n_present_key_empty}")
    else:
        plt.text(0.5, 0.5, "No populated spines found", ha="center", va="center")
        plt.title("Spine point-count distribution -- NONE POPULATED")
    plt.xlabel("Spine vertex count (clipped at p99.5 for display)")
    plt.ylabel("Number of annotations")
    plt.tight_layout()
    plt.savefig(outpath("spine_point_count_hist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ===================================================================
    # 6. bbox aspect ratio, area, segmentation area, bbox-vs-seg-area
    # ===================================================================
    bbox_x = np.array([a["bbox"][0] for a in annotations], dtype=float)
    bbox_y = np.array([a["bbox"][1] for a in annotations], dtype=float)
    bbox_w = np.array([a["bbox"][2] for a in annotations], dtype=float)
    bbox_h = np.array([a["bbox"][3] for a in annotations], dtype=float)
    seg_area = np.array([a["area"] for a in annotations], dtype=float)

    eps = 1e-6
    wh_ratio = bbox_w / np.maximum(bbox_h, eps)
    hw_ratio = bbox_h / np.maximum(bbox_w, eps)
    elong_ratio = np.maximum(wh_ratio, hw_ratio)  # long side / short side, >=1
    bbox_area = bbox_w * bbox_h
    area_fill_ratio = seg_area / np.maximum(bbox_area, eps)  # seg area as fraction of bbox area

    aspect_summary = pd.DataFrame({
        "percentile": percentiles,
        "w_over_h": np.percentile(wh_ratio, percentiles),
        "h_over_w": np.percentile(hw_ratio, percentiles),
        "elongation_long_over_short": np.percentile(elong_ratio, percentiles),
        "bbox_area_px": np.percentile(bbox_area, percentiles),
        "segmentation_area_px": np.percentile(seg_area, percentiles),
        "seg_area_over_bbox_area": np.percentile(area_fill_ratio, percentiles),
    })
    aspect_summary.to_csv(outpath("bbox_area_aspect_percentiles.csv"), index=False)
    print("\n--- bbox / area / aspect percentiles ---")
    print(aspect_summary.to_string(index=False))

    print(f"\nMean seg_area / bbox_area fill ratio: {area_fill_ratio.mean():.4f}")
    print(f"Median seg_area / bbox_area fill ratio: {np.median(area_fill_ratio):.4f}")
    print(f"Mean elongation (long/short bbox side): {elong_ratio.mean():.2f}")
    print(f"Median elongation: {np.median(elong_ratio):.2f}")

    # elongation histogram
    plt.figure(figsize=(8, 5))
    clip_elong = np.percentile(elong_ratio, 99)
    plt.hist(np.clip(elong_ratio, 1, clip_elong), bins=60, color="#a05a9c", edgecolor="black", linewidth=0.3)
    plt.xlabel(f"bbox elongation ratio = long side / short side (clipped at p99={clip_elong:.1f})")
    plt.ylabel("Number of annotations")
    plt.title(f"bbox aspect-ratio (elongation) distribution\n"
              f"median={np.median(elong_ratio):.2f}, p90={np.percentile(elong_ratio,90):.2f}, "
              f"p99={np.percentile(elong_ratio,99):.2f}")
    plt.tight_layout()
    plt.savefig(outpath("bbox_elongation_hist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # bbox area vs segmentation area scatter (log-log)
    plt.figure(figsize=(7, 7))
    plt.scatter(bbox_area, seg_area, s=4, alpha=0.25, color="#3b6ea5")
    lim_max = max(bbox_area.max(), seg_area.max()) * 1.1
    lim_min = max(min(bbox_area.min(), seg_area.min()), 1)
    xs = np.logspace(np.log10(lim_min), np.log10(lim_max), 50)
    plt.plot(xs, xs, "k--", linewidth=1, label="seg_area = bbox_area (upper bound)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("bbox area = bbox_w * bbox_h (px^2, log scale)")
    plt.ylabel("segmentation area, pycocotools RLE (px^2, log scale)")
    plt.title("Segmentation area vs bbox area\n(filaments are thin -> seg area should sit well below bbox area)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath("segarea_vs_bboxarea_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # fill-ratio histogram
    plt.figure(figsize=(8, 5))
    plt.hist(area_fill_ratio, bins=60, range=(0, 1), color="#c9622f", edgecolor="black", linewidth=0.3)
    plt.xlabel("segmentation_area / bbox_area (fill fraction)")
    plt.ylabel("Number of annotations")
    plt.title(f"How much of its own bbox does each filament mask actually fill?\n"
              f"mean={area_fill_ratio.mean():.3f}, median={np.median(area_fill_ratio):.3f}")
    plt.tight_layout()
    plt.savefig(outpath("seg_bbox_fill_ratio_hist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # sanity check: any seg_area > bbox_area (should not happen; flag if so)
    over_bbox = np.array(annotations)[area_fill_ratio > 1.0001] if len(annotations) else np.array([])
    n_over_bbox = int((area_fill_ratio > 1.0001).sum())
    over_bbox_examples = [a["id"] for a, r in zip(annotations, area_fill_ratio) if r > 1.0001][:10]
    print(f"\nAnnotations with segmentation area > bbox area (should be 0, sanity check): {n_over_bbox}")
    if over_bbox_examples:
        print("  examples:", over_bbox_examples)

    # ===================================================================
    # 7. categories / licenses verbatim dump
    # ===================================================================
    print("\n--- categories (verbatim) ---")
    print(json.dumps(categories, indent=2))
    print("\n--- licenses (verbatim) ---")
    print(json.dumps(licenses, indent=2))
    print(f"\n>>> licenses is a {type(licenses).__name__} in the actual JSON "
          f"(Kaggle docs prose claims it is a list) <<<")

    cat_verbatim_df = pd.DataFrame(categories)
    cat_verbatim_df.to_csv(outpath("categories_table_verbatim.csv"), index=False)

    if isinstance(licenses, dict):
        lic_verbatim_df = pd.DataFrame([licenses])
    else:
        lic_verbatim_df = pd.DataFrame(licenses)
    lic_verbatim_df.to_csv(outpath("licenses_table_verbatim.csv"), index=False)

    # ===================================================================
    # Flattened per-annotation CSV (shared contract with other agents)
    # ===================================================================
    flat_rows = []
    for a, npts, nspine in zip(annotations, poly_pts, spine_pts):
        flat_rows.append({
            "id": a["id"],
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "area": a["area"],
            "bbox_x": a["bbox"][0],
            "bbox_y": a["bbox"][1],
            "bbox_w": a["bbox"][2],
            "bbox_h": a["bbox"][3],
            "num_polygon_points": int(npts),
            "num_spine_points": int(max(nspine, 0)),
            "iscrowd": a.get("iscrowd", None),
        })
    flat_df = pd.DataFrame(flat_rows, columns=[
        "id", "image_id", "category_id", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "num_polygon_points", "num_spine_points", "iscrowd",
    ])
    flat_df.to_csv(outpath("annotations_flat.csv"), index=False)
    print(f"\nSaved flattened per-annotation CSV: {outpath('annotations_flat.csv')} ({len(flat_df)} rows)")

    # ===================================================================
    # Write a small text report capturing key printed numbers (for the
    # synthesis step / human review), in addition to the CSVs above.
    # ===================================================================
    report_lines = []
    report_lines.append("MAGFiLO annotation-level EDA -- key numbers")
    report_lines.append("=" * 60)
    report_lines.append(f"images[] entries: {len(images)}")
    report_lines.append(f"unique file stems: {len(set(os.path.splitext(im['file_name'])[0] for im in images))}")
    report_lines.append(f"annotations: {len(annotations)}")
    report_lines.append(f"images[].id/file_name stem mismatches: {len(id_mismatch_examples)}")
    report_lines.append(f"licenses is list? {licenses_is_list} (type={type(licenses).__name__})")
    report_lines.append("")
    report_lines.append("-- filaments per image --")
    report_lines.append(fpi_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("-- category distribution --")
    report_lines.append(cat_df.to_string(index=False))
    report_lines.append("")
    report_lines.append(f"-- iscrowd value counts -- {dict(iscrowd_vals)}")
    if nonzero_crowd_examples:
        report_lines.append(f"   nonzero iscrowd examples: {nonzero_crowd_examples}")
    report_lines.append("")
    report_lines.append("-- polygon point-count percentiles --")
    report_lines.append(poly_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("-- spine point-count summary --")
    report_lines.append(spine_summary.to_string(index=False))
    if missing_spine_examples:
        report_lines.append(f"   spine key missing examples: {missing_spine_examples}")
    if empty_spine_examples:
        report_lines.append(f"   spine present-but-empty examples: {empty_spine_examples}")
    report_lines.append("")
    report_lines.append("-- bbox/area/aspect percentiles --")
    report_lines.append(aspect_summary.to_string(index=False))
    report_lines.append(f"   seg_area > bbox_area count: {n_over_bbox} examples={over_bbox_examples}")
    report_lines.append("")
    report_lines.append("-- categories verbatim --")
    report_lines.append(json.dumps(categories, indent=2))
    report_lines.append("")
    report_lines.append("-- licenses verbatim --")
    report_lines.append(json.dumps(licenses, indent=2))

    with open(outpath("key_numbers_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("\nDone.")


if __name__ == "__main__":
    main()
