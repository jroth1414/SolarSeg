"""
xref_audit.py -- Cross-referencing / edge-case / data-quality audit for MAGFiLO 1.0 (Kaggle 2026).

Slice: cross-referencing images<->annotations<->disk, distribution outliers,
JPEG dimension vs JSON metadata cross-check, inter-annotator-batch mask
agreement (IoU) on repeated base images, raw-JSON null/malformed field scan,
and the exact shape of the top-level `licenses` field.

Run:  python xref_audit.py
Outputs all PNG/CSV artifacts under D:/SolarSeg/notebooks/eda_assets/, all
prefixed with "xref_".
"""
import json
import os
import re
import sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# pycocotools for RLE / IoU computation (mask union + IoU per batch pair)
from pycocotools import mask as maskUtils

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = "D:/SolarSeg"
DATA_ROOT = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "test_images")
ANN_JSON = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
DATA_TREE_ROOT = DATA_ROOT  # root to search for stray annotation files

OUT_DIR = os.path.join(ROOT, "notebooks", "eda_assets")
os.makedirs(OUT_DIR, exist_ok=True)
PREFIX = "xref_"


def outp(name):
    return os.path.join(OUT_DIR, PREFIX + name)


# ----------------------------------------------------------------------------
# Load raw JSON (text) once -- used both for json.load() and a raw-text scan
# for null/missing fields that json.load() would silently coerce/hide.
# ----------------------------------------------------------------------------
print("Loading annotation JSON ...")
with open(ANN_JSON, "r", encoding="utf-8") as f:
    raw_text = f.read()
data = json.loads(raw_text)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]
licenses_field = data["licenses"]
info_field = data["info"]

print(f"images: {len(images)}  annotations: {len(annotations)}  categories: {len(categories)}")

flagged_rows = []  # list of dicts: {id, id_type, reason}


def flag(id_, id_type, reason):
    flagged_rows.append({"id": id_, "id_type": id_type, "reason": reason})


# ============================================================================
# 8. licenses field shape -- ground truth check
# ============================================================================
licenses_shape_report = {
    "python_type": type(licenses_field).__name__,
    "is_list": isinstance(licenses_field, list),
    "is_dict": isinstance(licenses_field, dict),
    "raw_repr": repr(licenses_field),
}
print("licenses field:", licenses_shape_report)

# Cross-check: every images[].license value should reference this license's id
if isinstance(licenses_field, dict):
    license_ids_defined = {licenses_field.get("id")}
else:
    license_ids_defined = {L.get("id") for L in licenses_field} if isinstance(licenses_field, list) else set()

license_vals_used = Counter(img.get("license") for img in images)
unresolvable_license_refs = {k: v for k, v in license_vals_used.items() if k not in license_ids_defined}
print("license id values used in images[]:", dict(license_vals_used))
print("license ids NOT resolvable against the licenses field:", unresolvable_license_refs)

# ============================================================================
# 7. Raw JSON scan for null / missing / malformed fields in images[] / annotations[]
# ============================================================================
IMAGE_REQUIRED_KEYS = {"license", "file_name", "url", "height", "width", "date_captured", "id"}
ANN_REQUIRED_KEYS = {"segmentation", "area", "iscrowd", "spine", "image_id", "bbox", "category_id", "id"}

IMAGE_TYPE_EXPECT = {
    "license": (int,),
    "file_name": (str,),
    "url": (str,),
    "height": (int,),
    "width": (int,),
    "date_captured": (str,),
    "id": (str,),
}
ANN_TYPE_EXPECT = {
    "segmentation": (list,),
    "area": (int, float),
    "iscrowd": (int,),
    "spine": (list,),          # may be empty per prompt -- verified separately below
    "image_id": (str,),
    "bbox": (list,),
    "category_id": (int,),
    "id": (str,),
}

malformed_image_rows = []
for img in images:
    iid = img.get("id", "<NO-ID>")
    missing = IMAGE_REQUIRED_KEYS - set(img.keys())
    if missing:
        flag(iid, "image", f"missing keys: {sorted(missing)}")
        malformed_image_rows.append((iid, "missing_keys", sorted(missing)))
    for k in IMAGE_REQUIRED_KEYS & set(img.keys()):
        v = img[k]
        if v is None:
            flag(iid, "image", f"null value for key '{k}'")
            malformed_image_rows.append((iid, "null_value", k))
        elif not isinstance(v, IMAGE_TYPE_EXPECT[k]):
            flag(iid, "image", f"wrong type for key '{k}': expected {IMAGE_TYPE_EXPECT[k]}, got {type(v).__name__} ({v!r})")
            malformed_image_rows.append((iid, "wrong_type", f"{k}={v!r} ({type(v).__name__})"))
    # width/height sanity (non-positive / absurd values)
    h, w = img.get("height"), img.get("width")
    if isinstance(h, (int, float)) and isinstance(w, (int, float)):
        if h <= 0 or w <= 0:
            flag(iid, "image", f"non-positive height/width: {h}x{w}")
            malformed_image_rows.append((iid, "bad_dims", f"{h}x{w}"))

malformed_ann_rows = []
iscrowd_values = Counter()
spine_lengths = []
spine_missing_or_empty = 0
seg_multi_poly_count = 0
seg_open_ring_count = 0
seg_ring_gap_rows = []  # (annotation_id, image_id, gap_px, n_points) for ALL annotations
RING_GAP_FLAG_THRESHOLD_PX = 1.0  # only flag individual rows above this to avoid CSV noise
for ann in annotations:
    aid = ann.get("id", "<NO-ID>")
    missing = ANN_REQUIRED_KEYS - set(ann.keys())
    if missing:
        flag(aid, "annotation", f"missing keys: {sorted(missing)}")
        malformed_ann_rows.append((aid, "missing_keys", sorted(missing)))
    for k in ANN_REQUIRED_KEYS & set(ann.keys()):
        v = ann[k]
        if v is None:
            # spine=None would be a real finding; note it specifically
            flag(aid, "annotation", f"null value for key '{k}'")
            malformed_ann_rows.append((aid, "null_value", k))
        elif not isinstance(v, ANN_TYPE_EXPECT[k]):
            flag(aid, "annotation", f"wrong type for key '{k}': expected {ANN_TYPE_EXPECT[k]}, got {type(v).__name__} ({v!r})")
            malformed_ann_rows.append((aid, "wrong_type", f"{k}={v!r} ({type(v).__name__})"))

    # iscrowd distribution (docs claim always 0 -- verify)
    if isinstance(ann.get("iscrowd"), int):
        iscrowd_values[ann["iscrowd"]] += 1

    # spine presence/length variability
    spine = ann.get("spine")
    if not spine:
        spine_missing_or_empty += 1
    else:
        spine_lengths.append(len(spine))

    # segmentation shape checks
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        if len(seg) != 1:
            seg_multi_poly_count += 1
            flag(aid, "annotation", f"segmentation has {len(seg)} polygon(s), expected exactly 1")
        else:
            poly = seg[0]
            if isinstance(poly, list) and len(poly) >= 4:
                if len(poly) % 2 != 0:
                    flag(aid, "annotation", f"segmentation polygon has odd number of coords ({len(poly)})")
                else:
                    x0, y0 = poly[0], poly[1]
                    xn, yn = poly[-2], poly[-1]
                    gap_px = ((x0 - xn) ** 2 + (y0 - yn) ** 2) ** 0.5
                    seg_ring_gap_rows.append((aid, ann.get("image_id"), gap_px, len(poly) // 2))
                    if gap_px > 1e-9:
                        seg_open_ring_count += 1
                    if gap_px > RING_GAP_FLAG_THRESHOLD_PX:
                        flag(aid, "annotation",
                             f"segmentation ring first/last point gap = {gap_px:.2f}px "
                             f"(first=({x0},{y0}) last=({xn},{yn})) -- prompt's assumed "
                             f"x0==x_n-1,y0==y_n-1 invariant does NOT hold for this annotation")
            elif isinstance(poly, list):
                flag(aid, "annotation", f"segmentation polygon too short: {len(poly)} coords")

print(f"iscrowd value distribution: {dict(iscrowd_values)}")
print(f"annotations with missing/empty spine: {spine_missing_or_empty} / {len(annotations)}")
if spine_lengths:
    print(f"spine length stats (present only): min={min(spine_lengths)} max={max(spine_lengths)} "
          f"mean={np.mean(spine_lengths):.1f} median={np.median(spine_lengths):.1f}")
print(f"annotations with segmentation != 1 polygon: {seg_multi_poly_count}")

# --- Segmentation ring closure analysis (prompt assumed x0==x_n-1,y0==y_n-1 -- verify) ---
gap_df = pd.DataFrame(seg_ring_gap_rows, columns=["annotation_id", "image_id", "gap_px", "n_points"])
gap_df.to_csv(outp("segmentation_ring_gap_analysis.csv"), index=False)
n_exact = int((gap_df["gap_px"] <= 1e-9).sum())
n_tiny = int(((gap_df["gap_px"] > 1e-9) & (gap_df["gap_px"] <= 0.01)).sum())
n_subpixel = int(((gap_df["gap_px"] > 0.01) & (gap_df["gap_px"] <= 1.0)).sum())
n_real_gap = int((gap_df["gap_px"] > 1.0).sum())
print(f"annotations with EXACTLY closed ring (first==last, diff==0): {n_exact} / {len(gap_df)}")
print(f"annotations with unclosed ring (any nonzero gap): {seg_open_ring_count} / {len(gap_df)} "
      f"({100*seg_open_ring_count/len(gap_df):.1f}%)  <-- prompt's stated invariant does NOT hold")
print(f"  gap in (0, 0.01] px (float rounding noise): {n_tiny}")
print(f"  gap in (0.01, 1.0] px (sub-pixel, likely harmless): {n_subpixel}")
print(f"  gap > 1.0 px (real, meaningful open ring): {n_real_gap} ({100*n_real_gap/len(gap_df):.1f}%)")
print(f"  gap_px stats: min={gap_df['gap_px'].min():.4f} max={gap_df['gap_px'].max():.2f} "
      f"mean={gap_df['gap_px'].mean():.2f} median={gap_df['gap_px'].median():.3f}")

plt.figure(figsize=(8, 5))
plt.hist(gap_df["gap_px"].clip(upper=10), bins=50, color="#3ba05c", edgecolor="black", linewidth=0.3)
plt.axvline(1.0, color="red", linestyle="--", label="1.0 px threshold")
plt.xlabel("Distance between segmentation polygon's first and last point (px, clipped at 10)")
plt.ylabel("Number of annotations")
plt.title(f"Segmentation ring closure gap (n={len(gap_df)})\n"
          f"{n_real_gap} ({100*n_real_gap/len(gap_df):.1f}%) have a gap > 1px -- rings are NOT reliably closed")
plt.legend()
plt.tight_layout()
plt.savefig(outp("segmentation_ring_gap_histogram.png"), dpi=120, bbox_inches="tight")
plt.close()

pd.DataFrame(malformed_image_rows, columns=["image_id", "issue_type", "detail"]).to_csv(
    outp("malformed_image_fields.csv"), index=False)
pd.DataFrame(malformed_ann_rows, columns=["annotation_id", "issue_type", "detail"]).to_csv(
    outp("malformed_annotation_fields.csv"), index=False)

# ============================================================================
# 1. images[].file_name exists on disk? id suffix matches file_name?
# ============================================================================
train_files_on_disk = set(os.listdir(TRAIN_IMG_DIR))
print(f"Files physically present in train_images/: {len(train_files_on_disk)}")

ID_PREFIX_RE = re.compile(r"^\d{6}-(.+)$")

missing_file_rows = []
id_suffix_mismatch_rows = []
id_prefix_malformed_rows = []

for img in images:
    iid = img["id"]
    fname = img["file_name"]
    if fname not in train_files_on_disk:
        flag(iid, "image", f"file_name '{fname}' not found on disk in train_images/")
        missing_file_rows.append((iid, fname))

    m = ID_PREFIX_RE.match(iid)
    if not m:
        flag(iid, "image", f"id does not match expected 'NNNNNN-<stem>' pattern: {iid!r}")
        id_prefix_malformed_rows.append((iid, fname))
        continue
    stem_from_id = m.group(1)
    stem_from_file = os.path.splitext(fname)[0]
    if stem_from_id != stem_from_file:
        flag(iid, "image", f"id suffix '{stem_from_id}' != file_name stem '{stem_from_file}'")
        id_suffix_mismatch_rows.append((iid, fname, stem_from_id, stem_from_file))

print(f"images[] entries whose file_name is missing on disk: {len(missing_file_rows)}")
print(f"images[] entries with malformed id prefix pattern: {len(id_prefix_malformed_rows)}")
print(f"images[] entries where id suffix != file_name stem: {len(id_suffix_mismatch_rows)}")

pd.DataFrame(missing_file_rows, columns=["image_id", "file_name"]).to_csv(
    outp("missing_files_on_disk.csv"), index=False)
pd.DataFrame(id_suffix_mismatch_rows, columns=["image_id", "file_name", "id_suffix", "file_stem"]).to_csv(
    outp("id_filename_mismatches.csv"), index=False)

# Also: how many unique file_name stems are referenced total, vs the 707 actually on disk,
# vs any disk files that are NEVER referenced by any images[] entry (orphan images on disk).
referenced_filenames = {img["file_name"] for img in images}
unreferenced_on_disk = train_files_on_disk - referenced_filenames
referenced_but_absent = referenced_filenames - train_files_on_disk
print(f"unique file_name values referenced across images[]: {len(referenced_filenames)}")
print(f"files present on disk but never referenced in images[]: {len(unreferenced_on_disk)}")
if unreferenced_on_disk:
    print("  examples:", list(sorted(unreferenced_on_disk))[:10])
print(f"file_name values referenced but absent from disk: {len(referenced_but_absent)}")

# Sanity: also check test_images dir doesn't overlap/leak train filenames or vice versa
test_files_on_disk = set(os.listdir(TEST_IMG_DIR))
overlap_train_test = train_files_on_disk & test_files_on_disk
print(f"train/test filename overlap: {len(overlap_train_test)}")

# ============================================================================
# 4. Confirm test/ ships with NO annotation file anywhere in the delivered tree
# ============================================================================
json_files_in_tree = []
for dirpath, dirnames, filenames in os.walk(DATA_TREE_ROOT):
    for fn in filenames:
        if fn.lower().endswith(".json"):
            json_files_in_tree.append(os.path.join(dirpath, fn))
print(f"All .json files found anywhere under {DATA_TREE_ROOT}:")
for jf in json_files_in_tree:
    print("   ", jf)
json_files_under_test = [jf for jf in json_files_in_tree if os.path.normpath(jf).startswith(os.path.normpath(os.path.join(DATA_TREE_ROOT, "test")))]
print(f"JSON files specifically under test/: {len(json_files_under_test)} (expected 0)")
no_leakage_confirmed = (len(json_files_under_test) == 0)

# also check for any other annotation-looking file (csv, xml, txt) under test/
other_candidate_exts = (".csv", ".xml", ".txt", ".coco")
other_files_under_test = []
for dirpath, dirnames, filenames in os.walk(os.path.join(DATA_TREE_ROOT, "test")):
    for fn in filenames:
        if fn.lower().endswith(other_candidate_exts):
            other_files_under_test.append(os.path.join(dirpath, fn))
print(f"Other non-image candidate files under test/: {len(other_files_under_test)} -> {other_files_under_test}")

# ============================================================================
# 2. Orphaned annotations: image_id with no matching images[].id
# ============================================================================
image_id_set = {img["id"] for img in images}
orphan_annotations = [ann for ann in annotations if ann.get("image_id") not in image_id_set]
print(f"Orphaned annotations (image_id not found in images[].id): {len(orphan_annotations)}")
orphan_rows = [(a["id"], a.get("image_id")) for a in orphan_annotations]
pd.DataFrame(orphan_rows, columns=["annotation_id", "image_id"]).to_csv(
    outp("orphaned_annotations.csv"), index=False)
for aid, iid in orphan_rows:
    flag(aid, "annotation", f"orphaned: image_id '{iid}' not present in images[].id")

# ============================================================================
# 3. Distribution outliers: 0-annotation images, and > 99th percentile
# ============================================================================
ann_count_per_image = Counter()
for ann in annotations:
    if ann.get("image_id") in image_id_set:
        ann_count_per_image[ann["image_id"]] += 1

counts_all_images = np.array([ann_count_per_image.get(img["id"], 0) for img in images])
zero_ann_images = [img["id"] for img in images if ann_count_per_image.get(img["id"], 0) == 0]
p99 = float(np.percentile(counts_all_images, 99))
above_p99_images = [(img["id"], ann_count_per_image.get(img["id"], 0)) for img in images
                     if ann_count_per_image.get(img["id"], 0) > p99]
above_p99_images.sort(key=lambda x: -x[1])

print(f"images[] entries with 0 annotations: {len(zero_ann_images)}")
print(f"  examples: {zero_ann_images[:10]}")
print(f"99th percentile of annotations-per-image: {p99}")
print(f"images[] entries above 99th percentile: {len(above_p99_images)}")
print(f"  examples (id, count): {above_p99_images[:10]}")
print(f"annotations-per-image stats: min={counts_all_images.min()} max={counts_all_images.max()} "
      f"mean={counts_all_images.mean():.2f} median={np.median(counts_all_images):.1f}")

for iid in zero_ann_images:
    flag(iid, "image", "0 annotations for this images[] entry")
for iid, c in above_p99_images:
    flag(iid, "image", f"annotation count {c} exceeds 99th percentile ({p99:.1f})")

pd.DataFrame({"image_id": [img["id"] for img in images], "file_name": [img["file_name"] for img in images],
              "annotation_count": counts_all_images}).sort_values("annotation_count", ascending=False).to_csv(
    outp("annotation_counts_per_image.csv"), index=False)

# plot histogram
plt.figure(figsize=(8, 5))
plt.hist(counts_all_images, bins=range(0, int(counts_all_images.max()) + 2), color="#3b6fa0", edgecolor="black", linewidth=0.3)
plt.axvline(p99, color="red", linestyle="--", label=f"99th pct = {p99:.1f}")
plt.xlabel("Annotations per images[] entry")
plt.ylabel("Number of images[] entries")
plt.title("Distribution of annotation count per images[] entry (n=1154)")
plt.legend()
plt.tight_layout()
plt.savefig(outp("annotation_count_histogram.png"), dpi=120, bbox_inches="tight")
plt.close()

# ============================================================================
# 5. images[].width/height vs actual JPEG dimensions on disk
# ============================================================================
print("Checking JPEG dimensions on disk vs JSON metadata ...")
dim_mismatch_rows = []
dim_check_errors = []
# cache actual dims per unique file so we don't reopen the same JPEG repeatedly
# (many images[] entries -> same file_name across annotator batches)
file_dims_cache = {}
unique_filenames = sorted({img["file_name"] for img in images})
for fname in unique_filenames:
    fpath = os.path.join(TRAIN_IMG_DIR, fname)
    if not os.path.exists(fpath):
        continue  # already flagged above as missing file
    try:
        with Image.open(fpath) as im:
            file_dims_cache[fname] = im.size  # (width, height)
    except Exception as e:
        dim_check_errors.append((fname, str(e)))

for img in images:
    fname = img["file_name"]
    if fname not in file_dims_cache:
        continue
    actual_w, actual_h = file_dims_cache[fname]
    json_w, json_h = img.get("width"), img.get("height")
    if (actual_w, actual_h) != (json_w, json_h):
        flag(img["id"], "image", f"dim mismatch: JSON says {json_w}x{json_h}, actual JPEG is {actual_w}x{actual_h}")
        dim_mismatch_rows.append((img["id"], fname, json_w, json_h, actual_w, actual_h))

print(f"Unique files whose dims were read from disk: {len(file_dims_cache)}")
print(f"Files that errored while opening: {len(dim_check_errors)}")
print(f"images[] entries with width/height mismatch vs actual JPEG: {len(dim_mismatch_rows)}")
if dim_mismatch_rows:
    print("  examples:", dim_mismatch_rows[:5])
non_2048_files = [(fn, wh) for fn, wh in file_dims_cache.items() if wh != (2048, 2048)]
print(f"Files whose actual dims != 2048x2048: {len(non_2048_files)}")
if non_2048_files:
    print("  examples:", non_2048_files[:10])

pd.DataFrame(dim_mismatch_rows, columns=["image_id", "file_name", "json_width", "json_height", "actual_width", "actual_height"]).to_csv(
    outp("dimension_mismatches.csv"), index=False)

# ============================================================================
# 6. Inter-annotator-batch agreement (IoU) for repeated base images
# ============================================================================
print("Computing inter-annotator-batch mask-union IoU for images annotated by >1 batch ...")

# group images[] entries by base file_name (== base image); each entry is one "batch"
by_filename = defaultdict(list)
for img in images:
    by_filename[img["file_name"]].append(img)

multi_batch_files = {fn: entries for fn, entries in by_filename.items() if len(entries) > 1}
print(f"Base images (unique file_name) with >1 annotator-batch images[] entry: {len(multi_batch_files)} / {len(by_filename)}")

batch_count_dist = Counter(len(v) for v in by_filename.values())
print(f"Distribution of #batches per base image: {dict(sorted(batch_count_dist.items()))}")

# annotations indexed by image_id for fast lookup
ann_by_image_id = defaultdict(list)
for ann in annotations:
    if ann.get("image_id") in image_id_set:
        ann_by_image_id[ann["image_id"]].append(ann)


def union_mask_for_image_entry(img_entry, h, w):
    """Rasterize ALL annotations belonging to one images[] entry (one batch) into
    a single binary union mask (2048x2048), via pycocotools polygon->RLE->mask,
    then OR them together."""
    anns = ann_by_image_id.get(img_entry["id"], [])
    if not anns:
        return np.zeros((h, w), dtype=np.uint8)
    union = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        seg = ann["segmentation"]
        try:
            rles = maskUtils.frPyObjects(seg, h, w)
            rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
            m = maskUtils.decode(rle)
            if m.ndim == 3:
                m = m.any(axis=2).astype(np.uint8)
            union |= m.astype(np.uint8)
        except Exception as e:
            print(f"    WARN: failed to rasterize ann {ann['id']} for image {img_entry['id']}: {e}")
    return union


def iou_binary(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty -> trivially agree
    return inter / union


# Sample: to keep runtime reasonable, cap at min(60 base images, all) but here we go
# through ALL multi-batch base images since n is manageable (~a few hundred at most).
pair_iou_rows = []
sample_limit = None  # None = use all; set an int to sample for speed if needed
multi_items = list(multi_batch_files.items())
if sample_limit is not None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(multi_items), size=min(sample_limit, len(multi_items)), replace=False)
    multi_items = [multi_items[i] for i in idx]

for fname, entries in multi_items:
    # need actual dims; fall back to JSON dims if file missing (shouldn't happen given check #1)
    if fname in file_dims_cache:
        w, h = file_dims_cache[fname]
    else:
        w, h = entries[0]["width"], entries[0]["height"]
    masks = {}
    for e in entries:
        masks[e["id"]] = union_mask_for_image_entry(e, h, w)
    ids = list(masks.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            iou = iou_binary(masks[ids[i]], masks[ids[j]])
            a_area = int(masks[ids[i]].sum())
            b_area = int(masks[ids[j]].sum())
            pair_iou_rows.append({
                "file_name": fname,
                "image_id_a": ids[i],
                "image_id_b": ids[j],
                "n_batches_for_file": len(entries),
                "mask_area_a_px": a_area,
                "mask_area_b_px": b_area,
                "n_anns_a": len(ann_by_image_id.get(ids[i], [])),
                "n_anns_b": len(ann_by_image_id.get(ids[j], [])),
                "iou": iou,
            })

iou_df = pd.DataFrame(pair_iou_rows)
iou_df.sort_values("iou").to_csv(outp("interannotator_batch_iou.csv"), index=False)

print(f"Computed pairwise batch-union IoU for {len(iou_df)} batch-pairs across {len(multi_batch_files)} base images.")
if len(iou_df):
    print(iou_df["iou"].describe())
    low_iou = iou_df[iou_df["iou"] < 0.5]
    print(f"Batch-pairs with IoU < 0.5 (meaningful disagreement): {len(low_iou)} / {len(iou_df)}")
    if len(low_iou):
        print(low_iou[["file_name", "image_id_a", "image_id_b", "iou"]].head(10).to_string(index=False))
        for _, r in low_iou.iterrows():
            flag(r["image_id_a"], "image", f"low inter-batch mask IoU ({r['iou']:.3f}) vs batch {r['image_id_b']} (file {r['file_name']})")

    plt.figure(figsize=(8, 5))
    plt.hist(iou_df["iou"], bins=30, color="#a03b5c", edgecolor="black", linewidth=0.3)
    plt.xlabel("Pairwise batch-union mask IoU")
    plt.ylabel("Number of batch-pairs")
    plt.title(f"Inter-annotator-batch agreement (union-mask IoU)\n"
              f"n={len(iou_df)} pairs across {len(multi_batch_files)} base images, median={iou_df['iou'].median():.3f}")
    plt.tight_layout()
    plt.savefig(outp("interannotator_batch_iou_histogram.png"), dpi=120, bbox_inches="tight")
    plt.close()
else:
    print("No multi-batch base images found -- skipping IoU histogram.")

# ============================================================================
# Save the master flagged-items CSV
# ============================================================================
flagged_df = pd.DataFrame(flagged_rows, columns=["id", "id_type", "reason"])
flagged_df.to_csv(outp("flagged_items.csv"), index=False)
print(f"\nTotal flagged rows written to flagged_items.csv: {len(flagged_df)}")

# ============================================================================
# Final summary printed to stdout (also mirrored into a small text report)
# ============================================================================
summary_lines = [
    "=== xref_audit.py SUMMARY ===",
    f"images[] entries: {len(images)} | unique file_name values: {len(referenced_filenames)} | annotations[]: {len(annotations)}",
    f"licenses field python type: {licenses_shape_report['python_type']} ; raw value: {licenses_shape_report['raw_repr']}",
    f"license id values referenced by images[] not resolvable: {unresolvable_license_refs}",
    f"missing files on disk (file_name in JSON, absent in train_images/): {len(missing_file_rows)}",
    f"files on disk never referenced by any images[] entry: {len(unreferenced_on_disk)}",
    f"images[] id-prefix pattern malformed: {len(id_prefix_malformed_rows)}",
    f"images[] id-suffix != file_name stem mismatches: {len(id_suffix_mismatch_rows)}",
    f"orphaned annotations (image_id not in images[].id): {len(orphan_annotations)}",
    f"images[] entries with 0 annotations: {len(zero_ann_images)}",
    f"99th percentile annotations-per-image: {p99}",
    f"images[] entries above 99th percentile: {len(above_p99_images)}",
    f"JSON files found under test/ tree (expect 0): {len(json_files_under_test)}",
    f"other candidate annotation-like files under test/: {len(other_files_under_test)}",
    f"train/test filename overlap: {len(overlap_train_test)}",
    f"dimension mismatches (JSON vs actual JPEG): {len(dim_mismatch_rows)}",
    f"files with actual dims != 2048x2048: {len(non_2048_files)}",
    f"iscrowd value distribution: {dict(iscrowd_values)}",
    f"annotations with missing/empty spine: {spine_missing_or_empty} / {len(annotations)}",
    f"annotations with segmentation != 1 polygon: {seg_multi_poly_count}",
    f"annotations with unclosed segmentation ring (any nonzero gap): {seg_open_ring_count} / {len(gap_df)} "
    f"({100*seg_open_ring_count/len(gap_df):.1f}%) -- prompt's assumed x0==x_n-1,y0==y_n-1 invariant is FALSE in practice",
    f"  of those: gap<=0.01px(float noise)={n_tiny}, 0.01-1.0px(subpixel)={n_subpixel}, >1.0px(real gap)={n_real_gap} ({100*n_real_gap/len(gap_df):.1f}%)",
    f"  gap_px: min={gap_df['gap_px'].min():.4f} max={gap_df['gap_px'].max():.2f} mean={gap_df['gap_px'].mean():.2f} median={gap_df['gap_px'].median():.3f}",
    f"malformed/null/missing field rows (images): {len(malformed_image_rows)}",
    f"malformed/null/missing field rows (annotations): {len(malformed_ann_rows)}",
    f"multi-batch base images: {len(multi_batch_files)} / {len(by_filename)}",
    f"batch-pairs IoU computed: {len(iou_df)}",
    f"batch-pairs IoU median: {iou_df['iou'].median() if len(iou_df) else float('nan')}",
    f"total flagged rows: {len(flagged_df)}",
]
report_txt = "\n".join(summary_lines)
print("\n" + report_txt)
with open(outp("summary.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt + "\n")

print("\nDone.")
