"""Scratch script: render several barb-candidate annotations at a fixed
padding around their bbox, so we can eyeball which one best shows barb
(thread-like offshoot) structure before hardcoding a choice into the real
figure script. Not part of the deliverable output set."""
import sys
sys.path.insert(0, "D:/SolarSeg/notebooks/eda_assets/scripts")
from viz_common import load_coco, build_indices, ImageCache, draw_annotations
import matplotlib.pyplot as plt

coco = load_coco()
images_by_id, anns_by_image = build_indices(coco)
anns_by_id = {a["id"]: a for a in coco["annotations"]}
cache = ImageCache()

candidate_ids = [
    "1835f470-893f-439f-bba3-6037eabf14b3",  # 010102-20131016185334Ch, npts=316
    "d0a187df-997c-42f6-889c-1109f0718d13",  # 030403-20121108155954Bh, npts=312
    "17a7867a-611e-4c72-9e97-651ff8cd813d",  # 040301-20130211063134Lh, npts=309
    "eff25f2b-0499-4f4f-b2ba-7c28aac825d7",  # 010401-20140406195854Bh, npts=332
    "aa8d996d-aa94-46df-94bd-6d2989752109",  # 010101-20120206063134Lh, npts=344
    "33f6e4ca-d5d0-40cb-a87e-dfd6aff8dd4b",  # 040301-20111206102934Ch, npts=407
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, ann_id in zip(axes.flat, candidate_ids):
    a = anns_by_id[ann_id]
    im = images_by_id[a["image_id"]]
    arr = cache.get(im["file_name"])
    x, y, w, h = a["bbox"]
    pad = max(w, h) * 0.35
    x0, x1 = max(0, x - pad), min(arr.shape[1], x + w + pad)
    y0, y1 = max(0, y - pad), min(arr.shape[0], y + h + pad)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    draw_annotations(ax, [a], mask_alpha=0.32, lw=1.2, spine_lw=1.6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    npts = len(a["segmentation"][0]) // 2
    ax.set_title(f"{ann_id[:8]} npts={npts}\n{im['file_name']}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("D:/SolarSeg/notebooks/eda_assets/scripts/_barb_candidates_scratch.png", dpi=110)
print("saved")
