"""
Illustrative figures for the MAGFiLO / Solar Filament Segmentation EDA notebook.

Produces three standalone PNGs under D:/SolarSeg/notebooks/eda_assets/ :

  viz_sample_grid.png       - 12 train images, spread across all 6 observatory
                               codes and capture years 2011-2022, with filament
                               masks (colored by category_id) + spines overlaid.
  viz_barb_zoom.png         - a tight full-resolution crop on a single filament
                               with a very high polygon point count, chosen
                               because it visibly shows barb (thread-like
                               offshoot) structure.
  viz_overlap_cluster.png   - a crop showing several closely-adjacent /
                               overlapping filament instances in one image, to
                               illustrate the instance-separation challenge
                               that Panoptic Quality penalizes (fragmentation /
                               over-merging).

Every figure is produced by a reusable `plot_*` function (importable from this
module) so a synthesis notebook can call the same logic live in a cell rather
than only embedding the PNGs. Running this file directly (`python
viz_illustrative_figures.py`) regenerates all three from scratch.

How candidates were chosen (see notebooks/eda_assets/scripts/explore_candidates.py
and inspect_barb_candidates.py for the scratch exploration that led here):
  - Grid images: for each of the 6 observatory codes, 2 distinct file_names
    were picked spanning different years, restricted to images with a
    "legible" annotation count (3-9 masks) so the small subplot isn't
    overwhelmed.
  - Barb zoom: ranked all 8199 annotations by polygon point count (a proxy for
    boundary complexity / fine structure) and visually reviewed the top
    candidates with moderate bbox size (so a tight crop stays legible); chose
    annotation d0a187df-... (312 points) in image 20121108155954Bh.jpeg, which
    shows an unmistakable fishbone/feather pattern of barbs along the spine.
  - Overlap cluster: for every image with >=3 annotations, counted bbox-bbox
    IoU>0 pairs; chose image 010101-20150518145654Bh (5 overlapping pairs
    among 24 annotations) and cropped to the densest sub-region, which
    contains two crossing filaments plus a tight 5-filament cluster.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz_common import (
    ASSET_DIR, SLUG, CATEGORY_STYLE, SPINE_COLOR,
    load_coco, build_indices, ImageCache, draw_annotations, category_legend_handles,
)

# ---------------------------------------------------------------------------
# Figure 1: sample grid across observatory codes / capture years
# ---------------------------------------------------------------------------

# (file_name, images[].id) -- one distinct source image per cell, 2 per
# observatory code, spanning 2011-2022. See module docstring for how these
# were selected.
GRID_SAMPLES = [
    ("20120223152054Bh.jpeg", "040301-20120223152054Bh"),
    ("20170831170750Bh.jpeg", "010302-20170831170750Bh"),
    ("20110823120334Ch.jpeg", "010202-20110823120334Ch"),
    ("20211222120130Ch.jpeg", "060403-20211222120130Ch"),
    ("20140218230234Lh.jpeg", "040303-20140218230234Lh"),
    ("20210103223630Lh.jpeg", "030101-20210103223630Lh"),
    ("20120427162614Mh.jpeg", "040102-20120427162614Mh"),
    ("20220617185352Mh.jpeg", "010202-20220617185352Mh"),
    ("20130423072414Th.jpeg", "060102-20130423072414Th"),
    ("20161222133714Th.jpeg", "010101-20161222133714Th"),
    ("20131108045254Uh.jpeg", "040401-20131108045254Uh"),
    ("20170122034654Uh.jpeg", "040301-20170122034654Uh"),
]

OBS_CODE_NAME = {
    "Bh": "Big Bear",
    "Ch": "Cerro Tololo",
    "Lh": "Learmonth",
    "Mh": "Mauna Loa",
    "Th": "Teide",
    "Uh": "Udaipur",
}


def plot_sample_grid(coco, images_by_id, anns_by_image, cache, samples=GRID_SAMPLES,
                      ncols=4, save_path=None):
    """Grid of sample train images with mask+spine overlays. Returns the
    matplotlib Figure. Saves to `save_path` if given."""
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (file_name, image_id) in zip(axes, samples):
        im = images_by_id[image_id]
        arr = cache.get(im["file_name"])
        anns = anns_by_image.get(image_id, [])
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        draw_annotations(ax, anns, mask_alpha=0.4, lw=0.8, spine_lw=1.0)
        code = file_name[-7:-5]
        date_str = f"{file_name[:4]}-{file_name[4:6]}-{file_name[6:8]}"
        obs_name = OBS_CODE_NAME.get(code, code)
        ax.set_title(f"{date_str}  {obs_name} ({code})\n{len(anns)} filaments",
                      fontsize=10)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    handles = category_legend_handles(include_spine=True)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)
    fig.suptitle(
        "MAGFiLO train samples across observatories and years\n"
        "(masks colored by category_id; dashed line = annotated spine)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: tight zoom on a single filament showing barb structure
# ---------------------------------------------------------------------------

BARB_ANN_ID = "d0a187df-997c-42f6-889c-1109f0718d13"
BARB_PAD_FRAC = 0.18  # padding around the bbox, as a fraction of max(w, h)


def plot_barb_zoom(coco, images_by_id, anns_by_image, cache, ann_id=BARB_ANN_ID,
                    pad_frac=BARB_PAD_FRAC, save_path=None):
    """Full-resolution crop around a single filament chosen for its high
    polygon-point-count (structural complexity proxy) that visibly shows barb
    offshoots from the main spine. Returns the Figure."""
    anns_by_id = {a["id"]: a for a in coco["annotations"]}
    a = anns_by_id[ann_id]
    im = images_by_id[a["image_id"]]
    arr = cache.get(im["file_name"])

    x, y, w, h = a["bbox"]
    pad = max(w, h) * pad_frac
    x0, x1 = max(0, x - pad), min(arr.shape[1], x + w + pad)
    y0, y1 = max(0, y - pad), min(arr.shape[0], y + h + pad)

    npts = len(a["segmentation"][0]) // 2
    cat_name, _ = CATEGORY_STYLE[a["category_id"]]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    draw_annotations(ax, [a], mask_alpha=0.32, lw=1.3, spine_lw=1.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")

    handles = category_legend_handles(include_spine=True)
    # only show the one category actually present + spine
    handles = [h_ for h_ in handles if cat_name in h_.get_label() or "spine" in h_.get_label()]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=10)

    ax.set_title(
        f"Barb (thread-like offshoot) structure on a single filament\n"
        f"{im['file_name']}  |  category={cat_name}  |  "
        f"{npts} polygon vertices  |  bbox {w:.0f}x{h:.0f} px  (full 2048x2048 image)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: cluster of overlapping / closely-adjacent filaments
# ---------------------------------------------------------------------------

OVERLAP_IMAGE_ID = "010101-20150518145654Bh"
OVERLAP_CROP = (1080, 1550, 830, 1200)  # x0, x1, y0, y1 in pixel coords


def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def find_top_overlap_images(coco, anns_by_image, min_anns=3, top_n=20):
    """Utility used during candidate selection: ranks images by count of
    bbox-pairs with IoU>0. Returned here (not just in the scratch script) so
    a synthesis notebook can re-derive/verify the chosen example instead of
    trusting a hardcoded id."""
    scores = []
    for image_id, anns in anns_by_image.items():
        if len(anns) < min_anns:
            continue
        n_overlap = 0
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if bbox_iou(anns[i]["bbox"], anns[j]["bbox"]) > 0:
                    n_overlap += 1
        scores.append((n_overlap, len(anns), image_id))
    scores.sort(key=lambda t: (-t[0], -t[1]))
    return scores[:top_n]


def plot_overlap_cluster(coco, images_by_id, anns_by_image, cache,
                          image_id=OVERLAP_IMAGE_ID, crop=OVERLAP_CROP, save_path=None):
    """Crop of an image region containing several overlapping / closely
    adjacent filament instances, illustrating the instance-separation
    challenge PQ penalizes (fragmentation, over-/under-merging). Returns the
    Figure."""
    im = images_by_id[image_id]
    arr = cache.get(im["file_name"])
    anns = anns_by_image[image_id]
    x0, x1, y0, y1 = crop

    # annotations whose bbox intersects the crop, for an accurate subtitle count
    def bbox_intersects_crop(b):
        bx0, by0, bw, bh = b
        bx1, by1 = bx0 + bw, by0 + bh
        return not (bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1)

    anns_in_crop = [a for a in anns if bbox_intersects_crop(a["bbox"])]
    n_overlap_pairs = sum(
        1 for i in range(len(anns_in_crop)) for j in range(i + 1, len(anns_in_crop))
        if bbox_iou(anns_in_crop[i]["bbox"], anns_in_crop[j]["bbox"]) > 0
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    draw_annotations(ax, anns, mask_alpha=0.38, lw=1.2, spine_lw=1.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")

    handles = category_legend_handles(include_spine=True)
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=10)

    ax.set_title(
        f"Closely-adjacent / overlapping filament instances\n"
        f"{im['file_name']}  |  {len(anns_in_crop)} filaments in view, "
        f"{n_overlap_pairs} bbox-overlapping pair(s)  |  crop from full 2048x2048 image",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
def main():
    os.makedirs(ASSET_DIR, exist_ok=True)
    coco = load_coco()
    images_by_id, anns_by_image = build_indices(coco)
    cache = ImageCache()

    p1 = os.path.join(ASSET_DIR, f"{SLUG}_sample_grid.png")
    fig1 = plot_sample_grid(coco, images_by_id, anns_by_image, cache, save_path=p1)
    plt.close(fig1)
    print("saved", p1)

    p2 = os.path.join(ASSET_DIR, f"{SLUG}_barb_zoom.png")
    fig2 = plot_barb_zoom(coco, images_by_id, anns_by_image, cache, save_path=p2)
    plt.close(fig2)
    print("saved", p2)

    p3 = os.path.join(ASSET_DIR, f"{SLUG}_overlap_cluster.png")
    fig3 = plot_overlap_cluster(coco, images_by_id, anns_by_image, cache, save_path=p3)
    plt.close(fig3)
    print("saved", p3)

    # sanity check: re-derive the overlap example from scratch and confirm it's
    # near the top of the ranking (guards against the hardcoded id going stale
    # if the annotation file is ever regenerated/updated).
    top = find_top_overlap_images(coco, anns_by_image, top_n=10)
    ranked_ids = [t[2] for t in top]
    print("\nTop-10 overlap images (n_overlap_pairs, n_anns, image_id):")
    for t in top:
        print(" ", t)
    if OVERLAP_IMAGE_ID in ranked_ids:
        print(f"OK: {OVERLAP_IMAGE_ID} is rank {ranked_ids.index(OVERLAP_IMAGE_ID) + 1} by overlap-pair count")
    else:
        print(f"WARNING: {OVERLAP_IMAGE_ID} fell out of the top 10 -- re-check the hardcoded choice")


if __name__ == "__main__":
    main()
