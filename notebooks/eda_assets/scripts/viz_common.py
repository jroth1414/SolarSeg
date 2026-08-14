"""Reusable plotting/data-loading functions for the MAGFiLO illustrative
figures. Import this module from notebook cells or other scripts to
regenerate any of the three figures live.

Design notes
------------
- Masks are drawn as filled matplotlib Polygon patches directly from the
  COCO `segmentation` polygon coordinates (semi-transparent fill + solid
  edge), NOT rasterized through pycocotools. This is a deliberate choice:
  it renders exactly what the annotation *is* (a single closed polygon per
  filament) at full float precision, with no rasterization/anti-aliasing
  choices obscuring the shape -- appropriate for an illustrative EDA figure.
  (The eventual submission pipeline does need the polygon->RLE->mask
  round trip; that is out of scope for this slice.)
- Category colors and the spine style are centralized here so all three
  figures (and any notebook cell that reuses this module) are visually
  consistent.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

DATA_ROOT = "D:/SolarSeg/data/MAGFiLO_1.0_Kaggle_2026"
ANN_PATH = f"{DATA_ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = f"{DATA_ROOT}/train/train_images"

ASSET_DIR = "D:/SolarSeg/notebooks/eda_assets"
SLUG = "viz"

# category_id -> (name, color). id 4 ("Ambiguous") is defined in the schema
# but does not occur in the train annotations (see warnings) -- kept here
# so the legend/color mapping is still schema-correct if it ever appears.
CATEGORY_STYLE = {
    1: ("Left", "#1f77b4"),           # blue
    2: ("Right", "#ff7f0e"),          # orange
    3: ("Unidentifiable", "#2ca02c"), # green
    4: ("Ambiguous", "#d62728"),      # red
}
SPINE_COLOR = "#ffff00"  # bright yellow, contrasts with all 4 mask colors + gray background
MASK_ALPHA = 0.38


def load_coco(ann_path=ANN_PATH):
    with open(ann_path) as f:
        coco = json.load(f)
    return coco


def build_indices(coco):
    """Returns (images_by_id, anns_by_image) where images_by_id maps the
    COCO `images[].id` string to the image dict, and anns_by_image maps that
    same id to a list of its annotation dicts."""
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    return images_by_id, anns_by_image


class ImageCache:
    """Loads each JPEG from disk at most once, reused across all
    annotations/figures that reference it."""

    def __init__(self, img_dir=IMG_DIR):
        self.img_dir = img_dir
        self._cache = {}

    def get(self, file_name):
        if file_name not in self._cache:
            path = os.path.join(self.img_dir, file_name)
            with Image.open(path) as im:
                arr = np.array(im.convert("L"))
            self._cache[file_name] = arr
        return self._cache[file_name]


def draw_annotations(ax, anns, mask_alpha=MASK_ALPHA, draw_spine=True,
                      lw=1.0, spine_lw=1.4):
    """Draws filled+outlined polygons for each annotation's segmentation,
    colored by category_id, plus its spine polyline on top, onto `ax`.
    Assumes `ax` already has the base image shown with imshow (extent in
    pixel coordinates, origin upper-left, matching COCO polygon coords)."""
    for a in anns:
        cat_name, color = CATEGORY_STYLE.get(a["category_id"], ("Unknown", "#999999"))
        seg = a["segmentation"][0]
        pts = np.array(seg, dtype=float).reshape(-1, 2)
        poly = MplPolygon(pts, closed=True, facecolor=color, edgecolor=color,
                           alpha=mask_alpha, linewidth=lw, joinstyle="round")
        ax.add_patch(poly)
        # redraw a fully-opaque thin edge on top so boundaries stay crisp
        # even where masks overlap / alpha stacks
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=0.9)
        if draw_spine and a.get("spine"):
            sp = np.array(a["spine"], dtype=float).reshape(-1, 2)
            ax.plot(sp[:, 0], sp[:, 1], color=SPINE_COLOR, linewidth=spine_lw,
                     linestyle="--", solid_capstyle="round", dash_capstyle="round",
                     zorder=5)


def category_legend_handles(include_spine=True):
    handles = []
    for cid in sorted(CATEGORY_STYLE):
        name, color = CATEGORY_STYLE[cid]
        handles.append(Line2D([0], [0], marker="s", linestyle="none", markersize=10,
                               markerfacecolor=color, markeredgecolor=color,
                               alpha=0.85, label=f"{name} (id={cid})"))
    if include_spine:
        handles.append(Line2D([0], [0], color=SPINE_COLOR, linewidth=2, linestyle="--",
                               label="spine (centerline)"))
    return handles


def polygon_point_counts(coco):
    """Returns list of (n_points, annotation) for every annotation, sorted
    descending by point count -- used as the 'structural complexity' proxy
    for picking barb-showing candidates."""
    out = []
    for a in coco["annotations"]:
        n = len(a["segmentation"][0]) // 2
        out.append((n, a))
    out.sort(key=lambda t: -t[0])
    return out
