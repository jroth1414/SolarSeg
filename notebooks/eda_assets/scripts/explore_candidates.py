"""One-off exploration to pick good candidate images/annotations for the
illustrative figures. Not part of the reusable output -- just used to choose
which image_ids / annotation ids to hardcode in viz_illustrative_figures.py.
"""
import json
import collections

DATA_ROOT = "D:/SolarSeg/data/MAGFiLO_1.0_Kaggle_2026"
ANN_PATH = f"{DATA_ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

with open(ANN_PATH) as f:
    coco = json.load(f)

images_by_id = {im["id"]: im for im in coco["images"]}
anns_by_image = collections.defaultdict(list)
for a in coco["annotations"]:
    anns_by_image[a["image_id"]].append(a)

# 1. Observatory code / date spread for the grid image.
codes = collections.Counter(im["file_name"][-6:-5] + im["file_name"][-5:-4] for im in coco["images"])
print("observatory code counts (2-letter suffix before .jpeg):")
# actually the code is the last 2 chars before extension, e.g. 'Bh'
code_counter = collections.Counter(im["file_name"][:-5][-2:] for im in coco["images"])
print(code_counter.most_common(20))

# 2. Candidates for barb figure: high polygon point count, but not too huge a bbox
# (want a "tight zoom" that's still legible), and prefer ones with a populated spine.
cands = []
for a in coco["annotations"]:
    seg = a["segmentation"][0]
    npts = len(seg) // 2
    bbox = a["bbox"]
    w, h = bbox[2], bbox[3]
    has_spine = "spine" in a and a["spine"] is not None and len(a["spine"]) > 0
    cands.append((npts, w, h, w * h, has_spine, a["id"], a["image_id"], a["category_id"]))

cands.sort(key=lambda t: -t[0])
print("\nTop 25 by polygon point count:")
for c in cands[:25]:
    print(c)

# Look for a sweet spot: high point count, moderate bbox size (say area between
# 3000 and 40000 px^2, i.e. not a huge sprawling filament that won't show barbs
# at a legible zoom, not a tiny blob either), spine present.
print("\nFiltered candidates (npts>=120, 2500<=area<=60000, has_spine):")
filtered = [c for c in cands if c[0] >= 120 and 2500 <= c[3] <= 60000 and c[4]]
filtered.sort(key=lambda t: -t[0])
for c in filtered[:20]:
    print(c)

# 3. Candidates for overlap figure: images with many annotations whose bboxes
# are close/overlapping.
def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def bbox_dist(b1, b2):
    # center distance
    cx1, cy1 = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    cx2, cy2 = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

overlap_scores = []
for image_id, anns in anns_by_image.items():
    if len(anns) < 3:
        continue
    n_overlap_pairs = 0
    close_pairs = 0
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            iou = bbox_iou(anns[i]["bbox"], anns[j]["bbox"])
            if iou > 0.0:
                n_overlap_pairs += 1
            d = bbox_dist(anns[i]["bbox"], anns[j]["bbox"])
            maxdim = max(anns[i]["bbox"][2], anns[i]["bbox"][3], anns[j]["bbox"][2], anns[j]["bbox"][3])
            if d < maxdim * 1.5:
                close_pairs += 1
    overlap_scores.append((n_overlap_pairs, close_pairs, len(anns), image_id))

overlap_scores.sort(key=lambda t: (-t[0], -t[1]))
print("\nTop 20 images by bbox-overlap pair count:")
for s in overlap_scores[:20]:
    print(s)
