# Solar Filament Segmentation Challenge 2026

Kaggle competition: https://www.kaggle.com/competitions/filament-segmentation-2026
Sponsor: NSF National Solar Observatory (NSO). Runs as an IEEE BigData 2026 BigDataCup.

## Task

Instance segmentation of solar filaments in 2048x2048 grayscale H-Alpha images from
GONG observatories (converted from FITS to 8-bit JPEG). Each image can contain multiple
filaments; each filament gets its own predicted mask (not semantic/binary segmentation —
this is instance-level).

## Data (`data/`, gitignored — download manually, see `data/README.md`)

```
MAGFiLO_1.0_Kaggle_2026/
  train/
    train_images/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json   # COCO-style
  test/
    test_images/
```

- Filenames encode capture time + instrument: `YYYYMMDDHHMMSSII.jpeg`
  (e.g. `20260901165702Bh.jpeg` = Big Bear Observatory).
- Annotations are COCO-format JSON. `annotation.segmentation` is a single closed
  polygon per filament (not RLE at rest — RLE is only used in the submission file).
  `annotation.spine` is a polyline through the filament's centerline. `bbox` is
  `[x, y, w, h]` in pixels. `category_id`: 1=Left, 2=Right, 3=Unidentifiable, 4=Ambiguous.
- The same source image may appear multiple times under different `image_id` prefixes
  (independent annotator batches, e.g. `010101-...` vs `010102-...`) — treat these as
  distinct images, do not attempt to merge/dedupe annotators.
- Use `pycocotools` for polygon<->RLE<->mask conversions (`annToMask`, `encodeMask`,
  `decodeMask`).

## Evaluation

- Primary metric: **Panoptic Quality (PQ)** — matches predicted/GT segments by IoU,
  penalizes fragmentation (one-to-many) and over-merging (many-to-one). Also tracked:
  mean Dice (`torchmetrics.segmentation.DiceScore`), IoU distribution.
- Final judging (70% quant / 30% qual) also weighs pipeline writeup, code quality, and
  visual plausibility of masks on the H-Alpha images — this isn't a pure leaderboard
  competition, so keep the code readable and the pipeline well-documented.
- Constraint: inference may only use the test-directory H-alpha images (no other
  ground-truth metadata at inference time). External data/pretrained models are allowed
  if publicly available to all participants.
- Public leaderboard score is computed on ~50% of the test set only.

## Submission format

Single CSV, one row per predicted filament instance:

```
filament_id,segmentation_rle
20150125172714Mh_1,"<rle counts>"
20150125172714Mh_2,"<rle counts>"
```

- `filament_id` = `<image_id>_<n>`, n just needs to make rows unique per image.
- `segmentation_rle` = RLE **counts only** (no size — fixed 2048x2048), no quotes in
  the actual field content. Encode with `pycocotools.mask.encode`.
- Max 5 submissions/day, up to 2 selected as final. Final judging additionally requires
  a public code repo (with `requirements.txt` + an end-to-end notebook) and a 4-page
  report, submitted via a separate Google Form — not just a Kaggle CSV submission.

## Self-evaluation

Organizers published a self-eval notebook (Dice/PQ scoring + plots):
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

## Conventions for this repo

- `src/data/` — dataset loading, COCO parsing, RLE/polygon/mask conversion helpers.
- `src/models/` — model definitions / training code.
- `src/eval/` — Dice + Panoptic Quality scoring, matching predicted-vs-GT segments.
- `notebooks/` — exploration and the eventual end-to-end pipeline notebook required
  for final submission.
- `submissions/` — generated leaderboard CSVs (gitignored; keep the *scripts* that
  produced them, not the CSVs themselves, if the repo will be the public final-submission repo).
- `reports/` — the 4-page technical report draft.
- Images are single-channel grayscale — do not load as 3-channel RGB.
