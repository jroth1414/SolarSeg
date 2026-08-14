# Data download

Not tracked in git (750.97 MB, 888 files, CC BY-NC 4.0).

Layout once downloaded:

```
data/MAGFiLO_1.0_Kaggle_2026/
  train/
    train_images/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/
    test_images/
```

Requires accepting the competition rules on Kaggle first (`Join Competition`).

## Option A: script (used to fetch this copy)

Put a Kaggle API token in `.env` at the repo root (`KAGGLE_API_TOKEN=...`, gitignored),
then:

```bash
python src/data/download.py
```

This uses `kagglehub.competition_download`, which caches the raw download under
`~/.cache/kagglehub/competitions/filament-segmentation-2026/`, and copies it here.

## Option B: manual

1. Go to https://www.kaggle.com/competitions/filament-segmentation-2026/data
2. "Download All" -> "Download as zip"
3. Unzip into this folder to match the layout above.
