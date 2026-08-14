# Solar Filament Segmentation Challenge 2026

Working repo for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(NSF National Solar Observatory / IEEE BigData 2026 BigDataCup).

Instance segmentation of solar filaments in GONG H-Alpha solar images, scored on
Panoptic Quality and Dice score. See [`CLAUDE.md`](CLAUDE.md) for the full task,
data, and evaluation writeup used to brief Claude Code on this project.

## Setup

```bash
pip install -r requirements.txt
```

Then download the competition data — see [`data/README.md`](data/README.md).

## Layout

- `src/data/` — dataset loading, COCO parsing, RLE/polygon/mask conversion
- `src/models/` — model definitions and training
- `src/eval/` — Dice + Panoptic Quality scoring
- `notebooks/` — exploration + the end-to-end pipeline notebook
- `submissions/` — generated leaderboard CSVs
- `reports/` — 4-page technical report (final submission)
