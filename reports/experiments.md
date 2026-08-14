# Experiment log

Validation protocol: fixed grouped split (seed 0, val_fraction 0.15) — 175 val
entries / 106 unique base images, grouped by file_name so annotator-batch
duplicates never straddle the split. Metrics from `src/eval/metrics.py`
(PQ per Kirillov 2019, IoU>0.5 matching; Dice = per-image union-mask).

Reference points (measured on real data, see plan):
- Human-vs-human single-annotator agreement: PQ 0.346 (SQ 0.616, RQ 0.548)
- Majority-vote consensus vs single annotator: PQ 0.654 (SQ 0.797, RQ 0.812)
  → realistic ceiling for a consensus-predicting model
- CC-on-perfect-union-mask oracle: PQ 0.998 (instances never overlap)

| # | Experiment | Config | val PQ | SQ | RQ | Dice | TP/FP/FN | Decision |
|---|-----------|--------|--------|----|----|------|----------|----------|
| 1 | baseline_v1 | Mask R-CNN R50v2, 250/979 entries, 8 ep, 1024px, thr 0.5 | 0.3205 | 0.6127 | 0.5048 | 0.6165 | 850/1250/537 | FP-heavy; sweep threshold |
| 2 | baseline_v1 + thr sweep | same checkpoint, score_thresh 0.75 | 0.343 | — | — | — | — | +0.024 free; adopt sweep in all future evals |
| 3 | maskrcnn_full_v2 | full 979 entries, 16 ep, cosine 1e-4→1e-6, sweep 0.5–0.9 | *(running)* | | | | | Phase-0 bar for the U-Net fork |
| 4 | unet_v1 | SMP U-Net eff-b3, 1024² fg-biased crops, BCE+Dice, AMP, 40 ep, CC+min_area 200 | *(pending)* | | | | | Gate: beats #3 by +0.02 → U-Net primary |

## Notes

- 2026-08-14: #1 trained/evaluated end-to-end; submission baseline_v1.csv
  (2055 rows, 177/180 test images covered) verified format-clean.
- 2026-08-14: #2 threshold sweep measured on cached predictions (plan-phase
  analysis); 0.70–0.80 plateau, 0.75 best.
- Architecture pivot rationale: GT instances never overlap (0/6900 pairs) and
  CC extraction is lossless (PQ 0.998 oracle), so semantic quality is the whole
  game; duplicate annotator entries make a sigmoid U-Net a consensus estimator
  (ceiling PQ ~0.65 vs ~0.35 for annotator imitation).
