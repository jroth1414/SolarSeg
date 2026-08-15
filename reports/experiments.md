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
| 3 | maskrcnn_full_v2 | full 979 entries, 16 ep, cosine 1e-4→1e-6, best thr 0.80 | 0.3869 | 0.6279 | 0.5902 | 0.6457 | 835/598/552 | Phase-0 bar set: U-Net gate = 0.407. Loss 0.96→0.53; sweep plateau 0.75–0.85 |
| 4 | unet_v1 | SMP U-Net eff-b3, 1024² fg-biased crops, BCE+Dice, AMP, 40 ep, naive pp (0.5/200) | 0.3966 | 0.6507 | 0.5870 | 0.6798 | 938/856/449 | Beats #3 pre-sweep; val PQ still rising at ep40 → extend |
| 5 | unet_v2 | warm-restart from #4, +40 ep @ 1e-4 cosine, naive pp | 0.4015 | 0.6491 | 0.5963 | 0.6754 | 928/782/459 | Best @ ep25, then flat — length lever exhausted. Sweep pending |
| 5b | unet_v1 + pp sweep | thr 0.50, min_area 400, closing=True (294-config grid) | 0.4065 | — | — | — | — | +0.010 over naive; min_mean_prob axis dead; min_area 400 ≫ GT floor 200 → small CCs are FPs |
| 6 | unet_b5 | eff-b5 encoder from scratch, 40 ep (encoder ladder) | *(running)* | | | | | Gate: +0.01 over b3 line |

## Notes

- 2026-08-14: #1 trained/evaluated end-to-end; submission baseline_v1.csv
  (2055 rows, 177/180 test images covered) verified format-clean.
- 2026-08-14: #2 threshold sweep measured on cached predictions (plan-phase
  analysis); 0.70–0.80 plateau, 0.75 best.
- Architecture pivot rationale: GT instances never overlap (0/6900 pairs) and
  CC extraction is lossless (PQ 0.998 oracle), so semantic quality is the whole
  game; duplicate annotator entries make a sigmoid U-Net a consensus estimator
  (ceiling PQ ~0.65 vs ~0.35 for annotator imitation).
