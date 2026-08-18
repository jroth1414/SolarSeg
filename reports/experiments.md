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
| 5c | unet_v2 + pp sweep | thr 0.55, min_area 400, closing=True (trimmed 40-config grid) | **0.4095** | — | — | — | — | **Clears fork gate (0.4069) → U-Net is primary architecture.** First submission candidate |
| 6 | unet_b5 | eff-b5 encoder from scratch, 40 ep (encoder ladder) | 0.3908 | 0.6509 | 0.5782 | 0.6810 | 948/901/439 | **Refuted**: below b3 @ matched budget (0.3966). Capacity isn't the constraint — ladder stopped |
| 7 | unet_v2 + flip-TTA | 4× flip-averaged probs, pp thr 0.55/400/closing | 0.4193 | 0.6499 | 0.6206 | 0.6788 | 875/551/512 | +0.010; FP 782→551. Keep TTA permanently (~100s val cost) |
| 8 | unet_consensus | soft mean-of-batches targets, 601 unique imgs, 60 ep | 0.3994 | 0.6535 | 0.5919 | 0.6814 | 939/850/448 | Tie with per-entry (0.4015) — explicit soft target adds nothing over SGD averaging. Keep per-entry |
| 8b | unet_consensus + TTA | 4-flip, pp 0.55/400/closing | 0.4101 | 0.6502 | 0.6061 | 0.6840 | 883/639/504 | Weaker than v2+TTA (0.4193) |
| 8c | 2-model ensemble | mean(v2_tta, consensus_tta) maps, thr swept | 0.4139 | — | — | — | — | **Refuted**: weaker partner dilutes stronger model. Ensemble equals only → folds |
| 9 | TTA-maps pp re-sweep | 40-config grid on flip-averaged maps | 0.4193 | — | — | — | — | Optimum unchanged (thr 0.55 / 400 / closing) — pp config is stable |
| 10 | hysteresis growing | seed thr 0.55, grow ∈ {0.25…0.50}, no-merge guard | 0.4188 @ best | — | — | — | — | **Refuted**: monotonically worse; growing degrades the 876 TPs more than it recovers the 99 near-misses. Post-processing maxed out |

| 11 | 5-fold CV (unet_f0..f4) | 60 ep each, grouped kfold seed 0; per-fold TTA PQ 0.4026–0.4306 | — | — | — | — | — | Fold spread ±0.015 calibrates val noise floor (~0.01) |
| 12 | OOF stitched (TTA + tuned pp) | all 1154 entries, each scored by its held-out fold | **0.4221** | 0.6553 | 0.6206 | — | 5201/3285/2998 | Robust recipe estimate; test-time 5-model ensemble expected ≥ this |

Final submission candidates: `unet_5fold_tta.csv` (ensemble, primary) and `unet_v2_tta.csv` (single model, 0.4193 fixed-split val).

| 13 | unet_1536 | 1536² crops, bs2+accum2, 60 ep, naive pp | 0.4169 | 0.6491 | 0.6192 | 0.6806 | 906/619/481 | **Gate cleared** (+0.015 vs 1024 naive): SQ flat, RQ +0.023 — resolution helps *detection* of thin filaments. New recipe line |
| 13b | unet_1536 + TTA @ v2's pp | thr 0.55/400/closing | 0.4144 | 0.6491 | 0.6105 | 0.6786 | 829/479/558 | Below naive! pp operating point is model-specific — re-sweep per model (running) |
| 14 | unet_mitb2_1536 | SegFormer-family encoder @ 1536 recipe | — | — | — | — | — | **OOM on 16GB** (attention wants 29GB @ 1536; 2048 inference would need tiling). Family excluded |
| 14b | unet_1536 + TTA + own pp sweep | thr **0.40** / min_area **300** / closing | **0.4234** | — | — | — | — | New best single model (+0.004 paired vs v2's 0.4193). pp optimum moved 0.55→0.40 vs the 1024 model — re-sweep per model is mandatory |
| 15 | unet_convnext_1536 | tu-convnext_small @ 1536 recipe (encoder family pivot) | 0.3984 | 0.6479 | 0.5913 | 0.6893 | 902/805/485 | **Refuted** (−0.019 vs b3@1536): with b5, confirms encoder isn't the constraint. No multi-arch ensemble |
| 16 | 5-fold @ 1536 (unet1536_f0..f4) | 45 ep/fold; per-fold TTA PQ 0.4008–0.4275; OOF swept best 0.4200 @ 0.55/400 | 0.4200 | — | — | — | — | **OOF tie with 1024 line (0.4221)** — fixed-split resolution gain didn't replicate across folds |
| 17 | 2-line OOF ensemble | mean(1024-fold, 1536-fold) per entry, same partition | 0.4233 | — | — | — | — | +0.001 over 1024 line alone — no submission (< 0.01 discipline). **Recipe family exhausted**; remaining levers: spine-aux, pseudo-labeling, domain SSL reserve |
| 18 | unet_spineaux(+2) | 2-ch head, spine BCE weight 0.4, ~75 effective ep | 0.4011 | 0.6505 | 0.5935 | 0.6762 | 931/800/456 | **Refuted both gates**: PQ tie with 0.4015 line; fragmentation unchanged (40 vs 38), hallucinations slightly worse (310 vs 268). Spine channel adds nothing the mask target doesn't already teach |
| 19 | unet_pseudo | +180 test imgs w/ 10-model×4-flip soft pseudo-labels, 60 ep | 0.4145 | 0.6561 | 0.6108 | 0.6883 | 948/768/439 | **GATE CLEARED** (+0.013 naive vs 0.4015; best SQ/Dice yet; still rising @ ep60 → extending). TTA: 0.4207 pre-sweep |
| 19b | unet_pseudo + TTA + sweep | thr 0.45 / min_area 300 / closing | 0.4224 | — | — | — | — | +0.003 over plain line post-TTA (naive gain partly absorbed by TTA+sweep); ≈ single-1536 (0.4234) |
| 19c | unet_pseudo2 (extension) | +30 ep warm-restart | 0.4135 | — | — | — | — | Did not beat ep-60 original (0.4145) — pseudo line peaked; original checkpoint stands |
| 20 | 5-fold pseudo rebuild (unetps_f0..f4) | 60 ep/fold with pseudo-labels; per-fold TTA 0.4011–0.4261 | 0.4216 | — | — | — | — | OOF-swept: tie with plain line (0.4221) — per-fold gains didn't survive aggregation |
| 21 | 3-line OOF ensemble | mean over {1024, 1536, pseudo} fold maps | **0.4246** | — | — | — | — | Best OOF of the program. Submitted as unet_15model_tta.csv (1299 rows; 15 models × 4 flips, 613s test inference) |

**Program convergence:** every lever past TTA landed within ±0.005 at ensemble level. Local ceiling with this approach ≈ 0.42–0.425 OOF. Remaining strategic option: domain-SSL pretraining on external GONG archive. Efficiency note for report: single-model+TTA (~1 min test inference, PQ 0.4224) vs 15-model (~10 min, 0.4246).

Also: progressive non-empty guard added to postproc (threshold falls back 0.4→0.05 until a component exists; images emitting zero rows are guaranteed PQ=0 and scorer handling of absent images is unknown).

### Leaderboard calibration (2026-08-16, first uploads)

Both `unet_5fold_tta.csv` (OOF 0.4221) and `unet_v2_tta.csv` (val 0.4193) scored **0.36 public LB** (2-decimal display).
- Systematic offset: LB ≈ local val − 0.06; ordering preserved → keep optimizing val PQ.
- 2-decimal LB resolution makes small-delta submission probing blind; only submit val Δ ≥ 0.01.
- Standings context: leader 0.52 (outlier), then 0.40, dense pack 0.37–0.39; we're just behind the pack.

### Error taxonomy (unet_v2 + TTA, thr 0.55/400/closing) — feeds report rubric

FN side (511): pure_miss 268 (52%), partial_undercover 108 (21%), near_miss 99 (19%, mean best-IoU 0.454), fragmented 36 (7%).
FP side (551): partial_overcover 284 (52%), hallucinated 253 (46%), merged_pred 14 (3%).
Conclusions: watershed splitting pointless (3%); fragment-merging marginal (7%); the recoverable mass is boundary-extent calibration, which uniform growing can't fix (see #10) → remaining gains come from better probability maps (consensus targets, folds/ensembling).

## Notes

- 2026-08-14: #1 trained/evaluated end-to-end; submission baseline_v1.csv
  (2055 rows, 177/180 test images covered) verified format-clean.
- 2026-08-14: #2 threshold sweep measured on cached predictions (plan-phase
  analysis); 0.70–0.80 plateau, 0.75 best.
- Architecture pivot rationale: GT instances never overlap (0/6900 pairs) and
  CC extraction is lossless (PQ 0.998 oracle), so semantic quality is the whole
  game; duplicate annotator entries make a sigmoid U-Net a consensus estimator
  (ceiling PQ ~0.65 vs ~0.35 for annotator imitation).
