"""Probability-map -> filament instances post-processing.

Single source of truth for the semantic->instance step so training-time
validation, the offline post-processing sweep, and submission generation all
score/emit exactly the same masks.

Pipeline: threshold the sigmoid probability map -> 8-connected components ->
drop components below min_area (GT area floor is ~209px at p1, so 150-250 is
the sensible range) -> optional per-component mean-probability filter.

Non-empty guard: every train image has >=1 filament and an image with zero
predictions scores PQ=0 with certainty, so when nothing survives filtering we
emit the largest component found before filtering (if any exists at all).
"""

from __future__ import annotations

import cv2
import numpy as np


def probs_to_instances(
    prob: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 200,
    min_mean_prob: float = 0.0,
    closing: bool = False,
    non_empty_guard: bool = True,
    grow_threshold: float | None = None,
) -> list[np.ndarray]:
    """Convert one HxW float probability map into a list of HxW uint8 instance masks.

    Args:
        prob: HxW float array in [0,1].
        threshold: binarization threshold on prob.
        min_area: drop components smaller than this many pixels.
        min_mean_prob: additionally drop components whose mean probability
            (within the component) is below this. 0 disables the filter.
        closing: apply a 3x3 morphological closing before component extraction
            (seals 1-2px gaps; measured merge risk at that radius ~0.04% of pairs).
        non_empty_guard: emit the largest raw component when filters kill everything.
        grow_threshold: hysteresis growing -- extend each detected component into
            the connected region above this LOWER threshold. Components stay
            separated: a low-threshold region touching two or more detected
            components leaves them ungrown (no-merge guard). Targets the
            "found it but drew it too small" error bucket (near-miss IoU 0.4-0.5)
            without creating new components. None disables.
    """
    binary = (prob > threshold).astype(np.uint8)
    if closing:
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if grow_threshold is not None and grow_threshold < threshold:
        labels = _hysteresis_grow(prob, labels, n_labels, grow_threshold)

    instances = []
    largest_label, largest_area = None, 0
    for label in range(1, n_labels):  # 0 is background
        mask = (labels == label).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        if area > largest_area:
            largest_label, largest_area = label, area
        if area < min_area:
            continue
        if min_mean_prob > 0.0 and float(prob[mask.astype(bool)].mean()) < min_mean_prob:
            continue
        instances.append(mask)

    if not instances and non_empty_guard:
        if largest_label is not None:
            instances.append((labels == largest_label).astype(np.uint8))
        else:
            # Nothing crossed `threshold` anywhere in the image. Progressively
            # lower the threshold until some component appears and emit the
            # largest one: an image with zero rows is a guaranteed PQ=0 (every
            # train image has >=1 filament) and the public scorer's handling of
            # absent images is unknown -- one best-guess mask strictly
            # dominates none.
            for fallback_thr in (0.4, 0.3, 0.2, 0.1, 0.05):
                if fallback_thr >= threshold:
                    continue
                fb = (prob > fallback_thr).astype(np.uint8)
                n_fb, fb_labels, fb_stats, _ = cv2.connectedComponentsWithStats(fb, connectivity=8)
                if n_fb > 1:
                    biggest = 1 + int(np.argmax(fb_stats[1:, cv2.CC_STAT_AREA]))
                    instances.append((fb_labels == biggest).astype(np.uint8))
                    break

    return instances


def _hysteresis_grow(prob, seed_labels, n_seed_labels, grow_threshold):
    """Extend each seed component into its connected low-threshold region.

    A low-threshold component containing pixels of exactly ONE seed label grows
    that seed to the full low component; one containing two or more distinct
    seed labels leaves them all unchanged (no-merge guard).
    Returns a new labels array using the original seed label ids.
    """
    low_binary = (prob > grow_threshold).astype(np.uint8)
    n_low, low_labels = cv2.connectedComponents(low_binary, connectivity=8)

    out = seed_labels.copy()
    for low_label in range(1, n_low):
        low_mask = low_labels == low_label
        seeds_here = np.unique(seed_labels[low_mask])
        seeds_here = seeds_here[seeds_here != 0]
        if len(seeds_here) == 1:
            out[low_mask] = seeds_here[0]
    return out
