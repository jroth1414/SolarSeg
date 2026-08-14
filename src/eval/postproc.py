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
    """
    binary = (prob > threshold).astype(np.uint8)
    if closing:
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    instances = []
    largest_label, largest_area = None, 0
    for label in range(1, n_labels):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_label, largest_area = label, area
        if area < min_area:
            continue
        mask = (labels == label).astype(np.uint8)
        if min_mean_prob > 0.0 and float(prob[mask.astype(bool)].mean()) < min_mean_prob:
            continue
        instances.append(mask)

    if not instances and non_empty_guard and largest_label is not None:
        instances.append((labels == largest_label).astype(np.uint8))

    return instances
