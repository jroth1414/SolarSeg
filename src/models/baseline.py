"""Baseline Mask R-CNN model for the MAGFiLO solar-filament instance-segmentation task.

Architecture choices (documented per task instructions):

1. Backbone / weights: torchvision's ``maskrcnn_resnet50_fpn_v2`` (ResNet-50-FPN with
   the "v2" training recipe -- improved RPN/head design vs the original
   ``maskrcnn_resnet50_fpn``), initialized from COCO-pretrained weights
   (``MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT``) so the backbone/FPN/RPN start from
   features that already know how to detect "blob-ish things on a background" rather
   than from scratch. This is a reasonable general-purpose baseline detector; nothing
   here is filament-specific.

2. Single-channel grayscale input vs 3-channel-pretrained backbone: our images are
   single-channel (mode 'L', see FilamentDataset). The pretrained ResNet-50 stem
   expects 3-channel RGB-normalized input. Per the task's explicit guidance we take the
   simpler of the two documented options -- REPLICATE the single channel to 3 channels
   at input time (image.repeat(3, 1, 1)) rather than surgically modifying conv1 -- so
   the entire pretrained backbone (including conv1's learned RGB-edge filters, which
   still fire fine on a replicated grayscale image) stays intact. This happens in
   `GrayscaleTo3Channel`, wired in as a submodule so it travels with the model
   (state_dict save/load, .to(device), etc.) rather than living as loose glue code in
   the training script.

3. Heads: `box_predictor` (FastRCNNPredictor) and `mask_predictor` (MaskRCNNPredictor)
   are replaced for `num_classes=2` (background=0, filament=1), per the task's
   single-class modeling decision (category_id in the source COCO data is ignored --
   see src/data/coco_dataset.py docstring). The replaced heads are randomly initialized
   (torchvision's standard predictor init); every other weight (backbone/FPN/RPN/box
   head trunk/mask head trunk) is loaded from COCO pretraining.

4. Resolution: the model's internal `GeneralizedRCNNTransform` is reconfigured to
   min_size=800, max_size=1024 (down from the COCO-pretrained default min_size=800,
   max_size=1333) -- both to keep VRAM/step-time bounded for this smoke-test-scale
   baseline (per task: "reduced resolution ... full 2048x2048 can come later") and
   because torchvision only accepts one resize policy per model, so min_size/max_size
   have to be set at construction time here rather than patched later.
   Native images are 2048x2048; at max_size=1024 they get downsampled to 1024x1024
   before the backbone, so predicted masks come back at 1024x1024 and must be resized
   back to 2048x2048 by the caller when comparing against full-resolution GT (torchvision
   handles this automatically for its own outputs during eval -- see note in
   train_baseline.py -- but be aware of it if you touch this code).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class GrayscaleTo3Channel(nn.Module):
    """Replicates a single-channel [.., 1, H, W] input to 3 channels.

    Registered as a real submodule (not a free function) so it is captured by
    model.state_dict() / model.to(device) / torch.save the same way the rest of the
    model is, and so `build_model` returns one self-contained nn.Module that a caller
    can feed [1,H,W]-or-[B,1,H,W]-shaped grayscale tensors directly.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3] == 3:
            return x  # already 3-channel; no-op (defensive, shouldn't normally hit)
        if x.shape[-3] != 1:
            raise ValueError(f"expected 1-channel input, got shape {tuple(x.shape)}")
        return x.repeat_interleave(3, dim=-3)


class FilamentMaskRCNN(nn.Module):
    """Wraps a torchvision MaskRCNN with a grayscale->3ch adapter in front.

    Forward signature mirrors torchvision detection models exactly:
      - training:   forward(images: List[Tensor[1,H,W]], targets: List[dict]) -> loss dict
      - inference:  forward(images: List[Tensor[1,H,W]]) -> List[dict] (boxes/labels/scores/masks)
    where `images` is a list of single-channel [1,H,W] float tensors in [0,1] (exactly
    what FilamentDataset.__getitem__ / collate_fn produce) -- the 3-channel replication
    happens internally, per-image, before torchvision's own GeneralizedRCNNTransform
    (which does the resizing/normalizing/batching).
    """

    def __init__(self, maskrcnn: nn.Module):
        super().__init__()
        self.to_3ch = GrayscaleTo3Channel()
        self.maskrcnn = maskrcnn

    def forward(self, images, targets=None):
        images = [self.to_3ch(img) for img in images]
        if targets is not None:
            return self.maskrcnn(images, targets)
        return self.maskrcnn(images)


def build_model(
    num_classes: int = 2,
    min_size: int = 800,
    max_size: int = 1024,
    pretrained: bool = True,
) -> FilamentMaskRCNN:
    """Build the baseline Mask R-CNN.

    Args:
        num_classes: background + foreground classes. Default 2 (background, filament)
            per this task's single-class modeling decision -- do NOT pass 5 for the
            4 raw category_id values, they are not used for modeling (see module docstring
            and src/data/coco_dataset.py).
        min_size / max_size: torchvision GeneralizedRCNNTransform resize bounds. Defaults
            here (800/1024) are the reduced-resolution baseline-v1 scale requested by the
            task; pass e.g. min_size=1024-2048, max_size=2048 for a later full-resolution run
            (note: VRAM/step-time will grow substantially -- reduce batch_size accordingly).
        pretrained: use COCO-pretrained weights for everything except the replaced heads.
            Set False for a from-scratch ablation.

    Returns:
        FilamentMaskRCNN, an nn.Module ready for
        `model(images, targets)` (train, returns loss dict) or
        `model(images)` (eval, returns per-image prediction dicts).
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    maskrcnn = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,  # `weights` (full-model COCO weights) already covers the backbone;
        min_size=min_size,
        max_size=max_size,
    )

    # -- replace box head classifier/regressor for num_classes --
    in_features_box = maskrcnn.roi_heads.box_predictor.cls_score.in_features
    maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # -- replace mask head predictor for num_classes --
    in_features_mask = maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # matches the pretrained mask head's hidden dim
    maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return FilamentMaskRCNN(maskrcnn)


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(num_classes=2, min_size=512, max_size=1024, pretrained=True).to(device)
    check("box_predictor has num_classes=2 outputs",
          model.maskrcnn.roi_heads.box_predictor.cls_score.out_features == 2)
    check("mask_predictor has num_classes=2 outputs",
          model.maskrcnn.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2)

    # ---- training-mode forward: fake 1-channel images + targets ----
    model.train()
    imgs = [torch.rand(1, 600, 800, device=device), torch.rand(1, 500, 500, device=device)]
    targets = []
    for img in imgs:
        h, w = img.shape[-2:]
        boxes = torch.tensor([[10.0, 10.0, 100.0, 120.0], [50.0, 60.0, 200.0, 220.0]], device=device)
        labels = torch.ones((2,), dtype=torch.int64, device=device)
        masks = torch.zeros((2, h, w), dtype=torch.uint8, device=device)
        masks[0, 10:120, 10:100] = 1
        masks[1, 60:220, 50:200] = 1
        targets.append({"boxes": boxes, "labels": labels, "masks": masks})

    loss_dict = model(imgs, targets)
    check("train-mode forward returns a dict of losses", isinstance(loss_dict, dict))
    expected_loss_keys = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    check("loss dict has the expected keys", expected_loss_keys.issubset(loss_dict.keys()),
          f"got {sorted(loss_dict.keys())}")
    total_loss = sum(loss_dict.values())
    check("total loss is finite and > 0", torch.isfinite(total_loss) and total_loss.item() > 0,
          f"total_loss={total_loss.item()}")

    total_loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    check("backward() populates finite gradients", has_grad)

    # ---- eval-mode forward: predictions ----
    model.eval()
    with torch.no_grad():
        preds = model(imgs)
    check("eval-mode forward returns one dict per image", isinstance(preds, list) and len(preds) == len(imgs))
    for i, p in enumerate(preds):
        check(f"[{i}] prediction dict has boxes/labels/scores/masks",
              all(k in p for k in ("boxes", "labels", "scores", "masks")))

    # ---- grayscale-to-3ch adapter sanity ----
    g = torch.rand(1, 40, 40)
    g3 = model.to_3ch(g)
    check("GrayscaleTo3Channel produces 3 channels", tuple(g3.shape) == (3, 40, 40))
    check("GrayscaleTo3Channel channels are identical copies",
          bool(torch.equal(g3[0], g3[1]) and torch.equal(g3[1], g3[2])))

    print()
    if failures:
        print(f"SMOKE TEST: FAIL ({len(failures)} check(s) failed)")
        sys.exit(1)
    else:
        print("SMOKE TEST: PASS")
