"""U-Net semantic-segmentation model for the filament consensus pipeline.

Why semantic + connected components instead of instance segmentation:
- GT filament instances essentially never overlap (0/6900 sampled pairs share
  a pixel; 0.04% touch within 2px), and 8-connected components on a perfect
  union mask reproduce the instance decomposition at PQ 0.998 -- instance
  separation is free here.
- Duplicate annotator entries make a sigmoid net a pixel-wise consensus
  estimator, which scores ~PQ 0.65 against any single annotator vs ~0.35 for
  one-annotator imitation (measured).

The model is a stock SMP U-Net; `classes` stays parameterized so a later
experiment can add a spine-auxiliary output channel (classes=2) without
touching this file's callers.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "efficientnet-b3",
    in_channels: int = 1,
    classes: int = 1,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """SMP U-Net with an ImageNet-pretrained encoder adapted to 1-channel input.

    SMP handles in_channels=1 by summing the pretrained stem's RGB kernel
    weights, which preserves the pretrained edge/texture filters' response to a
    grayscale input (equivalent to feeding a replicated 3-channel image).
    Output is raw logits (no activation) -- apply sigmoid at the call site.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


if __name__ == "__main__":
    import sys

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"efficientnet-b3 U-Net params: {n_params / 1e6:.1f}M")

    failures = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # training-shape forward/backward at the real crop size
    model.train()
    x = torch.rand(2, 1, 1024, 1024, device=device)
    y = model(x)
    check("train forward 1024^2 logits shape", tuple(y.shape) == (2, 1, 1024, 1024), f"{tuple(y.shape)}")
    loss = y.mean()
    loss.backward()
    check("backward populates grads", any(p.grad is not None for p in model.parameters()))

    # full-resolution inference forward (the deployment path)
    model.eval()
    with torch.no_grad():
        y = model(torch.rand(1, 1, 2048, 2048, device=device))
    check("eval forward 2048^2 shape", tuple(y.shape) == (1, 1, 2048, 2048), f"{tuple(y.shape)}")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak CUDA memory this smoke test: {peak_gb:.2f} GB")

    print("\nSMOKE TEST:", "FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)
