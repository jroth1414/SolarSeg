"""U-Net v1 training run (Phase 1 of the experiment roadmap).

Trains an SMP U-Net (efficientnet-b3, ImageNet encoder) on ALL 979 grouped-train
entries with per-entry binary union-mask targets -- the duplicate annotator
entries ARE the consensus mechanism (see src/data/semantic_dataset.py docstring).

Recipe: BCE+Dice loss, AMP, AdamW 2e-4 with cosine decay, random fg-biased
1024^2 crops with d4 augmentation, batch 4. Validation every EVAL_EVERY epochs
runs full-resolution 2048^2 inference on the FULL val split and scores PQ/Dice
through the same probs_to_instances post-processing the submission path uses
(threshold 0.5 / min_area 200 / non-empty guard -- the offline sweep script
refines these afterward on cached probability maps).

Best checkpoint (by val PQ) is saved to checkpoints/unet_v1.pt; the best
epoch's val probability maps are cached as float16 .npy files under
checkpoints/unet_v1_valprobs/ for scripts/sweep_postproc.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.coco_dataset import get_grouped_split
from src.data.semantic_dataset import SemanticFilamentDataset, semantic_collate
from src.eval.metrics import compute_dice_per_image, compute_panoptic_quality
from src.eval.postproc import probs_to_instances
from src.models.unet import build_unet

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
JSON_PATH = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGES_DIR = DATA_ROOT / "train" / "train_images"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "unet_v1.pt"
VALPROBS_DIR = REPO_ROOT / "checkpoints" / "unet_v1_valprobs"

SEED = 0
VAL_FRACTION = 0.15

ENCODER = "efficientnet-b3"
CROP_SIZE = 1024
FG_BIAS = 0.7
EPOCHS = 40
BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 2e-4
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 5  # epochs between full-res val evaluations (plus a final one)

# submission-path post-processing defaults used for in-training val scoring;
# scripts/sweep_postproc.py refines these on the cached probability maps
PP_THRESHOLD = 0.5
PP_MIN_AREA = 200

LOG_EVERY = 50


def bce_dice_loss(logits, targets, bce_fn):
    bce = bce_fn(logits, targets)
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum() + 1.0
    den = probs.sum() + targets.sum() + 1.0
    dice = 1.0 - num / den
    return bce + dice, float(bce.item()), float(dice.item())


@torch.no_grad()
def evaluate(model, val_loader, device, save_probs_dir=None):
    """Full-resolution inference on the val split; returns (mean PQ dict, mean dice,
    per-entry prob-map dict if save_probs_dir else None saved to disk)."""
    model.eval()
    pq_list, dice_list = [], []
    if save_probs_dir is not None:
        save_probs_dir.mkdir(parents=True, exist_ok=True)

    for images, targets, ids in val_loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.sigmoid(logits.float())

        for b in range(images.shape[0]):
            prob = probs[b, 0].cpu().numpy()
            gt = targets[b, 0].numpy().astype(np.uint8)

            # instance-level GT: rasterize each annotation separately
            entry = val_loader.dataset.entries_by_id[ids[b]]
            gt_instances = entry_instance_masks(entry)

            pred_instances = probs_to_instances(
                prob, threshold=PP_THRESHOLD, min_area=PP_MIN_AREA, non_empty_guard=True
            )
            pq_list.append(compute_panoptic_quality(gt_instances, pred_instances))
            dice_list.append(compute_dice_per_image([gt], pred_instances))

            if save_probs_dir is not None:
                np.save(save_probs_dir / f"{ids[b]}.npy", prob.astype(np.float16))

    pq = {
        "pq": float(np.mean([x["pq"] for x in pq_list])),
        "sq": float(np.mean([x["sq"] for x in pq_list])),
        "rq": float(np.mean([x["rq"] for x in pq_list])),
        "tp": int(sum(x["tp"] for x in pq_list)),
        "fp": int(sum(x["fp"] for x in pq_list)),
        "fn": int(sum(x["fn"] for x in pq_list)),
    }
    return pq, float(np.mean(dice_list))


def entry_instance_masks(entry):
    """Rasterize each annotation of an entry to its own HxW uint8 mask."""
    from pycocotools import mask as maskUtils

    h, w = entry["height"], entry["width"]
    out = []
    for ann in entry["anns"]:
        rles = maskUtils.frPyObjects(ann["segmentation"], h, w)
        out.append(maskUtils.decode(maskUtils.merge(rles)).astype(np.uint8))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ids, val_ids = get_grouped_split(JSON_PATH, val_fraction=VAL_FRACTION, seed=SEED)
    print(f"grouped split: train={len(train_ids)} val={len(val_ids)}")

    train_ds = SemanticFilamentDataset(
        JSON_PATH, IMAGES_DIR, image_ids=train_ids, crop_size=CROP_SIZE,
        train=True, fg_bias=FG_BIAS,
    )
    val_ds = SemanticFilamentDataset(JSON_PATH, IMAGES_DIR, image_ids=val_ids, train=False)
    # id -> entry lookup used by evaluate() for instance-level GT
    val_ds.entries_by_id = {e["id"]: e for e in val_ds.entries}

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=semantic_collate, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=semantic_collate,
    )

    model = build_unet(encoder_name=ENCODER).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: U-Net {ENCODER}, {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    bce_fn = torch.nn.BCEWithLogitsLoss()

    best_pq = -1.0
    history = []
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== training: {EPOCHS} epochs, bs={BATCH_SIZE}, {len(train_loader)} steps/epoch, "
          f"crop {CROP_SIZE}^2, AdamW lr={LR} cosine ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        loss_sum = bce_sum = dice_sum = 0.0
        n_steps = 0

        for step, (images, targets, _ids) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss, bce_v, dice_v = bce_dice_loss(logits, targets, bce_fn)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += float(loss.item())
            bce_sum += bce_v
            dice_sum += dice_v
            n_steps += 1

            if (step + 1) % LOG_EVERY == 0 or (step + 1) == len(train_loader):
                print(f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                      f"loss={loss_sum / n_steps:.4f} (bce={bce_sum / n_steps:.4f} "
                      f"dice={dice_sum / n_steps:.4f}) elapsed={time.time() - t0:.1f}s")

        epoch_time = time.time() - t0
        row = {"epoch": epoch, "loss": loss_sum / n_steps, "bce": bce_sum / n_steps,
               "dice_loss": dice_sum / n_steps, "epoch_time_s": epoch_time}

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            ev_t0 = time.time()
            pq, dice = evaluate(model, val_loader, device, save_probs_dir=None)
            row.update({"val_pq": pq["pq"], "val_sq": pq["sq"], "val_rq": pq["rq"],
                        "val_dice": dice})
            print(f"[epoch {epoch}/{EPOCHS}] loss={row['loss']:.4f} | "
                  f"val PQ={pq['pq']:.4f} SQ={pq['sq']:.4f} RQ={pq['rq']:.4f} "
                  f"Dice={dice:.4f} (TP={pq['tp']} FP={pq['fp']} FN={pq['fn']}) "
                  f"| eval {time.time() - ev_t0:.0f}s")

            if pq["pq"] > best_pq:
                best_pq = pq["pq"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "encoder": ENCODER,
                        "epoch": epoch,
                        "val_pq": pq["pq"], "val_sq": pq["sq"], "val_rq": pq["rq"],
                        "val_dice": dice,
                        "pp_threshold": PP_THRESHOLD, "pp_min_area": PP_MIN_AREA,
                        "history": history + [row],
                    },
                    CHECKPOINT_PATH,
                )
                print(f"  new best (PQ={best_pq:.4f}) -> saved {CHECKPOINT_PATH}")
        else:
            print(f"[epoch {epoch}/{EPOCHS}] loss={row['loss']:.4f} time={epoch_time:.1f}s")

        history.append(row)

    # reload the best checkpoint and cache its val probability maps for the sweep
    print(f"\n=== caching best-checkpoint val probability maps -> {VALPROBS_DIR} ===")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    pq, dice = evaluate(model, val_loader, device, save_probs_dir=VALPROBS_DIR)
    print(f"best checkpoint (epoch {ckpt['epoch']}): val PQ={pq['pq']:.4f} "
          f"SQ={pq['sq']:.4f} RQ={pq['rq']:.4f} Dice={dice:.4f}")
    print(f"cached {len(list(VALPROBS_DIR.glob('*.npy')))} probability maps")


if __name__ == "__main__":
    main()
