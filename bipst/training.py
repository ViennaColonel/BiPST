"""Train one fixed-supervision stage; prompt generation stays in workflow.py.

Only automatic SAM masks supply gradient targets. Native validation references
select checkpoints. Main and auxiliary fits use the same optimizer and loss.
"""

from dataclasses import asdict
from pathlib import Path
import csv
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PairedCropDataset
from .image_io import read_image, read_mask
from .network import build_student_model, CombinedSegmentationLoss, binary_metrics, predict_native


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Paper runs use zero workers; initialize opted-in workers reproducibly."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def fresh_student(encoder_weights, device):
    # Construct the complete backbone and decoder before loading encoder tensors.
    # This preserves the original random decoder initialization sequence.
    model = build_student_model(False)
    state = torch.load(encoder_weights, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(
        {k[len("features.") :]: v for k, v in state.items() if k.startswith("features.")},
        strict=True,
    )
    return model.to(device)


@torch.inference_mode()
def validate(model, images, mask_dir, threshold=0.5, mixed_precision=True):
    """Evaluate full native images, without gradients or random crops."""
    rows = []
    for path in images:
        image = read_image(path)
        ref = read_mask(Path(mask_dir) / (path.stem + ".png")) > 127
        if image.shape[:2] != ref.shape:
            raise ValueError(f"Validation dimensions differ: {path.name}")
        pred = predict_native(model, image, mixed_precision=mixed_precision) >= threshold
        rows.append(binary_metrics(pred, ref).as_dict())
    if not rows:
        raise ValueError("Validation references are required")
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}


def train_stage(
    config,
    device,
    seed,
    images,
    labels,
    validation_images,
    validation_masks,
    encoder_weights,
    output,
    initial=None,
    epochs=20,
    patience=21,
    repeats=2,
    stage=1,
):
    """Fresh AdamW per stage; warm-start model tensors only when initial is set.

    The original routine used an internal round index of 1 for every fit. Its RNG
    seed was therefore fit_seed+1. Preserve that offset. Main fits receive base
    seed, base seed+1, base seed+2; auxiliaries receive base seed+100*fold index.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed + 1)
    model = fresh_student(encoder_weights, device)
    if initial is not None:
        model.load_state_dict(
            torch.load(initial, map_location="cpu", weights_only=True)["model_state"]
        )
    dataset = PairedCropDataset(images, labels, config, repeats)
    generator = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        worker_init_fn=seed_worker,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = CombinedSegmentationLoss(config.bce_weight, config.dice_weight, 0.0)
    use_amp = device.type == "cuda" and config.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = output / "model_best.pt"
    best, best_epoch, stale = -math.inf, 0, 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (output / "training_log.csv").open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(
            log,
            fieldnames=[
                "stage",
                "epoch",
                "train_loss",
                "val_dice",
                "val_iou",
                "val_precision",
                "val_recall",
                "learning_rate",
            ],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            total, batches = 0.0, 0
            for x, y, _ in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, _ = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach())
                batches += 1
            metrics = validate(
                model, validation_images, validation_masks, config.threshold, config.mixed_precision
            )
            row = dict(
                stage=stage,
                epoch=epoch,
                train_loss=total / batches,
                **{"val_" + k: v for k, v in metrics.items()},
                learning_rate=config.learning_rate,
            )
            writer.writerow(row)
            log.flush()
            print(
                f'{output.name}: epoch {epoch}/{epochs}, loss={total/batches:.5f}, validation Dice={metrics["dice"]:.5f}',
                flush=True,
            )
            if metrics["dice"] > best + 1e-8:
                best, best_epoch, stale = metrics["dice"], epoch, 0
                torch.save(
                    dict(
                        architecture="MobileNetV3-Small-UNet",
                        model_state=model.state_dict(),
                        round=stage,
                        epoch=epoch,
                        validation=metrics,
                        training=asdict(config),
                        seed=seed,
                    ),
                    checkpoint,
                )
            else:
                stale += 1
            if stale >= patience:
                break
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    summary = dict(
        best_epoch=best_epoch,
        epochs_trained=epoch,
        validation=payload["validation"],
        views_per_epoch=len(dataset),
        view_instances=len(dataset) * epoch,
        optimizer_steps=batches * epoch,
        fit_seed=seed,
        rng_seed=seed + 1,
        stage=stage,
        wall_seconds=time.perf_counter() - started,
        peak_allocated_mib=(
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
    )
    return checkpoint, summary
