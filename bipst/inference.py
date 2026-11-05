"""Independent student inference and complete direct SAM application routes."""

from pathlib import Path
import csv
import json
import numpy as np
import torch
from .workflow import image_paths, load_student, release, dump
from .image_io import read_image, write_image
from .network import predict_native


def choose_device(name):
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu" if name == "auto" else name
    )


def inputs(source):
    source = Path(source)
    files = [source] if source.is_file() else list(image_paths(source).values())
    if not files:
        raise ValueError("No input images found: " + str(source))
    return files


def infer_images(
    checkpoint, source, output, device="auto", threshold=0.5, mixed_precision=True, overlay=False
):
    """Save native 0/255 masks; this function loads only the compact student."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0,1)")
    torch.set_num_threads(4)
    model = load_student(checkpoint, choose_device(device))
    output = Path(output)
    records = []
    for path in inputs(source):
        image = read_image(path)
        mask = predict_native(model, image, mixed_precision=mixed_precision) >= threshold
        write_image(output / "masks" / (path.stem + ".png"), mask.astype("uint8") * 255)
        if overlay:
            shown = image.copy()
            # User-selected cyan, converted from RGB to BGR for OpenCV.
            shown[mask] = (0.65 * image[mask] + 0.35 * np.array([224, 220, 170])).astype("uint8")
            write_image(output / "overlays" / (path.stem + ".png"), shown)
        records.append(
            dict(
                image=path.name,
                height=image.shape[0],
                width=image.shape[1],
                foreground_fraction=float(mask.mean()),
            )
        )
    dump(output / "predictions.json", records)
    del model
    release()
    print(f"Saved {len(records)} native-resolution masks to {output}")


def direct_images(
    source, output, sam_weights, method="dtzsam", samrefiner_repository=None, device="cuda"
):
    """Both routes start from the raw image, without precomputed input masks."""
    import cv2
    from segment_anything import sam_model_registry, SamPredictor
    from .dtzsam import direct_positive

    predictor = SamPredictor(
        sam_model_registry["vit_h"](checkpoint=str(sam_weights)).eval().to(choose_device(device))
    )
    for p in predictor.model.parameters():
        p.requires_grad_(False)
    refiner = None
    if method == "refiner":
        from .refiner import BinarySAMRefiner

        refiner = BinarySAMRefiner(predictor.model, samrefiner_repository)
    output = Path(output)
    records = []
    for path in inputs(source):
        image = read_image(path)
        mask = direct_positive(predictor, image)
        record = dict(image=path.name, status="ok")
        if refiner is not None:
            try:
                mask, details = refiner(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask, predictor.features
                )
                record.update(details)
            except Exception as error:
                record.update(status="failed_retained_input", error=repr(error))
        write_image(output / "masks" / (path.stem + ".png"), mask.astype("uint8") * 255)
        records.append(record)
    dump(output / "predictions.json", records)
    print(f"Saved {len(records)} {method} outputs to {output}")
