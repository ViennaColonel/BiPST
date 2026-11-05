"""Reference-based evaluation after model selection; no training or prompting."""

from pathlib import Path
import csv
import numpy as np
from .image_io import read_image, read_mask, write_image
from .network import predict_native, binary_metrics
from .workflow import load, load_student, image_paths, release, dump, METHODS
from .inference import choose_device


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def metrics(mask, reference):
    if mask.shape != reference.shape:
        raise ValueError("Mask and reference must cover the same complete image")
    return dict(
        **binary_metrics(mask, reference).as_dict(),
        coverage_error_pp=100 * abs(float(mask.mean()) - float(reference.mean())),
    )


def evaluate_checkpoint(checkpoint, images, references, output, ids=None, device="auto"):
    available = image_paths(images)
    chosen = ids if ids is not None else sorted(available)
    model = load_student(checkpoint, choose_device(device))
    rows = []
    for k in chosen:
        mask = predict_native(model, read_image(available[k])) >= 0.5
        ref = read_mask(Path(references) / (k + ".png")) > 127
        rows.append(dict(image=k, **metrics(mask, ref)))
        write_image(Path(output) / "masks" / (k + ".png"), mask.astype("uint8") * 255)
    save_csv(Path(output) / "per_image.csv", rows)
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "image"}
    dump(Path(output) / "summary.json", summary)
    print(summary)
    del model
    release()
    return rows


def evaluate_study(study):
    """Save all rounds and selected models, then average images within each seed."""
    out = study.out / "evaluation"
    groups = {
        "annotated_training_subset": study.splits.get("train_hidden_gt", []),
        "validation": study.val,
        "historical_test": study.test,
    }
    partition = {k: g for g, ids in groups.items() for k in ids}
    refs = {k: read_mask(study.paths["reference_masks"] / (k + ".png")) > 127 for k in partition}
    rows = []
    for method in METHODS:
        for seed in study.spec["seeds"]:
            record = study.out / f"runs/{method}/seed_{seed}/complete.json"
            if not record.exists():
                continue
            meta = load(record)
            for stage, cp in enumerate(meta["round_checkpoints"], 1):
                model = load_student(study.out / cp, study.device)
                for k in partition:
                    mask = (
                        predict_native(
                            model,
                            read_image(study.images[k]),
                            mixed_precision=study.cfg.mixed_precision,
                        )
                        >= study.cfg.threshold
                    )
                    rows.append(
                        dict(
                            method=method,
                            seed=seed,
                            stage=stage,
                            selected=stage == meta["selected_round"],
                            image=k,
                            partition=partition[k],
                            **metrics(mask, refs[k]),
                        )
                    )
                del model
                release()
    for method, folder in [("dtzsam", "initial"), ("refiner", "refined")]:
        if not (study.out / folder / "complete.json").exists():
            continue
        for k in partition:
            mask = read_mask(study.out / folder / "pseudo_labels" / (k + ".png")) > 127
            rows.append(
                dict(
                    method=method,
                    seed=None,
                    stage=0,
                    selected=True,
                    image=k,
                    partition=partition[k],
                    **metrics(mask, refs[k]),
                )
            )
    save_csv(out / "per_image.csv", rows)
    seed_rows, summary = [], []
    chosen = [r for r in rows if r["selected"]]
    keys = ["dice", "iou", "coverage_error_pp"]
    for method in sorted({r["method"] for r in chosen}):
        for group in groups:
            rr = [r for r in chosen if r["method"] == method and r["partition"] == group]
            if not rr:
                continue
            means = []
            for seed in sorted({str(r["seed"]) for r in rr}):
                rs = [r for r in rr if str(r["seed"]) == seed]
                value = {k: float(np.mean([r[k] for r in rs])) for k in keys}
                seed_rows.append(dict(method=method, partition=group, seed=seed, **value))
                means.append(value)
            summary.append(
                dict(
                    method=method,
                    partition=group,
                    **{k: float(np.mean([r[k] for r in means])) for k in keys},
                    seed_sd_dice=(
                        float(np.std([r["dice"] for r in means], ddof=1))
                        if len(means) > 1
                        else None
                    ),
                )
            )
    save_csv(out / "seed_results.csv", seed_rows)
    save_csv(out / "summary.csv", summary)
    print("Evaluation saved:", out)
