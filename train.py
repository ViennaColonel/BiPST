"""One-command training using the existing BiPST implementation.

Edit TRAIN_SETTINGS and PARAMETER_OVERRIDES below, then run this file in your IDE.
First-time users must supply their dataset, split manifest and pretrained weights.
Command-line options remain optional and override the values written here.

Relative configuration paths are based on this repository, even when launched elsewhere.
This entry point calls the training workflow in bipst/workflow.py directly.
"""

from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent

# ==================== EDIT YOUR TRAINING SETTINGS HERE ====================
# Paths are relative to this repository. Absolute paths are also supported.
# None selects configs/train.local.json if present, otherwise the public template.
TRAIN_SETTINGS = {
    "config": None,
    "method": "bipst",
    "seed": 3408,  # Use "all" for all seeds in the base config.
    "check_only": False,  # True checks inputs without fitting models.
}

# These values override the base JSON. Set a value to None to inherit the JSON.
# Dataset grouping is data metadata: provide it in the paths.splits JSON file.
PARAMETER_OVERRIDES = {
    "paths.images": None,  # Example: "data/images" or "E:/my_data/images".
    "paths.validation_masks": None,
    "paths.reference_masks": None,  # Optional; only used for later evaluation.
    "paths.splits": None,  # Example: "configs/splits.local.json".
    "paths.sam_weights": None,  # Generic SAM ViT-H checkpoint, not a student.
    "paths.encoder_weights": None,  # Generic ImageNet MobileNetV3-Small weights.
    "paths.output": None,  # Use a new directory when changing settings.
    "device": None,  # "cuda", "cpu", or inherit with None.
    "training.learning_rate": 0.001,
    "training.weight_decay": 0.0001,
    "training.batch_size": 2,
    "training.first_epochs": 100,
    "training.first_patience": 10,
    "training.later_epochs": 20,
    "training.auxiliary_epochs": 40,
    "training.auxiliary_patience": 8,
    "training.views_per_image": 2,
    "training.crop_size": 384,
    "training.num_workers": 0,
    "training.mixed_precision": True,
}
# ======================= END OF USER SETTINGS =============================


def repository_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def setup():
    for name in ("train", "splits"):
        target = ROOT / f"configs/{name}.local.json"
        if target.exists():
            print("Kept existing configuration:", target)
        else:
            shutil.copy2(ROOT / f"configs/{name}.example.json", target)
            print("Created:", target)
    print("Edit paths in train.local.json and image IDs/groups in splits.local.json.")
    print("Then run python train.py.")


def preflight(config, assignments):
    # Standard-library config validation happens before importing PyTorch or SAM.
    from bipst.config import read_config

    spec, _, paths, partitions = read_config(config, assignments)
    missing = []
    for key in ("images", "validation_masks", "sam_weights", "encoder_weights"):
        if not paths[key].exists():
            missing.append(f"{key}: {paths[key]}")
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n  " + "\n  ".join(missing))
    print("Configuration:", config, flush=True)
    print("Training images:", len(partitions["splits"]["train"]), flush=True)
    print("Output directory:", paths["output"], flush=True)
    return spec, paths


def create_study(config, assignments):
    # Load PyTorch only when checking real inputs or starting training.
    from bipst.workflow import Study

    return Study(config, assignments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup", action="store_true", help="Create local config templates; do not train"
    )
    default_config = TRAIN_SETTINGS["config"] or (
        "configs/train.local.json"
        if (ROOT / "configs/train.local.json").is_file()
        else "configs/train.example.json"
    )
    parser.add_argument("--config", default=default_config)
    parser.add_argument(
        "--method",
        choices=["bipst", "fixed", "refiner_student", "one_update", "probability_only", "all"],
        default=TRAIN_SETTINGS["method"],
    )
    parser.add_argument(
        "--seed",
        default=str(TRAIN_SETTINGS["seed"]),
        help="Integer seed, or all for the config's seeds",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=TRAIN_SETTINGS["check_only"],
        help="Validate real inputs without fitting models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected settings without loading data or weights",
    )
    args = parser.parse_args(argv)
    if args.setup:
        setup()
        return 0
    config = repository_path(args.config)
    # Later CLI assignments take precedence over settings at the top of this file.
    assignments = [
        key + "=" + json.dumps(value)
        for key, value in PARAMETER_OVERRIDES.items()
        if value is not None
    ] + args.set
    seed = None if args.seed == "all" else int(args.seed)
    # The one-update route branches from BiPST, so BiPST runs first in an all-method run.
    methods = (
        ["bipst", "fixed", "one_update", "probability_only", "refiner_student"]
        if args.method == "all"
        else [args.method]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config),
                    "methods": methods,
                    "seed": "all" if seed is None else seed,
                    "check_only": args.check,
                    "overrides": assignments,
                },
                indent=2,
            )
        )
        return 0
    if not config.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config}\nEdit TRAIN_SETTINGS or run python train.py --setup. See docs/QUICKSTART.md."
        )
    try:
        _, paths = preflight(config, assignments)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}\nSet your input paths in PARAMETER_OVERRIDES and supply a split JSON. Templates can be created with python train.py --setup."
        ) from error
    study = create_study(config, assignments)
    if args.check:
        print(
            json.dumps(
                {
                    "device": str(study.device),
                    "train_images": len(study.train),
                    "validation_images": len(study.val),
                    "test_images": len(study.test),
                    "main_views_per_epoch": len(study.train) * study.cfg.views_per_image,
                    "groups": [len(group) for group in study.folds],
                    "output": str(study.out),
                },
                indent=2,
            )
        )
        return 0
    seeds = study.spec["seeds"] if seed is None else [seed]
    for method in methods:
        for selected_seed in seeds:
            print("Selected student:", study.train_route(method, selected_seed))
    print("Training finished. Selected models are under:", paths["output"] / "runs")
    print("Use <method>/seed_<seed>/best_student.pt with demo.py --checkpoint.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Training setup: {error}", file=sys.stderr)
        raise SystemExit(2)
